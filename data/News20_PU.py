import os
from typing import Tuple, List

import numpy as np
from sklearn.datasets import fetch_20newsgroups

from .data_utils import (
    PUDataset,
    split_train_val,
    create_pu_training_set,
    print_dataset_statistics,
    resample_by_prevalence,
)


def _prefix_from_category(cat: str) -> str:
    return cat.split(".")[0].lower()


def load_20news_pu(
    data_dir: str = "datasets/",
    n_labeled: int | None = None,
    labeled_ratio: float = 0.2,
    val_ratio: float = 0.2,
    target_prevalence: float | None = None,
    selection_strategy: str = "random",
    scenario: str = "single",
    case_control_mode: str = "naive_mode",
    random_seed: int = 42,
    true_positive_label: int = 1,
    true_negative_label: int = 0,
    pu_labeled_label: int = 1,
    pu_unlabeled_label: int = -1,
    with_replacement: bool = True,
    print_stats: bool = True,
    dataset_log_file: str | None = None,
    # SBERT settings
    sbert_model_name: str = "all-MiniLM-L6-v2",
    sbert_embeddings_path: str | None = None,
    sbert_model_path: str | None = None,
    positive_classes: List[str] | None = None,
    negative_classes: List[str] | None = None,
) -> Tuple[PUDataset, PUDataset, PUDataset]:
    """
    Load and preprocess 20 Newsgroups dataset for PU learning.

    Positive class (P): categories whose prefix is in {'alt', 'comp', 'misc', 'rec'}
    Negative class (N): categories whose prefix is in {'sci', 'soc', 'talk'}

    Text preprocessing: Uses Sentence-BERT embeddings (precomputed or on-the-fly)

    Returns:
        (train_dataset, val_dataset, test_dataset) as PUDataset objects
    """
    rng = np.random.RandomState(random_seed)

    pos_prefixes = set((positive_classes or ["alt", "comp", "misc", "rec"]))
    neg_prefixes = set((negative_classes or ["sci", "soc", "talk"]))

    data_home = os.path.join(data_dir, "20news_sklearn")

    # Download 20 Newsgroups
    train_bunch = fetch_20newsgroups(
        subset="train",
        data_home=data_home,
        remove=(),
        shuffle=True,
        random_state=random_seed,
    )
    test_bunch = fetch_20newsgroups(
        subset="test",
        data_home=data_home,
        remove=(),
        shuffle=True,
        random_state=random_seed,
    )

    # Map category -> binary label based on prefix sets
    def map_targets_to_binary(
        target: np.ndarray, target_names: List[str]
    ) -> np.ndarray:
        binary = np.zeros_like(target, dtype=int)
        for i, t in enumerate(target):
            prefix = _prefix_from_category(target_names[int(t)])
            if prefix in pos_prefixes:
                binary[i] = 1
            elif prefix in neg_prefixes:
                binary[i] = 0
            else:
                # Default unknown prefixes to negative
                binary[i] = 0
        return binary

    y_train_bin = map_targets_to_binary(train_bunch.target, train_bunch.target_names)
    y_test_bin = map_targets_to_binary(test_bunch.target, test_bunch.target_names)

    # Use SBERT embeddings - MUST be precomputed
    if sbert_embeddings_path and os.path.exists(sbert_embeddings_path):
        print(f"Loading precomputed SBERT embeddings from {sbert_embeddings_path}")
        embeddings_data = np.load(sbert_embeddings_path)
        X_train = embeddings_data["train_embeddings"].astype(np.float32)
        X_test = embeddings_data["test_embeddings"].astype(np.float32)
    else:
        # Auto-compute and save embeddings if not found
        print(f"⚠️  Precomputed embeddings not found at {sbert_embeddings_path}")
        print("   Auto-computing embeddings...")
        
        try:
            from sentence_transformers import SentenceTransformer
            
            # Use local model or download
            if sbert_model_path and os.path.isdir(sbert_model_path):
                model = SentenceTransformer(sbert_model_path)
                print(f"   Using local model: {sbert_model_path}")
            else:
                model_name = sbert_model_name.replace("sentence-transformers/", "")
                print(f"   Using model: {model_name}")
                model = SentenceTransformer(model_name)
            
            # Compute embeddings
            print("   Computing train embeddings...")
            X_train = model.encode(
                train_bunch.data, batch_size=256, show_progress_bar=True, convert_to_numpy=True
            ).astype(np.float32)
            print("   Computing test embeddings...")
            X_test = model.encode(
                test_bunch.data, batch_size=256, show_progress_bar=True, convert_to_numpy=True
            ).astype(np.float32)
            
            # Save for future use
            if sbert_embeddings_path:
                print(f"   Saving embeddings to {sbert_embeddings_path}...")
                os.makedirs(os.path.dirname(sbert_embeddings_path), exist_ok=True)
                np.savez_compressed(
                    sbert_embeddings_path,
                    train_embeddings=X_train,
                    test_embeddings=X_test,
                    train_labels=train_bunch.target.astype(np.int32),
                    test_labels=test_bunch.target.astype(np.int32),
                )
                print("   ✅ Embeddings saved!")
            else:
                print("   ⚠️  No sbert_embeddings_path specified; skipping save.")
            
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Please run: pip install sentence-transformers"
            )

    # L2 normalize SBERT embeddings to stabilize scale across batches
    from numpy.linalg import norm

    def _l2_normalize(mat: np.ndarray) -> np.ndarray:
        d = np.maximum(norm(mat, axis=1, keepdims=True), 1e-12)
        return (mat / d).astype(np.float32)

    X_train = _l2_normalize(X_train)
    X_test = _l2_normalize(X_test)

    # Train/Val split
    X_train, y_train_bin, X_val, y_val_bin = split_train_val(
        X_train, y_train_bin, val_ratio, random_state=random_seed
    )

    # Optional test prevalence adjustment
    if target_prevalence is not None and target_prevalence > 0:
        X_test, y_test_bin = resample_by_prevalence(
            X_test, y_test_bin, target_prevalence, random_seed
        )

    # Create PU training set (features kept dense, shape (N, D))
    pu_train_features, pu_train_true_labels_01, train_labeled_mask = (
        create_pu_training_set(
            X_train,
            y_train_bin,
            n_labeled=n_labeled,
            labeled_ratio=labeled_ratio,
            selection_strategy=selection_strategy,
            scenario=scenario,
            with_replacement=with_replacement,
            case_control_mode=case_control_mode,
        )
    )

    # Map to final label coding
    final_pu_train_true_labels = np.full_like(
        pu_train_true_labels_01, true_negative_label
    )
    final_pu_train_true_labels[pu_train_true_labels_01 == 1] = true_positive_label
    final_val_labels = np.full_like(y_val_bin, true_negative_label)
    final_val_labels[y_val_bin == 1] = true_positive_label
    final_test_labels = np.full_like(y_test_bin, true_negative_label)
    final_test_labels[y_test_bin == 1] = true_positive_label
    final_pu_train_labels = np.full(
        len(pu_train_true_labels_01), pu_unlabeled_label, dtype=int
    )
    final_pu_train_labels[train_labeled_mask == 1] = pu_labeled_label

    # Build datasets
    train_dataset = PUDataset(
        features=pu_train_features,
        pu_labels=final_pu_train_labels,
        true_labels=final_pu_train_true_labels,
    )
    val_dataset = PUDataset(
        features=X_val,
        pu_labels=final_val_labels,
        true_labels=final_val_labels,
    )
    test_dataset = PUDataset(
        features=X_test,
        pu_labels=final_test_labels,
        true_labels=final_test_labels,
    )

    # Stats logging
    print_dataset_statistics(
        train_dataset,
        val_dataset,
        test_dataset,
        train_labeled_mask,
        positive_classes=[],  # Not applicable for text; prefixes documented above
        negative_classes=[],
        true_positive_label=true_positive_label,
        true_negative_label=true_negative_label,
        pu_labeled_label=pu_labeled_label,
        pu_unlabeled_label=pu_unlabeled_label,
        val_ratio=val_ratio,
        log_file=dataset_log_file,
        also_print=print_stats,
    )

    return train_dataset, val_dataset, test_dataset
