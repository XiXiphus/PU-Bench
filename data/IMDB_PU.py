import os
from typing import Tuple, List

import numpy as np

from .data_utils import (
    PUDataset,
    split_train_val,
    create_pu_training_set,
    print_dataset_statistics,
    resample_by_prevalence,
)


def _load_imdb_texts_via_hf(
    root_dir: str,
) -> tuple[list[str], list[int], list[str], list[int]]:
    """Load IMDB using HuggingFace datasets. This avoids torchtext binary deps.

    Downloads to cache_dir under the provided root_dir to keep project-contained.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        raise ImportError(
            "datasets not installed. Please `pip install datasets` or provide ./datasets/aclImdb"
        ) from e

    cache_dir = os.path.join(root_dir, "hf_cache")
    ds = load_dataset("imdb", cache_dir=cache_dir)

    train_texts = list(ds["train"]["text"])  # type: ignore
    train_labels = [int(x) for x in ds["train"]["label"]]  # type: ignore
    test_texts = list(ds["test"]["text"])  # type: ignore
    test_labels = [int(x) for x in ds["test"]["label"]]  # type: ignore

    return train_texts, train_labels, test_texts, test_labels


def _load_imdb_texts_via_torchtext(
    root_dir: str,
) -> tuple[list[str], list[int], list[str], list[int]]:
    # torchtext 0.14+ API
    from torchtext.datasets import IMDB

    train_iter = IMDB(root=root_dir, split="train")
    test_iter = IMDB(root=root_dir, split="test")

    def _to_lists(it):
        texts, labels = [], []
        for label, text in it:
            texts.append(text)
            labels.append(1 if str(label).lower().strip() == "pos" else 0)
        return texts, labels

    train_texts, train_labels = _to_lists(train_iter)
    test_texts, test_labels = _to_lists(test_iter)
    return train_texts, train_labels, test_texts, test_labels


def _load_imdb_texts_from_local(
    root_dir: str, auto_download: bool = False
) -> tuple[list[str], list[int], list[str], list[int]]:
    base = os.path.join(root_dir, "aclImdb")
    if not (os.path.isdir(base)):
        if auto_download:
            # Try auto-download from Stanford if missing (disabled by default)
            try:
                url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
                os.makedirs(root_dir, exist_ok=True)
                tar_path = os.path.join(root_dir, "aclImdb_v1.tar.gz")
                import urllib.request
                import tarfile

                print(f"Downloading IMDB dataset to {tar_path} ...")
                urllib.request.urlretrieve(url, tar_path)
                print("Extracting IMDB dataset ...")
                with tarfile.open(tar_path, "r:gz") as tar:
                    tar.extractall(root_dir)
                try:
                    os.remove(tar_path)
                except Exception:
                    pass
            except Exception as e:
                raise FileNotFoundError(
                    f"IMDB local folder not found under {base} and auto-download failed: {e}"
                )
        else:
            raise FileNotFoundError(
                f"IMDB local folder not found under {base}. "
                f"Please download and extract aclImdb_v1.tar.gz to {root_dir}"
            )

    train_dir = os.path.join(base, "train")
    test_dir = os.path.join(base, "test")
    if not (os.path.isdir(train_dir) and os.path.isdir(test_dir)):
        raise FileNotFoundError(f"IMDB local folder not found under {base}")

    def _read_split(split_dir: str) -> tuple[list[str], list[int]]:
        texts, labels = [], []
        for sentiment, label in [("pos", 1), ("neg", 0)]:
            sdir = os.path.join(split_dir, sentiment)
            if not os.path.isdir(sdir):
                continue
            for fname in sorted(os.listdir(sdir)):
                if not fname.endswith(".txt"):
                    continue
                fpath = os.path.join(sdir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        txt = f.read()
                except Exception:
                    with open(fpath, "r", encoding="latin-1", errors="ignore") as f:
                        txt = f.read()
                texts.append(txt)
                labels.append(label)
        return texts, labels

    train_texts, train_labels = _read_split(train_dir)
    test_texts, test_labels = _read_split(test_dir)
    if len(train_texts) == 0 or len(test_texts) == 0:
        raise RuntimeError("IMDB local folder is present but contains no data")
    return train_texts, train_labels, test_texts, test_labels


def load_imdb_pu(
    data_dir: str = "./",
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
) -> Tuple[PUDataset, PUDataset, PUDataset]:
    rng = np.random.RandomState(random_seed)

    # 1) Load raw texts and labels
    # Priority: local folder → local tar.gz → HuggingFace → torchtext
    train_texts: list[str]
    train_labels: list[int]
    test_texts: list[str]
    test_labels: list[int]
    
    # Check if local data exists first
    local_aclimdb_path = os.path.join(data_dir, "aclImdb")
    local_tar_path = os.path.join(data_dir, "aclImdb_v1.tar.gz")
    
    if os.path.isdir(local_aclimdb_path) and os.path.exists(os.path.join(local_aclimdb_path, "train", "pos")):
        # Use local extracted data
        print(f"Using local IMDB data from {local_aclimdb_path}")
        train_texts, train_labels, test_texts, test_labels = (
            _load_imdb_texts_from_local(data_dir)
        )
    elif os.path.isfile(local_tar_path) and os.path.getsize(local_tar_path) > 1000000:
        # Local tar.gz exists and is not empty (>1MB), extract it
        print(f"Found local IMDB archive {local_tar_path}, extracting...")
        import tarfile
        with tarfile.open(local_tar_path, "r:gz") as tar:
            tar.extractall(data_dir)
        print(f"Extracted to {local_aclimdb_path}")
        train_texts, train_labels, test_texts, test_labels = (
            _load_imdb_texts_from_local(data_dir)
        )
    else:
        # Fallback to HuggingFace or download
        print("Local IMDB data not found, trying HuggingFace...")
        try:
            train_texts, train_labels, test_texts, test_labels = _load_imdb_texts_via_hf(
                data_dir
            )
        except Exception:
            # Last resort: auto-download from Stanford
            print("HuggingFace failed, falling back to Stanford download...")
            train_texts, train_labels, test_texts, test_labels = (
                _load_imdb_texts_from_local(data_dir)
            )

    # 2) Use SBERT embeddings - MUST be precomputed
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
                train_texts, batch_size=256, show_progress_bar=True, convert_to_numpy=True
            ).astype(np.float32)
            print("   Computing test embeddings...")
            X_test = model.encode(
                test_texts, batch_size=256, show_progress_bar=True, convert_to_numpy=True
            ).astype(np.float32)
            
            # Save for future use
            if sbert_embeddings_path:
                print(f"   Saving embeddings to {sbert_embeddings_path}...")
                os.makedirs(os.path.dirname(sbert_embeddings_path), exist_ok=True)
                np.savez_compressed(
                    sbert_embeddings_path,
                    train_embeddings=X_train,
                    test_embeddings=X_test,
                    train_labels=np.array(train_labels, dtype=np.int32),
                    test_labels=np.array(test_labels, dtype=np.int32),
                )
                print("   ✅ Embeddings saved!")
            else:
                print("   ⚠️  No sbert_embeddings_path specified; skipping save.")
            
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Please run: pip install sentence-transformers"
            )

    # L2 normalize SBERT embeddings to stabilize across batches
    from numpy.linalg import norm

    def _l2_normalize(mat: np.ndarray) -> np.ndarray:
        d = np.maximum(norm(mat, axis=1, keepdims=True), 1e-12)
        return (mat / d).astype(np.float32)

    X_train = _l2_normalize(X_train)
    X_test = _l2_normalize(X_test)

    y_train_bin = np.array(train_labels, dtype=int)
    y_test_bin = np.array(test_labels, dtype=int)

    X_train, y_train_bin, X_val, y_val_bin = split_train_val(
        X_train, y_train_bin, val_ratio, random_state=random_seed
    )

    if target_prevalence is not None and target_prevalence > 0:
        X_test, y_test_bin = resample_by_prevalence(
            X_test, y_test_bin, target_prevalence, random_seed
        )

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

    print_dataset_statistics(
        train_dataset,
        val_dataset,
        test_dataset,
        train_labeled_mask,
        positive_classes=[],
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
