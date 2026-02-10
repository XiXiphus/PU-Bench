# PU-Bench

A unified open-source benchmark for **Positive-Unlabeled (PU) learning**.

PU-Bench provides a standardized framework for evaluating PU learning algorithms under consistent conditions, covering data generation, training, and evaluation in a single reproducible pipeline. It currently integrates **18 state-of-the-art methods** across **8 datasets** spanning text, image, and tabular modalities.

> **Paper**: *PU-Bench: A Unified Benchmark for Rigorous and Reproducible PU Learning* (ICLR 2026)

---

## Table of Contents

- [PU-Bench](#pu-bench)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Dependency Notes](#dependency-notes)
  - [Quick Start](#quick-start)
  - [Project Structure](#project-structure)
  - [Configuration System](#configuration-system)
    - [Dataset Configuration](#dataset-configuration)
    - [Method Configuration](#method-configuration)
  - [Core Concepts](#core-concepts)
    - [Data Sampling Schemes](#data-sampling-schemes)
    - [Labeling Mechanisms (SCAR / SAR)](#labeling-mechanisms-scar--sar)
    - [Evaluation Metrics](#evaluation-metrics)
  - [Supported Methods \& Datasets](#supported-methods--datasets)
    - [Methods (18)](#methods-18)
    - [Datasets (8)](#datasets-8)
  - [How to Extend PU-Bench](#how-to-extend-pu-bench)
    - [Adding a New PU Method](#adding-a-new-pu-method)
    - [Adding a New Labeling Strategy (SAR)](#adding-a-new-labeling-strategy-sar)
    - [Adding a New Dataset](#adding-a-new-dataset)
  - [Results \& Outputs](#results--outputs)
  - [Contributing](#contributing)
  - [License](#license)

---

## Installation

```bash
git clone https://github.com/XiXiphus/PU-Bench.git
cd PU-Bench
pip install -r requirements.txt
```

Key dependencies: PyTorch, torchvision, scikit-learn, sentence-transformers, pyyaml, rich, faiss-cpu.

---

## Dependency Notes

- `requirements.txt` is intentionally **unpinned** to keep installation flexible and avoid version-locking across platforms.
- In our experience, installing the **latest stable versions** from `requirements.txt` works for normal usage.
- If you encounter environment-specific issues, please open an issue or PR with your platform, Python/PyTorch versions, and error log.

---

## Quick Start

**Run a single method on one dataset (conventional setting):**

```bash
python run_train.py \
  --dataset-config config/datasets_typical/param_sweep_mnist.yaml \
  --methods nnpu
```

**Run multiple methods:**

```bash
python run_train.py \
  --dataset-config config/datasets_typical/param_sweep_cifar10.yaml \
  --methods nnpu vpu distpu p3mixc
```

**Run all methods on all datasets:**

```bash
python run_train.py \
  --dataset-config config/datasets_typical/param_sweep_mnist.yaml \
                    config/datasets_typical/param_sweep_cifar10.yaml \
                    config/datasets_typical/param_sweep_imdb_sbert.yaml
# omit --methods to run all 18 methods
```

**Preview planned runs (dry run):**

```bash
python run_train.py \
  --dataset-config config/datasets_vary_c/param_sweep_mnist.yaml \
  --methods nnpu vpu --dry-run
```

---

## Project Structure

```
PU-Bench/
│
├── run_train.py                 # Main entry point
│
├── config/
│   ├── methods/                 # Per-method hyperparameter YAMLs (18 files)
│   │   ├── nnpu.yaml
│   │   ├── vpu.yaml
│   │   └── ...
│   ├── datasets_typical/        # Conventional setting (cc, SCAR, c=0.1)
│   ├── datasets_vary_c/         # Varying label ratio c
│   ├── datasets_vary_e/         # Varying labeling mechanism (SAR)
│   ├── method_loader.py         # YAML loader for method configs
│   └── run_param_sweep.py       # Expands dataset config into run grid
│
├── data/
│   ├── data_utils.py            # PU data generation core (SCAR/SAR, SS/CC)
│   ├── MNIST_PU.py              # Dataset-specific loader
│   ├── CIFAR10_PU.py
│   └── ...                      # One loader per dataset
│
├── backbone/
│   ├── models.py                # Standard CNN/MLP backbones
│   ├── mix_models.py            # P3Mix-specific backbones
│   ├── meta_models.py           # LaGAM meta-learning backbones
│   ├── vaepu_models.py          # VAE-PU generative models
│   ├── cgenpu_models.py         # CGenPU GAN models
│   └── puet/                    # PU Extra Trees
│
├── loss/
│   ├── loss_nnpu.py             # nnPU / uPU loss
│   ├── loss_vpu.py              # VPU variational loss
│   ├── loss_distpu.py           # Dist-PU distribution alignment loss
│   └── ...                      # One file per loss function
│
├── train/
│   ├── base_trainer.py          # Abstract base class for all methods
│   ├── train_utils.py           # Evaluation, model selection, checkpointing
│   ├── nnpu_trainer.py          # nnPU trainer implementation
│   └── ...                      # One trainer per method
│
└── results/                     # Auto-generated outputs
    └── seed_{seed}/
        ├── {experiment}.json    # Metrics, timing, memory, hyperparameters
        └── logs/                # Per-run training logs
```

---

## Configuration System

PU-Bench is fully **configuration-driven**. All experiments are defined by two YAML files: one for the dataset and one for the method. No code changes are needed to adjust experimental conditions.

### Dataset Configuration

Located in `config/datasets_typical/`, `config/datasets_vary_c/`, and `config/datasets_vary_e/`.

```yaml
# config/datasets_typical/param_sweep_mnist.yaml

dataset_class: MNIST
data_dir: ./datasets
random_seeds: [2, 25, 42, 52, 99, 103, 250, 666, 777, 2026]

c_values: [0.1]                        # Label ratio(s): fraction of positives that are labeled
scenarios: [case-control]              # "single" (SS) or "case-control" (CC)
selection_strategies: ["random"]       # Labeling mechanism (see below)

val_ratio: 0.01                        # Fraction of training data used for validation
target_prevalence: null                # Set to override natural class prior π
with_replacement: true                 # Sampling with replacement for CC

label_scheme:                          # Multi-class → binary mapping
  positive_classes: [0, 2, 4, 6, 8]   # Even digits are "positive"
  negative_classes: [1, 3, 5, 7, 9]   # Odd digits are "negative"
```

**Grid expansion**: The launcher computes the Cartesian product of `random_seeds × c_values × scenarios × selection_strategies`. Each combination becomes one independent training run.

To sweep label ratios, simply list multiple values:

```yaml
# config/datasets_vary_c/param_sweep_mnist.yaml
c_values: [0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
```

To evaluate SAR labeling mechanisms:

```yaml
# config/datasets_vary_e/param_sweep_mnist.yaml
c_values: [0.05, 0.5]
selection_strategies: ["sar_pusb", "sar_lbeA", "sar_lbeB"]
```

| Field                  | Description                                                                 |
| ---------------------- | --------------------------------------------------------------------------- |
| `dataset_class`        | Dataset name; must match a loader in `data/`                                |
| `random_seeds`         | List of seeds for reproducibility (10 seeds recommended)                    |
| `c_values`             | Label frequency *c* = P(S=1 \| Y=1); fraction of positives labeled          |
| `scenarios`            | `"single"` = single-training-set (SS), `"case-control"` = case-control (CC) |
| `selection_strategies` | `"random"` (SCAR), `"sar_pusb"` (S4), `"sar_lbeA"` (S2), `"sar_lbeB"` (S3)  |
| `label_scheme`         | Maps original classes to binary positive/negative                           |

### Method Configuration

Located in `config/methods/`. One YAML file per method.

```yaml
# config/methods/nnpu.yaml

nnpu:
  optimizer: "adam"
  lr: 0.0003
  weight_decay: 0.0001
  batch_size: 256
  num_epochs: 40
  seed: 42

  # nnPU-specific hyperparameters
  gamma: 1.0           # Non-negative risk gradient weight
  beta: 0.0            # Regularization term

  label_scheme:        # How PU labels are encoded for this method
    true_positive_label: 1
    true_negative_label: 0
    pu_labeled_label: 1       # Labeled positive → +1
    pu_unlabeled_label: -1    # Unlabeled → -1

  checkpoint:
    enabled: true
    save_model: false
    monitor: "val_f1"         # Metric to monitor for best checkpoint
    mode: "max"
    early_stopping:
      enabled: true
      patience: 10
      min_delta: 0.0001
```

| Field                       | Description                                                                              |
| --------------------------- | ---------------------------------------------------------------------------------------- |
| `optimizer`                 | `"adam"` or `"sgd"`                                                                      |
| `lr`, `weight_decay`        | Learning rate and L2 regularization                                                      |
| `batch_size`, `num_epochs`  | Training configuration                                                                   |
| `label_scheme`              | How ±1 / 0/1 labels are assigned to PU data                                              |
| `checkpoint.monitor`        | Metric for model selection (`val_f1`, `val_accuracy`, etc.)                              |
| `checkpoint.early_stopping` | Stop training when monitored metric stalls                                               |
| *(method-specific)*         | Any additional keys are passed to the Trainer (e.g., `gamma`, `beta`, `pretrain_epochs`) |

---

## Core Concepts

### Data Sampling Schemes

PU-Bench supports two standard PU data generation scenarios:

| Scheme                       | Config Value     | Description                                                                                                                 |
| ---------------------------- | ---------------- | --------------------------------------------------------------------------------------------------------------------------- |
| **Single-training-set (SS)** | `"single"`       | A fraction *c* of positives are labeled; the rest (positives + negatives) form the unlabeled set U. Dataset size unchanged. |
| **Case-control (CC)**        | `"case-control"` | Labeled positives LP are sampled from P, then returned to form U = P ∪ N. U preserves the population class mixture.         |

### Labeling Mechanisms (SCAR / SAR)

The `selection_strategies` field controls *how* positives are selected for labeling:

| Strategy       | Config Value | Mechanism                           | Propensity e(x)           |
| -------------- | ------------ | ----------------------------------- | ------------------------- |
| **S1 — SCAR**  | `"random"`   | Uniform random selection            | Constant: e(x) = c        |
| **S2 — LBE-A** | `"sar_lbeA"` | Favors high-posterior positives     | e(x) ∝ p̂(x)^k, k=10       |
| **S3 — LBE-B** | `"sar_lbeB"` | Favors boundary/ambiguous positives | e(x) ∝ (1.5 + δ − p̂(x))^k |
| **S4 — PUSB**  | `"sar_pusb"` | Deterministic top-scoring selection | Top-N by p̂(x)^α, α=20     |

For SAR strategies (S2–S4), an auxiliary logistic regression classifier is first trained on the full PN data to compute p̂(x), which is then used to derive instance-dependent propensity scores.

### Evaluation Metrics

All metrics are computed on a held-out ground-truth test set:

| Metric      | Description                                            |
| ----------- | ------------------------------------------------------ |
| `accuracy`  | Overall classification accuracy                        |
| `precision` | TP / (TP + FP)                                         |
| `recall`    | TP / (TP + FN)                                         |
| `f1`        | Macro-F1 score (harmonic mean of precision and recall) |
| `auc`       | Area under the ROC curve                               |
| `risk`      | Unbiased PU risk estimate (zero-one surrogate)         |

Model selection uses the **validation macro-F1** by default (configurable via `checkpoint.monitor`).

---

## Supported Methods & Datasets

### Methods (18)

| Category                                 | Methods                                                                                                                                    |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| **Risk-Minimization Estimators**         | nnPU, PUSB (nnpusb), VPU, MPE-PU (bbepu), LBE-PU (lbe), PUET, Dist-PU (distpu), PULDA                                                      |
| **Disambiguation-Guided Supervised ERM** | Self-PU (selfpu), P3Mix-C (p3mixc), P3Mix-E (p3mixe), Robust-PU (robustpu), Holistic-PU (holisticpu), LaGAM-PU (lagam), PUL-CPBF (pulcpbf) |
| **Generative Distribution Matching**     | VAE-PU (vaepu), PAN (pan), CGenPU (cgenpu)                                                                                                 |

The name in parentheses is the CLI identifier used in `--methods` and the corresponding YAML filename.

A fully supervised **PN** baseline (`pn`) is also available as an oracle reference.

### Datasets (8)

| Modality | Dataset   | Input             | Positive vs. Negative            |
| -------- | --------- | ----------------- | -------------------------------- |
| Text     | IMDb      | 384-d SBERT       | Positive vs. Negative sentiment  |
| Text     | 20News    | 384-d SBERT       | Topics 0–3 vs. 4–6               |
| Image    | MNIST     | 28×28 grayscale   | Even digits vs. Odd digits       |
| Image    | F-MNIST   | 28×28 grayscale   | Tops/Outerwear vs. Footwear/Bags |
| Image    | CIFAR-10  | 32×32×3 color     | Vehicles vs. Animals             |
| Image    | ADNI      | 128×128 grayscale | NonDemented vs. Demented         |
| Tabular  | Connect-4 | 126-d one-hot     | Win vs. Loss/Draw                |
| Tabular  | Spambase  | 57-d numeric      | Spam vs. Not Spam                |

Text datasets are pre-encoded into 384-d dense vectors using [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). Backbone models (CNN / MLP) are automatically selected based on the dataset.

---

## How to Extend PU-Bench

### Adding a New PU Method

PU-Bench uses a **Trainer** abstraction. Each method is a subclass of `BaseTrainer` that implements exactly two methods: `create_criterion()` and `train_one_epoch()`.

**Step 1: Create a method config** — `config/methods/mymethod.yaml`

```yaml
mymethod:
  optimizer: "adam"
  lr: 0.001
  weight_decay: 0.0001
  batch_size: 256
  num_epochs: 50

  # Your method-specific hyperparameters
  temperature: 0.5
  alpha: 1.0

  label_scheme:
    true_positive_label: 1
    true_negative_label: 0
    pu_labeled_label: 1
    pu_unlabeled_label: -1

  checkpoint:
    enabled: true
    save_model: false
    monitor: "val_f1"
    mode: "max"
    early_stopping:
      enabled: true
      patience: 10
      min_delta: 0.0001
```

**Step 2: Implement the Trainer** — `train/mymethod_trainer.py`

```python
"""mymethod_trainer.py"""
import torch
from .base_trainer import BaseTrainer


class MyMethodTrainer(BaseTrainer):

    def create_criterion(self):
        # Return your loss function. self.prior holds the class prior π.
        # Access method-specific hyperparameters from self.params.
        temperature = self.params.get("temperature", 0.5)
        return MyCustomLoss(prior=self.prior, temperature=temperature)

    def train_one_epoch(self, epoch_idx: int):
        self.model.train()
        for x, t, y_true, idx, pseudo in self.train_loader:
            # x:       input features          [B, ...]
            # t:       PU labels               [B]  (+1 = labeled positive, -1 = unlabeled)
            # y_true:  ground-truth labels      [B]  (for debugging only, DO NOT use in training)
            # idx:     sample indices           [B]
            # pseudo:  pseudo-label scores      [B]
            x, t = x.to(self.device), t.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(x).view(-1)
            loss = self.criterion(outputs, t)
            loss.backward()
            self.optimizer.step()
```

**Step 3: (Optional) Create a custom loss** — `loss/loss_mymethod.py`

```python
"""loss_mymethod.py"""
import torch
import torch.nn as nn


class MyCustomLoss(nn.Module):
    def __init__(self, prior: float, temperature: float = 0.5):
        super().__init__()
        self.prior = prior
        self.temperature = temperature

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # outputs: model logits  [B]
        # targets: PU labels     [B]  (+1 labeled positive, -1 unlabeled)
        positive_mask = (targets == 1)
        unlabeled_mask = (targets == -1)
        # ... your loss computation ...
        return loss
```

**Step 4: Register in `run_train.py`**

Add one line to the `TRAINER_IMPORT_PATHS` dictionary:

```python
TRAINER_IMPORT_PATHS = {
    # ... existing methods ...
    "mymethod": "train.mymethod_trainer.MyMethodTrainer",
}
```

**Step 5: Run it**

```bash
python run_train.py \
  --dataset-config config/datasets_typical/param_sweep_cifar10.yaml \
  --methods mymethod
```

> **What `BaseTrainer` handles for you**: data loading, model/optimizer creation, per-epoch evaluation on train/val/test sets, metric logging (Rich tables + log files), checkpoint saving, early stopping, GPU memory tracking, and JSON result export. You only write the training logic.

---

### Adding a New Labeling Strategy (SAR)

To add a new instance-dependent labeling mechanism, modify `data/data_utils.py`:

**Step 1: Add the strategy logic** in `create_pu_training_set()`

```python
# In data/data_utils.py, inside create_pu_training_set():

# 1. Register your strategy name in the SAR pre-computation block:
if selection_strategy in ["sar_pusb", "sar_lbeA", "sar_lbeB", "sar_mynew"]:
    flat_features = features.reshape(features.shape[0], -1)
    pn_probs = compute_pn_scores(flat_features, labels)

# 2. Add an elif branch for your strategy:
elif selection_strategy == "sar_mynew":
    scores = pn_probs[pos_indices]
    # Define your custom propensity function e(x).
    # Example: favor mid-range posterior positives (near decision boundary)
    weights = np.exp(-((scores - 0.5) ** 2) / 0.1)
    weights = np.maximum(weights, 0)
    p = weights / weights.sum()
    labeled_pos_idx = np.random.choice(
        pos_indices, size=n_labeled, replace=False, p=p
    )
```

**Step 2: Use it in a dataset config**

```yaml
selection_strategies: ["sar_mynew"]
```

The rest of the pipeline (data loading, training, evaluation) requires no changes.

---

### Adding a New Dataset

**Step 1: Create a dataset loader** — `data/MyDataset_PU.py`

Follow the existing loaders as a template. Your loader function must:
1. Load raw data and produce binary labels (0/1).
2. Split into train/validation/test.
3. Call `create_pu_training_set()` to generate PU labels.
4. Return three `PUDataset` objects.

```python
"""MyDataset_PU.py"""
import numpy as np
from data.data_utils import PUDataset, create_pu_training_set, split_train_val


def load_mydataset_pu(config: dict):
    # 1. Load your data
    features, labels = ...  # np.ndarray, labels ∈ {0, 1}

    # 2. Train/test split (use your own logic or sklearn)
    train_f, test_f, train_y, test_y = ...

    # 3. Train/validation split
    train_f, train_y, val_f, val_y = split_train_val(
        train_f, train_y, val_ratio=config.get("val_ratio", 0.01)
    )

    # 4. Generate PU labels on training set
    train_f, train_y, labeled_mask = create_pu_training_set(
        features=train_f,
        labels=train_y,
        labeled_ratio=config["labeled_ratio"],
        selection_strategy=config.get("selection_strategy", "random"),
        scenario=config.get("scenario", "case-control"),
    )

    # 5. Convert to PU label encoding
    pu_labeled = config["label_scheme"]["pu_labeled_label"]      # typically +1
    pu_unlabeled = config["label_scheme"]["pu_unlabeled_label"]  # typically -1
    pu_labels = np.where(labeled_mask == 1, pu_labeled, pu_unlabeled)

    # 6. Return PUDataset objects
    train_ds = PUDataset(train_f, pu_labels, train_y)
    val_ds   = PUDataset(val_f, np.full(len(val_y), pu_unlabeled), val_y)
    test_ds  = PUDataset(test_f, np.full(len(test_y), pu_unlabeled), test_y)
    return train_ds, val_ds, test_ds
```

**Step 2: Register in `train/train_utils.py`**

In the `prepare_loaders()` function, add a branch for your dataset:

```python
if dataset_class == "MyDataset":
    from data.MyDataset_PU import load_mydataset_pu
    train_ds, val_ds, test_ds = load_mydataset_pu(data_config)
```

Also add model selection logic in `select_model()` to assign a backbone.

**Step 3: Create dataset configs**

```yaml
# config/datasets_typical/param_sweep_mydataset.yaml
dataset_class: MyDataset
data_dir: ./datasets
random_seeds: [42]
c_values: [0.1]
scenarios: [case-control]
selection_strategies: ["random"]
val_ratio: 0.01
label_scheme:
  positive_classes: [1]
  negative_classes: [0]
```

---

## Results & Outputs

After each run, results are saved to `results/seed_{seed}/{experiment_name}.json`:

```json
{
  "experiment": "MNIST_case-control_random_c0.1_seed42",
  "runs": {
    "nnpu": {
      "method": "nnpu",
      "timing": {
        "start": "2026-01-15T10:00:00",
        "end": "2026-01-15T10:05:30",
        "duration_seconds": 330.0
      },
      "max_gpu_memory_bytes": 524288000,
      "best": {
        "epoch": 25,
        "metrics": {
          "train_accuracy": 0.9512,
          "test_accuracy": 0.9485,
          "test_f1": 0.9470,
          "test_auc": 0.9823,
          "test_precision": 0.9501,
          "test_recall": 0.9440
        }
      },
      "hyperparameters": {"...": "..."}
    }
  }
}
```

Multiple methods running on the same dataset config are merged into the same JSON file under separate keys. Training logs are saved to `results/seed_{seed}/logs/`.

---

## Contributing

PU-Bench is an actively maintained project and we welcome contributions from the community:

- **New PU methods**: If you have developed a new PU learning algorithm, we encourage you to submit a pull request to integrate it into the benchmark. Follow the [Adding a New PU Method](#adding-a-new-pu-method) guide above to get started.
- **Improvements to existing methods**: If you find bugs, performance issues, or have better implementations for any of the currently integrated methods, PRs for corrections and improvements are equally welcome.
- **New datasets or labeling strategies**: Extensions to the benchmark's coverage are always appreciated.

Please ensure your contribution includes the corresponding YAML config, follows the existing code style, and passes basic sanity checks on at least one dataset before submitting. We will review and merge PRs on a rolling basis to keep PU-Bench up to date with the latest advances in PU learning.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
