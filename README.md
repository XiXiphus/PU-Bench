

<p align="center">
  <img width="521" alt="PU-Bench" src="https://github.com/user-attachments/assets/a5e7ce0b-496b-43f8-8804-5b26804679e9" />
</p>
<p align="center">
  <strong>A unified open-source benchmark for rigorous and reproducible Positive-Unlabeled learning.</strong>
</p>

<p align="center">
  <a href="https://openreview.net/forum?id=tb8DabMbMq">
    <img src="https://img.shields.io/badge/Paper-ICLR%202026-1f4b99?style=flat-square" alt="ICLR 2026 Paper">
  </a>
  <a href="https://github.com/XiXiphus/PU-Bench/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-00a896?style=flat-square" alt="MIT License">
  </a>
  <a href="https://github.com/XiXiphus/PU-Bench">
    <img src="https://img.shields.io/badge/Python-%E2%89%A53.11-3776ab?style=flat-square&logo=python&logoColor=white" alt="Python >=3.11">
  </a>
  <a href="https://github.com/astral-sh/uv">
    <img src="https://img.shields.io/badge/managed%20with-uv-5c4ee5?style=flat-square" alt="managed with uv">
  </a>
</p>

<p align="center">
  <a href="#installation">Installation</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#supported-methods--datasets">Methods & Datasets</a> ·
  <a href="#results--outputs">Results</a> ·
  <a href="#citation">Citation</a>
</p>

---

PU-Bench provides a standardized framework for evaluating PU learning algorithms under consistent conditions, covering data generation, training, and evaluation in a single reproducible pipeline.

It currently integrates **18 PU methods** plus a fully supervised **PN oracle baseline** across datasets spanning **text**, **image**, and **tabular** modalities.

> **Paper:** [PU-Bench: A Unified Benchmark for Rigorous and Reproducible PU Learning](https://openreview.net/forum?id=tb8DabMbMq) — ICLR 2026  
> **Related benchmark:** PU-Bench supports the PU-only validation metrics `PA` and `PAUC` introduced in [Accessible, Realistic, and Fair Evaluation of Positive-Unlabeled Learning Algorithms](https://openreview.net/forum?id=5R11h5o44C) — ICLR 2026, exposed here as `proxy_acc` and `proxy_auc`. For the authors' official benchmark release, see [wu-dd/PUBench](https://github.com/wu-dd/PUBench).

---

## Table of Contents

- [PU-Bench](#pu-bench)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Dependency Management](#dependency-management)
  - [Quick Start](#quick-start)
  - [Verification](#verification)
  - [Project Structure](#project-structure)
  - [Configuration System](#configuration-system)
    - [Dataset Configuration](#dataset-configuration)
    - [Method Configuration](#method-configuration)
  - [Core Concepts](#core-concepts)
    - [Data Sampling Schemes](#data-sampling-schemes)
    - [Labeling Mechanisms (SCAR / SAR)](#labeling-mechanisms-scar--sar)
    - [Evaluation Metrics](#evaluation-metrics)
  - [Supported Methods \& Datasets](#supported-methods--datasets)
    - [Methods](#methods)
    - [Datasets](#datasets)
  - [How to Extend PU-Bench](#how-to-extend-pu-bench)
    - [Adding a New PU Method](#adding-a-new-pu-method)
    - [Adding a New Labeling Strategy (SAR)](#adding-a-new-labeling-strategy-sar)
    - [Adding a New Dataset](#adding-a-new-dataset)
  - [Results \& Outputs](#results--outputs)
  - [Contributing](#contributing)
  - [License](#license)
  - [Citation](#citation)

---

## Installation

```bash
git clone https://github.com/XiXiphus/PU-Bench.git
cd PU-Bench
uv sync --locked
```

Dependencies are managed only through `pyproject.toml` and `uv.lock`. Use
`uv run` for training commands so they execute inside the locked project
environment.

---

## Dependency Management

- PU-Bench uses `uv` as the dependency manager. `pyproject.toml` is the single source for dependency metadata, and `uv.lock` records the resolved versions used for reproducible installs.
- Run `uv sync --locked` after cloning. Run experiments with `uv run python -u run_train.py ...`.
- When dependencies change, edit `pyproject.toml`, run `uv lock`, then commit both `pyproject.toml` and `uv.lock`.
- If you encounter environment-specific issues, please open an issue or PR with your platform, Python, PyTorch, `uv` version, and error log.

---

## Quick Start

**Run a single method on one dataset (conventional setting):**

```bash
uv run python -u run_train.py \
  --dataset-config config/datasets_typical/param_sweep_mnist.yaml \
  --methods nnpu
```

**Run multiple methods:**

```bash
uv run python -u run_train.py \
  --dataset-config config/datasets_typical/param_sweep_cifar10.yaml \
  --methods nnpu vpu distpu p3mixc
```

**Run all methods on all datasets:**

```bash
uv run python -u run_train.py \
  --dataset-config config/datasets_typical/param_sweep_mnist.yaml \
                   config/datasets_typical/param_sweep_cifar10.yaml \
                   config/datasets_typical/param_sweep_imdb_sbert.yaml
# omit --methods to run all available methods
```

**Preview planned runs (dry run):**

```bash
uv run python -u run_train.py \
  --dataset-config config/datasets_vary_c/param_sweep_mnist.yaml \
  --methods nnpu vpu --dry-run
```

Add `--plan-json /tmp/pu_bench_plan.json` to export the expanded run plan.

By default, checkpointing and early stopping use `val_proxy_acc` on a PU-structured validation split. If you prefer AUC-oriented model selection, set `checkpoint.monitor: "val_proxy_auc"` in the corresponding method YAML.

---

## Verification

For a fast local sanity check after editing configs or trainers:

```bash
uv run python -m unittest discover -s tests
uv run python scripts/check_arch_contracts.py
uv run python -u run_train.py \
  --dataset-config config/datasets_smoke/param_sweep_mnist_seed2.yaml \
  --methods nnpu nnpusb --dry-run --plan-json /tmp/pu_bench_plan.json
```

---

## Project Structure

```
PU-Bench/
│
├── run_train.py                 # Main entry point
│
├── config/
│   ├── methods/                 # Per-method hyperparameter YAMLs
│   │   ├── nnpu.yaml
│   │   ├── vpu.yaml
│   │   └── ...
│   ├── datasets_typical/        # Conventional setting (cc, SCAR, c=0.1)
│   ├── datasets_vary_c/         # Varying label ratio c
│   ├── datasets_vary_e/         # Varying labeling mechanism (SAR)
│   ├── method_loader.py         # YAML loader for method configs
│   ├── experiment_plan.py       # Explicit run planning and validation
│   ├── schema_adapter.py        # Flat/namespaced config compatibility
│   └── run_param_sweep.py       # Expands dataset config into run grid
│
├── data/
│   ├── data_utils.py            # PU data generation core (SCAR/SAR, SS/CC)
│   ├── MNIST_PU.py              # Dataset-specific loader
│   ├── CIFAR10_PU.py
│   └── ...                      # One loader per dataset
│
├── backbone/
│   └── models.py                # Shared public CNN/MLP backbones
│
├── train/
│   ├── base_trainer.py          # Abstract base class for all methods
│   ├── data_factory.py          # PU DataLoader construction
│   ├── model_factory.py         # Shared public backbone selection
│   ├── metrics.py               # Oracle/proxy evaluation helpers
│   ├── checkpointing.py         # Checkpoint and bundle helpers
│   ├── registry.py              # Method-to-trainer dispatch table
│   ├── common/                  # Shared PU risk/loss primitives
│   ├── bbepu/                   # BBE-PU trainer, estimator, and method loss
│   ├── cgenpu/                  # CGenPU trainer and private GAN models
│   ├── distpu/                  # Dist-PU trainer, loss, mixup, pseudo labels
│   ├── lbe/                     # LBE-PU EM trainer and source-aligned formulas
│   ├── nnpu/                    # nnPU trainer and package-local loss exports
│   ├── nnpusb/                  # nnPUSB trainer and source-mapped risk
│   ├── p3mix/                   # P3Mix-C/E trainers and private mixup models
│   ├── puet/                    # PU Extra Trees package
│   ├── robustpu/                # Robust-PU trainer, SPL utilities, private models
│   ├── vpu/                     # VPU trainer and variational loss
│   └── ...                      # One trainer per method
│
└── results/                     # Auto-generated outputs
    └── seed_{seed}/
        ├── {experiment}.json    # Best metrics, timing, dataset stats, hyperparameters
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

val_ratio: 0.01                        # Fraction of PU training data used for validation
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
| `val_ratio`            | Fraction of the PU-structured training partition held out for validation    |
| `label_scheme`         | Maps original classes to binary positive/negative                           |

### Method Configuration

Located in `config/methods/`. One YAML file per method.

```yaml
# config/methods/nnpu.yaml

nnpu:
  metadata:
    method_key: "nnpu"
    trainer_key: "nnpu"
    alignment:
      level: "source_faithful_kernel_benchmark_wrapper"
      source_reproduction: false

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
    monitor: "val_proxy_acc"  # Default PU-only model-selection metric
    mode: "max"
    early_stopping:
      enabled: true
      patience: 10
      min_delta: 0.0001
```

| Field                       | Description                                                                                             |
| --------------------------- | ------------------------------------------------------------------------------------------------------- |
| `optimizer`                 | `"adam"` or `"sgd"`                                                                                     |
| `lr`, `weight_decay`        | Learning rate and L2 regularization                                                                     |
| `batch_size`, `num_epochs`  | Training configuration                                                                                  |
| `label_scheme`              | How ±1 / 0/1 labels are assigned to PU data                                                             |
| `checkpoint.monitor`        | Fully qualified metric key for model selection (e.g. `val_proxy_acc`, `val_proxy_auc`, `val_oracle_f1`) |
| `checkpoint.early_stopping` | Stop training when monitored metric stalls                                                              |
| `metadata.method_key`       | Stable method identifier recorded in plans and result files                                             |
| `metadata.trainer_key`      | Registry key used to resolve the trainer class                                                          |
| `metadata.alignment`        | Optional source-alignment note distinguishing source-faithful kernels from benchmark adaptations         |
| *(method-specific)*         | Any additional keys are passed to the Trainer (e.g., `gamma`, `beta`, `pretrain_epochs`)                |

Metric names follow the pattern `<split>_<metric>`. Oracle metrics are prefixed with `oracle_`; PU-only proxy metrics are prefixed with `proxy_`. Method metadata is copied into expanded plan JSONs and per-run result summaries so source-faithful and benchmark-adapted entries can be audited without inspecting YAML by hand.

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
| **S4 — PUSB**  | `"sar_pusb"` | Source-style selected-bias accept-reject rule | Mean/max-normalized p̂(x) |

For SAR strategies (S2–S4), an auxiliary logistic regression classifier is first trained on the full PN data to compute p̂(x), which is then used to derive instance-dependent propensity scores.

### Evaluation Metrics

PU-Bench reports two metric families:

| Family | Keys                                                                              | Split(s)           | Uses true labels?  | Primary use                             |
| ------ | --------------------------------------------------------------------------------- | ------------------ | ------------------ | --------------------------------------- |
| Oracle | `oracle_accuracy`, `oracle_precision`, `oracle_recall`, `oracle_f1`, `oracle_auc` | train / val / test | Yes                | Benchmark analysis and oracle ablations |
| Proxy  | `proxy_acc`, `proxy_auc`                                                          | train / val        | No, only PU labels | Realistic model selection               |

- `proxy_acc` is the Proxy Accuracy (PA) criterion from [Wang et al. (ICLR 2026)](https://openreview.net/forum?id=5R11h5o44C). In this repository, `scenario: "single"` uses the paper's OS-style formula, while `scenario: "case-control"` uses the TS-style formula.
- `proxy_auc` corresponds to Proxy AUC (PAUC). In logs and result files, the paper's PAUC metric is exposed as `proxy_auc`.
- `proxy_acc` uses the class prior `π`; `proxy_auc` does not.
- Proxy metrics are intentionally not computed on the test split.
- By default, model selection uses `val_proxy_acc`. To optimize for ranking quality instead, set `checkpoint.monitor: "val_proxy_auc"`.

---

## Supported Methods & Datasets

### Methods

| Category                                 | Methods                                                                                                                                    |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| **Risk-Minimization Estimators**         | nnPU, PUSB (nnpusb), VPU, MPE-PU (bbepu), LBE-PU (lbe), PUET, Dist-PU (distpu), PULDA                                                      |
| **Disambiguation-Guided Supervised ERM** | Self-PU (selfpu), P3Mix-C (p3mixc), P3Mix-E (p3mixe), Robust-PU (robustpu), Holistic-PU (holisticpu), LaGAM-PU (lagam), PUL-CPBF (pulcpbf) |
| **Generative Distribution Matching**     | VAE-PU (vaepu), PAN (pan), CGenPU (cgenpu)                                                                                                 |

The name in parentheses is the CLI identifier used in `--methods` and the corresponding YAML filename.

A fully supervised **PN** baseline (`pn`) is also available as an oracle reference.

### Datasets

| Modality | Dataset      | Input             | Positive vs. Negative                  |
| -------- | ------------ | ----------------- | -------------------------------------- |
| Text     | IMDb         | 384-d SBERT       | Positive vs. Negative sentiment        |
| Text     | 20News       | 384-d SBERT       | `alt/comp/misc/rec` vs. `sci/soc/talk` |
| Image    | MNIST        | 28×28 grayscale   | Even digits vs. Odd digits             |
| Image    | F-MNIST      | 28×28 grayscale   | Tops/Outerwear vs. Footwear/Bags       |
| Image    | CIFAR-10     | 32×32×3 color     | Vehicles vs. Animals                   |
| Image    | AlzheimerMRI | 128×128 grayscale | NonDemented vs. Demented               |
| Tabular  | Connect-4    | 126-d one-hot     | Win vs. Loss/Draw                      |
| Tabular  | Spambase     | 57-d numeric      | Spam vs. Not Spam                      |
| Tabular  | Mushrooms    | Dense one-hot     | Poisonous vs. Edible                   |

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
    monitor: "val_proxy_acc"
    mode: "max"
    early_stopping:
      enabled: true
      patience: 10
      min_delta: 0.0001
```

**Step 2: Implement the Trainer** — `train/mymethod/trainer.py`

```python
"""train/mymethod/trainer.py"""
import torch
from ..base_trainer import BaseTrainer


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

**Step 3: (Optional) Create a method-local loss** — `train/mymethod/losses.py`

```python
"""train/mymethod/losses.py"""
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

**Step 4: Register in `train/registry.py`**

Add one line to the `TRAINER_IMPORT_PATHS` dictionary:

```python
TRAINER_IMPORT_PATHS = {
    # ... existing methods ...
    "mymethod": "train.mymethod.trainer.MyMethodTrainer",
}
```

**Step 5: Run it**

```bash
uv run python -u run_train.py \
  --dataset-config config/datasets_smoke/param_sweep_mnist_seed2.yaml \
  --methods mymethod
```

> **What `BaseTrainer` handles for you**: data loading, model/optimizer creation, per-epoch evaluation on train/val/test sets, oracle and proxy metric logging, checkpoint saving, early stopping, GPU memory tracking, and JSON result export. You only write the training logic.

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
2. Split into train/test.
3. Call `create_pu_training_set()` on the training partition.
4. Split validation **after PU labeling** with `split_pu_val()` so the validation set preserves the labeled-positive vs. unlabeled structure.
5. Return three `PUDataset` objects.

```python
"""MyDataset_PU.py"""
import numpy as np
from data.data_utils import PUDataset, create_pu_training_set, split_pu_val


def load_mydataset_pu(config: dict):
    # 1. Load your data
    features, labels = ...  # np.ndarray, labels ∈ {0, 1}

    # 2. Train/test split (use your own logic or sklearn)
    train_f, test_f, train_y, test_y = ...

    # 3. Generate PU labels on the full training partition
    pu_features, pu_true_labels, labeled_mask = create_pu_training_set(
        features=train_f,
        labels=train_y,
        labeled_ratio=config["labeled_ratio"],
        selection_strategy=config.get("selection_strategy", "random"),
        scenario=config.get("scenario", "case-control"),
    )

    # 4. Split validation AFTER PU labeling to preserve PU structure
    (
        pu_train_f,
        pu_train_y,
        train_labeled_mask,
        pu_val_f,
        pu_val_y,
        val_labeled_mask,
    ) = split_pu_val(
        pu_features,
        pu_true_labels,
        labeled_mask,
        val_ratio=config.get("val_ratio", 0.01),
        random_state=config.get("random_seed", 42),
    )

    # 5. Convert masks to method-specific PU label encoding
    pu_labeled = config["label_scheme"]["pu_labeled_label"]      # typically +1
    pu_unlabeled = config["label_scheme"]["pu_unlabeled_label"]  # typically -1
    train_pu_labels = np.where(train_labeled_mask == 1, pu_labeled, pu_unlabeled)
    val_pu_labels = np.where(val_labeled_mask == 1, pu_labeled, pu_unlabeled)

    # 6. Return PUDataset objects
    train_ds = PUDataset(pu_train_f, train_pu_labels, pu_train_y)
    val_ds = PUDataset(pu_val_f, val_pu_labels, pu_val_y)
    test_ds = PUDataset(test_f, test_y, test_y)
    return train_ds, val_ds, test_ds
```

**Step 2: Register in the training factories**

In `train/data_factory.py`, add a `prepare_loaders()` branch for your dataset:

```python
if dataset_class == "MyDataset":
    from data.MyDataset_PU import load_mydataset_pu
    train_ds, val_ds, test_ds = load_mydataset_pu(data_config)
```

In `train/model_factory.py`, add public benchmark backbone selection only. Put
method-private or adapter-specific selectors under the method package.

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
  "updated_at": "2026-03-23T09:30:00Z",
  "runs": {
    "nnpu": {
      "method": "nnpu",
      "monitor": "val_proxy_acc",
      "timing": {
        "start": "2026-01-15T10:00:00",
        "end": "2026-01-15T10:05:30",
        "duration_seconds": 330.0
      },
      "max_gpu_memory_bytes": 524288000,
      "best": {
        "epoch": 25,
        "metrics": {
          "train_oracle_accuracy": 0.9512,
          "train_proxy_acc": 0.9031,
          "val_oracle_f1": 0.9448,
          "val_proxy_acc": 0.8917,
          "val_proxy_auc": 0.9624,
          "test_oracle_auc": 0.9823
        }
      },
      "global_epochs": 40,
      "hyperparameters": {"...": "..."}
    }
  }
}
```

Metrics inside `best.metrics` use fully qualified names such as `val_proxy_acc` or `test_oracle_f1`. Multiple methods running on the same dataset config are merged into the same JSON file under separate keys. Training logs are saved to `results/seed_{seed}/logs/`.

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

---

## Citation

If you use PU-Bench in your work, please cite:

```bibtex
@inproceedings{
chen2026pubench,
title={{PU}-{BENCH}: A {UNIFIED} {BENCHMARK} {FOR} {RIGOROUS} {AND} {REPRODUCIBLE} {PU} {LEARNING}},
author={Qiuyi Chen and Haiyang Zhang and Leqi Zhang and Changchun Li and Jia Wang and Wei Wang},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=tb8DabMbMq}
}
```
