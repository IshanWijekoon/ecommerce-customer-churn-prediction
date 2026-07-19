# E-commerce Customer Churn Prediction

Predicting customer churn for an e-commerce platform using **Logistic Regression** and a small **Random Forest** as baselines, **Genetic Algorithm (GA)** and **Particle Swarm Optimization (PSO)** for binary feature selection, and **TabNet** as the final predictor — with SHAP explainability. All analysis lives in Jupyter notebooks (no `src` package, no dashboard).

## Problem

Customer churn is costly: acquiring a new buyer typically costs more than retaining an existing one. This project frames churn as a **binary classification** problem on tabular behavioural and demographic features, prioritizing **Recall, F1, and PR-AUC** over Accuracy because the positive class is imbalanced (~17% churn).

## Repository structure

```
ecommerce-customer-churn-prediction/
├── data/
│   ├── raw/           # Original Excel (gitignored; see Dataset)
│   ├── interim/       # Snapshots from early notebooks
│   ├── processed/     # Model-ready arrays / CSV
│   └── external/      # Optional external sources
├── notebooks/         # Analysis pipeline (01–09)
├── models/            # Fitted artefacts (preprocessor, TabNet)
├── reports/
│   └── figures/       # Exported charts & HTML profiles
├── environment.yml    # Conda environment (primary)
├── requirements.txt   # Pip fallback / CI
└── README.md
```

## Setup

### Conda (recommended)

```bash
conda env create -f environment.yml
conda activate ecommerce-churn
```

### Pip fallback

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### Quick smoke check

```bash
python -c "import pandas, sklearn, plotly, torch; from pytorch_tabnet.tab_model import TabNetClassifier; print('ok')"
```

## Notebook convention (header cell)

Every notebook starts with the same path + seed cell. Copy this into the first code cell:

```python
from pathlib import Path
import random

import numpy as np

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Walk up from the notebook cwd until we find the repo root
cwd = Path.cwd().resolve()
PROJECT_ROOT = next(
    (p for p in [cwd, *cwd.parents] if (p / "environment.yml").exists()),
    cwd,
)

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

print(f"Project root : {PROJECT_ROOT}")
print(f"Random seed  : {RANDOM_SEED}")
```

Run notebooks **01 → 09** in order after placing the dataset under `data/raw/`.

| # | Notebook | Purpose |
|---|----------|---------|
| 01 | Data Understanding | Schema, DQ, feature dictionary, churn balance |
| 02 | Exploratory Data Analysis | Deep EDA + K-means / hierarchical segmentation |
| 03 | Data Preprocessing | Impute, encode, scale, FE, train/test split |
| 04 | Baseline Models | Logistic Regression + small Random Forest |
| 05 | GA Feature Selection | Binary Genetic Algorithm (from scratch) |
| 06 | PSO Feature Selection | Binary PSO + GA vs PSO winner |
| 07 | TabNet Model | Attentive tabular ANN on winning subset |
| 08 | SHAP Explainability | Global + local explanations |
| 09 | Business Insights | Retention plays + synthesis |

## Dataset

| Item | Detail |
|------|--------|
| Source | [Ankit Verma — E-commerce Customer Churn Analysis and Prediction (Kaggle)](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction) |
| Source file (original name) | `E Commerce Dataset.xlsx` |
| Location in repo | `data/raw/E_Commerce_Dataset.xlsx` |
| Sheets | `Data Dict` (feature definitions), `E Comm` (observations) |
| Size | ~5,630 customers × 20 columns |

Raw Excel under `data/raw/` is **gitignored**. Place the file at `data/raw/E_Commerce_Dataset.xlsx` after cloning.

## Reproduce

```bash
conda env create -f environment.yml
conda activate ecommerce-churn
# Windows + PyTorch OpenMP quirk (if needed):
# set KMP_DUPLICATE_LIB_OK=TRUE
jupyter notebook
# Run notebooks/01 → notebooks/09 in order
```

### Observed pipeline summary (seed=42)

After a full run on this machine:

| Stage | Result |
|-------|--------|
| Churn rate | ~16.8% |
| Baselines | LR + small RF (`reports/04_baseline_metrics.json`) |
| GA vs PSO winner | **GA** (17 features; see `data/processed/nia_feature_selection_winner.json`) |
| TabNet (winning subset) | Strong F1 / PR-AUC vs all-features baseline (`reports/07_tabnet_metrics.json`) |

Exact numbers vary slightly by hardware but the artefacts under `reports/` and `data/processed/` are the source of truth after you re-run.

## Nature-inspired algorithms

- **GA (notebook 05):** Binary chromosomes = feature masks; tournament selection, crossover, mutation, elitism. Fitness = stratified CV F1 of a small RF minus a sparsity penalty.
- **PSO (notebook 06):** Continuous positions mapped to binary masks via a sigmoid transfer function; personal/global best updates. Same fitness as GA for a fair comparison.
- **TabNet (notebook 07):** Trained on the winning feature subset (and once on all features for a short before/after table).

## Research references

1. Holland, J. H. (1992). *Adaptation in Natural and Artificial Systems*. MIT Press. (Genetic Algorithms)
2. Kennedy, J., & Eberhart, R. (1995). Particle Swarm Optimization. *IEEE ICNN*.
3. Arik, S. Ö., & Pfister, T. (2021). TabNet: Attentive Interpretable Tabular Learning. *AAAI*.
4. Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*.
5. Verma, A. (2020). E-commerce Customer Churn Analysis and Prediction [Dataset]. Kaggle.

## License

See [LICENSE](LICENSE).
