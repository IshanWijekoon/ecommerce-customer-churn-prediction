# E-commerce Customer Churn Prediction

Predicting customer churn for an e-commerce platform using classical tree ensembles, **Harris Hawks Optimization (HHO)** for binary feature selection, and TabNet — with SHAP explainability and a Streamlit dashboard.

## Problem

Customer churn is costly: acquiring a new buyer typically costs more than retaining an existing one. This project frames churn as a **binary classification** problem on tabular behavioural and demographic features, prioritizing **Recall, F1, and PR-AUC** over Accuracy because the positive class is imbalanced (~17% churn).

## Repository structure

```
ecommerce-customer-churn-prediction/
├── data/
│   ├── raw/           # Original Excel (gitignored; see Dataset)
│   ├── interim/       # Validated snapshots
│   ├── processed/     # Model-ready arrays / parquet
│   └── external/      # Optional external sources
├── notebooks/         # Analysis pipeline (01–09)
├── src/
│   ├── data/          # Loaders & schema validation
│   ├── features/      # Feature engineering
│   ├── models/        # Baselines & trainers
│   ├── optimization/  # HHO feature selection (from scratch)
│   ├── evaluation/    # Metrics & plots
│   ├── explainability/# SHAP helpers
│   └── utils/         # Paths, config, seeding, logging
├── dashboard/         # Streamlit app (Phase 10)
├── models/            # Fitted artefacts (gitignored binaries)
├── reports/           # Figures & HTML profiles
├── environment.yml    # Conda environment (primary)
├── requirements.txt   # Pip fallback / CI
└── pyproject.toml     # Editable install of `src`
```

## Setup

### Conda (recommended)

```bash
conda env create -f environment.yml
conda activate ecommerce-churn
```

The environment installs this repo in editable mode (`pip install -e .`) so `import src...` works from notebooks and scripts.

### Pip fallback

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Quick smoke check

```bash
python -c "from src.data import load_ecommerce_data, load_data_dict; print(load_ecommerce_data().shape); print(load_data_dict().shape)"
```

## Dataset

| Item | Detail |
|------|--------|
| Source file (original name) | `E Commerce Dataset.xlsx` |
| Location in repo | `data/raw/E_Commerce_Dataset.xlsx` |
| Sheets | `Data Dict` (feature definitions), `E Comm` (observations) |

Raw Excel under `data/raw/` is **gitignored**. Place the file at `data/raw/E_Commerce_Dataset.xlsx` after cloning (or restore from your local copy of `E Commerce Dataset.xlsx`).

> Full documentation (architecture diagram, citations, screenshots, future work) will be completed in the final documentation phase.

## License

See [LICENSE](LICENSE).
