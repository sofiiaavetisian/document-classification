# Document Classification Project

Current status: environment + dataset preparation are implemented.

## Task 1: Environment Setup

### Install required dependencies

```bash
python -m pip install -r requirements.txt
```

### Optional dependencies

```bash
python -m pip install -r requirements-optional.txt
```

### Conda environment alternative

```bash
conda env create -f environment.yml
conda activate doccls-ocr-ml
```

### Validate environment (includes Tesseract check)

```bash
python scripts/check_environment.py
```

If Tesseract is missing, install it and rerun the check:
- macOS: `brew install tesseract`
- Ubuntu/Debian: `sudo apt-get update && sudo apt-get install -y tesseract-ocr`
- Windows: install Tesseract (UB Mannheim build), add it to `PATH`

## Task 2: Dataset Preparation (already implemented)

Target classes:
- invoice
- form
- resume
- email
- budget

Use existing local dataset:

```bash
python scripts/prepare_five_class_dataset.py \
  --project-root . \
  --source-root /path/to/downloaded_dataset \
  --copy-mode symlink \
  --compute-hash
```

Or download directly from Kaggle:

```bash
python scripts/prepare_five_class_dataset.py \
  --project-root . \
  --download \
  --copy-mode symlink \
  --compute-hash
```

Outputs:
- `data/processed/metadata_all.csv`
- `data/processed/metadata_five_classes.csv`
- `data/processed/train.csv`
- `data/processed/val.csv`
- `data/processed/test.csv`
- `data/processed/five_class_subset/{train,val,test}/{class_name}/...`
- `data/processed/class_balance_by_split.csv`
- `data/processed/dataset_summary.json`
