# Document Classification Project (Step 1)

This step prepares a clean 5-class subset dataset:
- invoice
- form
- resume
- email
- budget

## Script

Use:

```bash
python scripts/prepare_five_class_dataset.py \
  --project-root . \
  --source-root /path/to/downloaded_dataset \
  --copy-mode symlink
```

Or download directly from Kaggle:

```bash
python scripts/prepare_five_class_dataset.py --project-root . --download
```

## Outputs

- `data/processed/metadata_all.csv`
- `data/processed/metadata_five_classes.csv`
- `data/processed/train.csv`
- `data/processed/val.csv`
- `data/processed/test.csv`
- `data/processed/five_class_subset/{train,val,test}/{class_name}/...`
- `data/processed/class_balance_by_split.csv`
- `data/processed/dataset_summary.json`

