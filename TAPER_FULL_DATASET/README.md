# TAPER_FULL_DATASET

This folder is intentionally left empty in the GitHub repository.

The full TAPER dataset is too large to host directly on GitHub. To use the repository, download the full dataset from the external dataset hosting page:

**Full dataset link:** https://kaggle.com/datasets/a952369ec83dc30512b58f18418af3e6652276c775f51f9b3eb2e67b55d6661c

After downloading, place the dataset contents inside this folder so that the expected structure is available to the preprocessing script.

Expected structure:

```text
TAPER_FULL_DATASET/
├── raw/
│   ├── 1/
│   ├── 2/
│   ├── ...
│   └── 11/
└── preprocessed/
    ├── rings/
    └── tape_thick/
```

The `raw/` directory contains the original XIMU3 recordings and tap pad data. These files can be used with `taps_extraction.py` to regenerate tap-level processed CSV files.

The `preprocessed/` directory contains the processed tap sequences used for the model evaluation reported in the paper.

Participant folders are anonymised using numeric IDs from `1` to `11`.