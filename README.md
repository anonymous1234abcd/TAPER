# TAPER: Tap-interaction Evaluation Resource

This repository contains the code used to preprocess and evaluate the **TAPER** dataset: a high-resolution tap-interaction dataset for time-series classification under varying sensing conditions.

The dataset contains tap interactions recorded using an IMU sensor mounted on the finger. Each tap belongs to one of four surface classes, collected from manufactured resin cylindrical surfaces with different hardness levels. The code in this repository supports two main workflows:

1. **Model evaluation** on the released preprocessed dataset.
2. **Tap extraction** from the raw IMU and tap-pad recordings for users who want to inspect, modify, or extend the preprocessing pipeline.

The repository is anonymised for review. Please do not add author, institution, or identifying information while the associated submission is under double-blind review.

---

## Full dataset

The full dataset is **not stored directly in this GitHub repository** because of its size. Instead, this repository includes an empty placeholder folder named:

```text
TAPER_FULL_DATASET/
```

Download the full dataset from:

```text
https://www.kaggle.com/datasets/a952369ec83dc30512b58f18418af3e6652276c775f51f9b3eb2e67b55d6661c
```

After downloading and extracting the dataset, place its contents inside the `TAPER_FULL_DATASET/` folder so that the local repository structure matches the paths expected by the notebook and preprocessing script.

---

## Repository contents

```text
.
├── evaluation_notebook.ipynb      # Model evaluation notebook
├── taps_extraction.py             # Script for extracting tap windows from raw recordings
├── TAPER_FULL_DATASET/            # Empty placeholder for the externally hosted full dataset
├── your-processed-taps/           # Empty output folder for regenerated processed taps
│   ├── 1/
│   ├── 2/
│   ├── 3/
│   ├── 4/
│   ├── 5/
│   ├── 6/
│   ├── 7/
│   ├── 8/
│   ├── 9/
│   ├── 10/
│   └── 11/
└── README.md                      # Repository documentation
```

The folders `your-processed-taps/1` to `your-processed-taps/11` represent anonymised participants. They are intentionally empty in the GitHub repository and are used as output locations if users regenerate processed tap files from the raw dataset.
---

## Dataset structure expected by the code

After downloading the full dataset, the `TAPER_FULL_DATASET/` folder should contain the released dataset files. The evaluation notebook expects the preprocessed dataset to be arranged as follows:

```text
TAPER_FULL_DATASET/
└── pre-processed/
    ├── rings/
    │   ├── rings_32khz.csv
    │   ├── rings_16.0khz.csv
    │   ├── rings_8.0khz.csv
    │   └── ...
    └── tape_thick/
        ├── tape_thick_32khz.csv
        ├── tape_thick_16.0khz.csv
        ├── tape_thick_8.0khz.csv
        └── ...
```

Each preprocessed CSV file contains one tap per row:

```text
label, x1, x2, x3, ..., xn
```

where:

- `label` is the surface class label.
- `x1 ... xn` are the accelerometer magnitude samples for the extracted tap window.
- Labels are integer encoded as:
  - `0`: soft
  - `1`: medium
  - `2`: hard
  - `3`: harder

The released preprocessed data is the version used for the model training and evaluation reported in the associated submission. The raw data is included as an additional resource for transparency, inspection, and future work.

---

## Installation

Create a Python environment and install the required packages.

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

pip install numpy pandas matplotlib scikit-learn tensorflow keras jupyter
```

For raw tap extraction, also install:

```bash
pip install ximu3csv
```

Depending on your system and TensorFlow version, GPU support may require additional CUDA/cuDNN setup. CPU execution is sufficient for running the scripts, but the deep learning evaluations may take longer.

---

## Running the model evaluation notebook

Open the notebook:

```bash
jupyter notebook evaluation_notebook.ipynb
```

or:

```bash
jupyter lab evaluation_notebook.ipynb
```

The notebook evaluates representative time-series classification models across multiple sampling rates, including:

- Multi-Layer Perceptron (MLP)
- CNN-BiLSTM
- Fully Convolutional Network (FCN)
- ResNet
- Support Vector Machine (SVM)
- Random Forest
- ROCKET

By default, the notebook loads either the `rings` or `tape_thick` mounting configuration using the `mounting_tech` variable:

```python
mounting_tech = "rings"      # or "tape_thick"
```

The helper function:

```python
data, data_X, data_y = get_data(mounting_tech="rings", sampling_rate=32)
```

loads a selected sampling-rate version of the dataset and returns:

- `data`: the full CSV array.
- `data_X`: tap time-series features.
- `data_y`: class labels.

The evaluation loops test multiple downsampled sampling rates using:

```python
for i in [1, 2, 4, 8, 16, 32, 64, 128, 256, 320, 640, 1280, 2560, 3200]:
    sampling_rate = 32 / i
```

This corresponds to sampling rates from `32 kHz` down to `0.01 kHz`.

---

## Running tap extraction from raw data

The `taps_extraction.py` script is provided for users who want to regenerate tap windows from the raw IMU and tap-pad recordings.

Before running the script, ensure that the externally hosted full dataset has been downloaded and placed inside `TAPER_FULL_DATASET/`.

The raw dataset should follow this structure:

```text
TAPER_FULL_DATASET/
└── raw/
    ├── 1/
    │   ├── rings/
    │   │   ├── soft/
    │   │   ├── medium/
    │   │   ├── hard/
    │   │   └── harder/
    │   └── tape_thick/
    │       ├── soft/
    │       ├── medium/
    │       ├── hard/
    │       └── harder/
    ├── 2/
    ├── 3/
    └── ...
```

The repository also includes the following output structure:

```text
your-processed-taps/
├── 1/
├── 2/
├── 3/
├── 4/
├── 5/
├── 6/
├── 7/
├── 8/
├── 9/
├── 10/
└── 11/
```

These folders are empty by default. When the extraction script is run, generated CSV files are written into the corresponding anonymised participant folder.

To run the extraction script:

```bash
python taps_extraction.py
```

The script:

1. Reads XIMU3 raw sensor files.
2. Selects the IMU accelerometer channel.
3. Reads tap-pad force signals.
4. Detects tap events using a force threshold.
5. Extracts accelerometer magnitude windows around each tap event.
6. Downsamples the extracted tap windows across multiple sampling rates.
7. Saves labelled tap sequences as CSV files.

The current script writes outputs to:

```text
your-processed-taps/{participant}/{mounting_tech}_data_{sampling_rate}khz.csv
```

Example outputs include:

```text
your-processed-taps/1/rings_data_32.0khz.csv
your-processed-taps/1/rings_data_16.0khz.csv
your-processed-taps/1/rings_data_8.0khz.csv
```

---

## Important configuration options in `taps_extraction.py`

The extraction script includes several variables that can be edited depending on the dataset subset being processed:

```python
taps_no_each = 72
mounting_tech = "rings"  # or "tape_thick"
surfaces = ["soft", "medium", "hard", "harder"]
```

The main loop processes anonymised participants and downsampling factors:

```python
for person in range(1, 12):
    participant = person

    for factor in [1, 2, 4, 8, 16, 32, 64, 128, 256, 320, 640, 1280, 2560, 3200]:
        downsample_factor = factor
```

The downsampling factor maps to the effective sampling rate as:

```python
sampling_rate_khz = 32 / downsample_factor
```

For example:

| Downsample factor | Sampling rate |
|---:|---:|
| 1 | 32 kHz |
| 2 | 16 kHz |
| 4 | 8 kHz |
| 32 | 1 kHz |
| 320 | 0.1 kHz |
| 3200 | 0.01 kHz |

The extraction script currently processes one mounting technique at a time. To process both mounting configurations, run the script once with:

```python
mounting_tech = "rings"
```

and once with:

```python
mounting_tech = "tape_thick"
```

---

## Notes on raw vs preprocessed data

The preprocessed CSV files are the primary dataset used in the associated model evaluation experiments. These files are intended to make evaluation straightforward and reproducible without requiring users to rerun the raw extraction pipeline.

The raw data is included separately to support transparency, inspection, and future work. Users can modify the extraction window size, force threshold, downsampling strategy, or accelerometer representation if they want to explore alternative preprocessing choices.

The current extraction script uses accelerometer vector magnitude:

```python
magnitude = np.linalg.norm(accelerometer, axis=1)
```

The raw three-axis accelerometer data is also available in the raw recordings, so users can adapt the script to extract `x`, `y`, and `z` channels directly instead of magnitude-only sequences.

---

## Reproducibility notes

- The preprocessed CSV files in `TAPER_FULL_DATASET/pre-processed/` are the files used for the evaluation experiments in the associated submission.
- The raw data in `TAPER_FULL_DATASET/raw/` can be used to regenerate processed tap windows using `taps_extraction.py`.
- The `your-processed-taps/` folder is reserved for outputs created when regenerating tap files locally.
- The same integer class labels are used across mounting configurations and sampling rates.
- The same sampling-rate sweep is used across evaluated models.

For full reproducibility, ensure that the externally hosted dataset is downloaded and placed in the expected folder structure before running the notebook or extraction script.

---

## Known implementation notes

The code is provided as research code accompanying the dataset. Users may need to adjust paths and output directories depending on where the dataset is stored locally.

Before running the full notebook, check that all file names match the expected pattern:

```text
{mounting_tech}_{sampling_rate}khz.csv
```

Examples:

```text
rings_32khz.csv
tape_thick_32khz.csv
rings_16.0khz.csv
tape_thick_16.0khz.csv
```

Before running raw extraction, check that the participant folders exist under both:

```text
TAPER_FULL_DATASET/raw/
your-processed-taps/
```

If the output folders do not exist, create them manually before running the script.

---

## Citation

Citation details will be added after review.

<!-- For anonymous review, please cite this repository as:

```bibtex
@misc{taper_anonymous,
  title        = {TAPER: Tap-Interaction Evaluation Resource},
  author       = {Anonymous Author(s)},
  year         = {2026},
  note         = {Code and dataset repository submitted for anonymous review}
}
``` -->

---

## License

Add the repository license here.

Recommended options:

- Code: MIT License or Apache-2.0 License.
- Dataset: CC BY 4.0, if attribution is acceptable and consistent with the dataset release requirements.