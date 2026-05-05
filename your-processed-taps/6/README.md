# Processed taps output folder

This folder is intentionally left empty in the GitHub repository.

It is used as an output directory when regenerating processed tap CSV files from the raw TAPER dataset using `taps_extraction.py`.

Each numbered folder represents one anonymised participant. For example:

```text
your-processed-taps/
├── 1/
├── 2/
├── 3/
├── ...
└── 11/
```

When the extraction script is run, processed CSV files for this participant are written into the corresponding numbered folder.

Example output filenames may look like:

```text
rings_data_32.0khz.csv
rings_data_16.0khz.csv
rings_data_8.0khz.csv
...
rings_data_0.01khz.csv
```

Each CSV row represents one extracted tap. The first column is the surface label, and the remaining columns contain the tap time-series values.

Surface labels:

```text
0 = soft
1 = medium
2 = hard
3 = harder
```

The folders are included so that users can run the raw-data extraction code without needing to manually recreate the output structure.