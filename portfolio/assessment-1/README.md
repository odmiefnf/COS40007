# Portfolio Assessment 1

Student: Nguyen Van Tuan Kiet — 104772099

Class: COS40007

Repository: https://github.com/odmiefnf/COS40007

Dataset: [Concrete Compressive Strength Data Set on Kaggle](https://www.kaggle.com/datasets/elikplim/concrete-compressive-strength-data-set).

Original attribution: I-Cheng Yeh, [UCI Concrete Compressive Strength dataset](https://doi.org/10.24432/C5PK67).
Kaggle lists its mirror licence as unknown; UCI identifies the original dataset as CC BY 4.0.

## Files

- `Portfolio_Assessment_1_COS40007_NguyenVanTuanKiet.ipynb`: executable analysis with saved outputs.
- `Portfolio_Assessment_1_COS40007_NguyenVanTuanKiet.pdf`: concise report following the supplied example.
- `Portfolio_Assessment_1_COS40007_NguyenVanTuanKiet.docx`: editable version of the report.
- `data/concrete_data.csv`: unchanged Kaggle source CSV.
- `requirements.txt`: recorded analysis package versions.
- `GENAI_EVIDENCE.md`: assistance and verification record.

## Run

Use Python 3.12 or compatible. From this folder:

```bash
python -m pip install -r requirements.txt jupyterlab
python -m jupyter lab
```

Open the notebook and run cells in order. It contains a copy of the source CSV in its metadata,
so no network download is required to execute it. Outputs are written to `output/`.

The source CSV SHA256 is
`1903285682ab24ebc5c1d4f79f8b9a389d01c23d528d3b64b52174847315c945`.

The analysis uses 1,030 raw observations and 1,005 observations after removing 25 exact full-row
duplicates. Numerical outlier flags are retained. No predictive model is trained.

The code was previously rerun in a fresh process. The PDF layout was inspected; native Word
pagination could not be rendered because LibreOffice was unavailable. Review the Word document
before editing or exporting. The report is an AI-assisted draft requiring student verification
and confirmation that the actual AI use is accepted by the tutor. Canvas upload remains separate.
