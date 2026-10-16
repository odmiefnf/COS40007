# Portfolio Assessment 2 - Energy Efficiency

Nguyen Van Tuan Kiet | 104772099 | COS40007 | 16 October 2026

This is the replacement version requested by the student after discovering that the
earlier concrete data exactly matched the Studio 2 sample excluded by Studio 3.
Use the Energy version for review. The earlier concrete analysis remains separately
in the original workspace `portfolio2/` as a reference and is not the selected submission version.

## Deliverables

- `Portfolio_Assessment_2_COS40007_NguyenVanTuanKiet.ipynb`: self-contained executed
  notebook covering Tasks 1-10, new EDA, all code/outputs, six DT reflections and four SVM reflections.
- `Portfolio_Assessment_2_COS40007_NguyenVanTuanKiet.pdf`: complete report.
- `data/ENB2012_data.xlsx`: unchanged source workbook.
- `data/energy_data.csv`: numerical CSV export used by the notebook.
- `output/`: figures, comparison tables, fold results, holdout predictions, source hashes,
  split manifest and independent verification.
- `DEMO_GUIDE_VI.md`: Vietnamese preparation notes for the live demo.
- `GENAI_EVIDENCE.md`: accurate record of substantial assistant involvement.

## Dataset and scope

Tsanas, A. and Xifara, A. (2012). Energy Efficiency [Dataset]. UCI Machine Learning
Repository. https://doi.org/10.24432/C51307 . CC BY 4.0.

768 simulated building configurations; eight inputs. Heating load Y1 is the target.
Cooling load Y2 is excluded from predictors. Orientation and glazing distribution are
nominal and one-hot encoded. The EDA for this replacement is included in the notebook;
it is not claimed to be the previously submitted concrete Portfolio 1.

Source workbook SHA256:
`0089fcffc1415e41e2ff63730cca5280efe54fc43d722dfde1b0aaa808e35dc4`

Analysis CSV SHA256:
`483904d33ab6db8bb0d1791d81410c18567c6b967035325a844de8ca97046ddc`

## Reproduce

Use Python 3.12 and install dependencies in your own environment:

```text
python -m pip install -r requirements.txt
python -m pip install jupyterlab
python -m jupyter lab
```

Open the notebook from this folder and select Restart Kernel and Run All. The data are
included; model execution needs no network. Seed 42 defines a 537/231 train/test split.
Training-only target terciles are 14.463333333333333 and 28.406666666666666 dataset units.
All 537 training observations divide into three classes of 179.

Five DT feature configurations use the same 5-fold splits. The four SVM procedures use
five outer folds and five inner folds for tuning. Grouped sensitivity analysis holds out
entire building geometries and repeats tuning within grouped training folds.

The complete notebook is the portable source of analysis. The local builder scripts
were used to prepare it; they are not needed for running the submission notebook.
To regenerate the PDF from saved outputs after installing requirements:

```text
python tools/render_report.py --output Portfolio_Assessment_2_COS40007_NguyenVanTuanKiet.pdf
```

Poppler is required only for the optional `--render` visual QA command.

## Main findings and limitations

CV selects M3 Engineered for DT and S3 K-Best for SVM. The observed holdout accuracies are
95.24% and 96.10%; S3 nested row-CV accuracy is 92.36%. However, holding out complete
building geometries gives mean accuracy about 65.28% for DT and 68.65% for the retuned
SVM procedure. All 12 geometries occur on both sides of the ordinary row split. Therefore,
the strong random-split result must not be presented as validated accuracy for unseen shapes.

## Student actions before submission

1. Read, understand and revise the analysis; student verification is still pending.
2. Keep the GenAI declaration accurate. Actual assistance includes design, implementation,
   interpretation and report drafting, beyond the brief's listed examples. Confirm its
   acceptability and take responsibility for the reviewed work; do not insert an untrue
   independent-authorship statement.
3. The new EDA is included because the dataset changed. If required, use it to discuss the
   change from Portfolio 1 with the tutor.
4. Practise the live studio demo (20 rubric points); see the Vietnamese guide.
5. Check the Canvas deadline and submit the PDF through Canvas. The notebook/data are published to the student-selected repository above. Canvas submission has not been performed.

Canvas displayed 18 October 2026 at 23:59 as the due date; availability ends 22 October.
The April date in the supplied brief is stale. Interpret displayed times in the Canvas
account's configured timezone.


## GitHub repository

https://github.com/odmiefnf/COS40007

PA2 files: https://github.com/odmiefnf/COS40007/tree/main/portfolio/assessment-2

The student selected the same repository as Portfolio 1. Code, source data, reports and reproducibility files are placed under `portfolio/assessment-2`. Canvas submission remains separate.
