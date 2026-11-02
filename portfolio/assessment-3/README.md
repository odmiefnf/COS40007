# Portfolio Assessment 3 - Energy Efficiency (reference implementation)

Nguyen Van Tuan Kiet | 104772099 | COS40007 | Prepared 2 November 2026

This package follows option 3d and continues the replacement Energy Efficiency
Portfolio 2 dataset. It is a substantially AI-generated reference, not evidence of
independent student authorship. The brief explicitly prohibits whole-code GenAI
generation. Disclosure does not override that restriction. Student review,
independent work within permitted assistance, and the live studio demo remain the
student's responsibility. No claim of completed student verification is made.

## Contents

- One self-contained executed notebook for Parts A-C, D and references.
- One PDF report with brief Part A, detailed Part B, comparisons, reflection and
  the mandatory four-part GenAI disclosure.
- `models/`: Random Forest and sklearn MLP pipelines (`.pkl`), three Keras ANN
  variants (`.h5`, `.keras`), preprocessing and training histories (`.pkl`).
- `data/`: unchanged Energy CSV and inherited Portfolio 2 split membership.
- `output/`: metrics, predictions, learning histories, figures and independent QA.
- `tools/`: notebook builder/executor, model inference, verifier and PDF renderer.
- `DEMO_GUIDE_VI.md`: preparation for the in-person studio demonstration.
- `GENAI_EVIDENCE.md`: actual assistance and exact user prompts.

## Reproduce

Use Python 3.11. Install `requirements.txt` in a separate environment, then open the
notebook from this directory and Restart Kernel / Run All. Keras uses its PyTorch
CPU backend; TensorFlow is not required. The notebook is self-contained and does
not download data. For the build process used here:

```text
python -m pip install -r requirements.txt
python tools/build_notebook.py
python tools/execute_notebook.py
python tools/verify_analysis.py
python tools/render_report.py
```

The builder recreates the notebook and clears stored outputs, so it is unnecessary
when reviewing the delivered executed notebook. The executor uses the active Python
interpreter and a local Jupyter kernel configuration. Exact execution versions are
recorded in `output/analysis_manifest.json` and `requirements-lock.txt`.

Example inference using the original X1-X8 schema:

```text
python tools/predict.py --input data/energy_data.csv --output predictions.csv
```

Only load trusted pickle files. Version compatibility matters when reloading them.

## Fixed task and validation

Source: Tsanas, A. and Xifara, A. (2012), Energy Efficiency, UCI Machine Learning
Repository, https://doi.org/10.24432/C51307 , CC BY 4.0. There are 768 simulated
building configurations. Eight X variables predict the relative Lower/Middle/Higher
heating-load bands defined in Portfolio 2. Y1 and Y2 never enter model predictors.
The unchanged source CSV SHA256 is
`483904d33ab6db8bb0d1791d81410c18567c6b967035325a844de8ca97046ddc`.

Preserve the PA2 537/231 development/test membership. Reserve 108 development rows
for validation, leaving 429 fitting rows. Encode two nominal variables and standardise
six numerical variables. Fit preprocessing only within fitting partitions, including
each CV fold. ANN variants use the same input preparation and rows.

Traditional models use five-fold CV. Three ANN variants compare batch sizes 16/32/64
with fixed 64-32-16 architecture, dropout 0.2, at most 50 epochs, early stopping and
learning-rate reduction. ANN selection uses validation loss. Test results are reported
only after choices are fixed in this run. The PA2 test rows were already examined;
they are a reused comparison set, not an untouched new external test. Three complete
geometry holdout folds assess a different generalisation setting with fixed models.

## GitHub and Canvas status

Existing repository: https://github.com/odmiefnf/COS40007 .

Assessment 3 is prepared locally in `portfolio3_publication/portfolio/assessment-3`
on branch `assessment-3-reference`. Its new history is independent of the existing
main branch. Commits are made as real milestones complete, explicitly attributed to
OpenAI Codex. No backdating or student-authorship claim is made. No remote push or
Canvas submission has been performed. The report does not present a future subfolder
URL as an already-published artifact.

Canvas displayed Due 1 November at 23:59 and Available until 5 November at 23:59
when accessed on 2 November 2026; interpret displayed times in the account timezone.
The May date printed inside the option PDF is stale relative to the live assignment.
The class identifier COS40007 is used consistently, matching the prior package and
the user's acceptance of that temporary naming choice.
