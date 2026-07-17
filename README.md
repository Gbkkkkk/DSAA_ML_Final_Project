# DSAA2011 Final Project: IBM AML Transaction Detection

This repository contains a reproducible machine learning project for detecting money laundering in the IBM synthetic AML transaction dataset.

## Main Artifacts

- `project_groupID_IBM_AML.ipynb`: course notebook with the full reproducible pipeline and result discussion.
- `reports/report_groupID_IBM_AML.pdf`: project report draft.
- `slides/presentation_groupID_IBM_AML.pdf`: 10-minute presentation slide deck.
- `requirements_groupID_IBM_AML.txt`: Python package versions.
- `src/aml_project_pipeline.py`: end-to-end experiment pipeline.
- `figures/`: generated visualizations.
- `results/`: generated metrics and intermediate result tables.
- `groupID_IBM_AML_submission.zip`: packaged submission bundle.

## Reproduce Results

Install dependencies:

```bash
pip install -r requirements_groupID_IBM_AML.txt
```

Run the full pipeline:

```bash
python src/aml_project_pipeline.py
```

The pipeline automatically downloads `HI-Small_Trans.csv.zip` if it is not present in `data/`.

## Main Result

The best model is `HistGradientBoosting + Graph Features`:

- F1: 0.642
- ROC-AUC: 0.976
- PR-AUC: 0.723

The graph-feature ablation shows that adding graph-aware account behavior features improves the HistGradientBoosting model from PR-AUC 0.556 to 0.723.

## Notes

Before final submission, replace `groupID` in filenames with the real course group ID and adapt the credit section to the actual team contributions.
