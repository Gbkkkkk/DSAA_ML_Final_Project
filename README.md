# DSAA2011 IBM AML Project

Group: zxszeto's group

Members: Bokang GAO, Zi Xuan SZETO, Minghua XIONG

## Project Overview

This project studies anti-money-laundering detection on the IBM synthetic transaction dataset. The central question is whether transaction-level features plus account graph behavior features improve laundering detection after considering class imbalance, feature ablation, natural class-prior stress, and time-based leakage risk.

## Submission Files

- `report_zxszeto_group_IBM_AML.pdf`: final report compiled with the provided `neurips_2025.sty` template from `Styles.zip`.
- `report_zxszeto_group_IBM_AML.tex`: final LaTeX report source.
- `neurips_2025.sty`: course-provided LaTeX style required to compile the report.
- `presentation_zxszeto_group_IBM_AML.pdf`: 10-page presentation deck.
- `project_zxszeto_group_IBM_AML.ipynb`: executed technical notebook with outputs preserved.
- `requirements_zxszeto_group_IBM_AML.txt`: Python dependencies.
- `zxszeto_group_IBM_AML.zip`: packaged project archive.

## Directory Structure

- `src/`: reproducible core pipeline, enhancement analysis, and formal model-selection/time-audit experiments.
- `figures/`: EDA, t-SNE, clustering, ROC/PR, confusion matrix, and ablation visualizations.
- `results/`: model metrics, clustering metrics, feature selection results, natural-prior stress test, and time-based leakage audit outputs.
- `data/README.txt`: dataset download note.

## Dataset Source

The dataset is IBM Transactions for Anti Money Laundering, available from Kaggle:

https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml

The scripts can use a programmatic mirror only as a fallback. The raw Kaggle data file is not bundled in this submission archive to keep the package portable.

## Quick Notebook Execution

From the extracted project root:

```bash
pip install -r requirements_zxszeto_group_IBM_AML.txt
jupyter execute project_zxszeto_group_IBM_AML.ipynb
```

The notebook defaults to quick-review mode and reads the submitted `results/` and `figures/` artifacts. Its outputs are preserved. The path assertion at the top reports a clear error if it is launched outside the extracted project root.

## Requirements Installation

The requirements file records the versions used in the validated project
environment, including the Jupyter/IPython packages needed to execute the
notebook:

```bash
python -m venv clean_env
clean_env\Scripts\python -m pip install -r requirements_zxszeto_group_IBM_AML.txt
```

## Full Pipeline Execution

The raw dataset is required for full recomputation. Set `RUN_FULL_PIPELINE = True` in the notebook, or run:

```bash
python src/aml_project_pipeline.py
python src/aml_project_enhancements.py
python src/aml_project_model_selection.py
```

The core pipeline produces preprocessing, EDA, t-SNE, supervised metrics, ROC/PR curves, and train/test/full confusion matrices. The enhancement pipeline produces account clustering, graph-feature and mutual-information ablations, the natural class-prior stress test, and the time-based leakage audit.
The model-selection pipeline performs three-fold GridSearchCV with preprocessing
inside each fold, out-of-fold threshold selection, a held-out test evaluation,
and a separate chronological 60/20/20 experiment with strictly expanding
history features. Each historical feature is recorded before the current
transaction updates account state.

## Report Compilation

The report source and course style are included at the ZIP root. Compile twice:

```bash
pdflatex -interaction=nonstopmode -halt-on-error report_zxszeto_group_IBM_AML.tex
pdflatex -interaction=nonstopmode -halt-on-error report_zxszeto_group_IBM_AML.tex
```

## Presentation File

The 10-page presentation is `presentation_zxszeto_group_IBM_AML.pdf`.

## Main Findings

Accuracy is misleading for AML because the full-data laundering rate is about
0.1019%. Formal three-fold GridSearchCV gives the tuned retrospective HGB graph
model mean CV PR-AUC 0.721 +/- 0.005 and held-out PR-AUC 0.723, compared with
0.561 +/- 0.009 and 0.556 for tuned base HGB. A deliberately frozen
training-period graph snapshot fails in the chronological audit (PR-AUC 0.459
versus 0.683), exposing a representation mismatch. Recomputing each graph
feature from prior transactions only repairs that failure: the independently
tuned strict-history model reaches chronological test PR-AUC 0.829 versus
0.713 for base features, a paired gain of +0.116 with a 95% bootstrap interval
of [+0.100, +0.130]. This is evidence for online historical feature value on
the modeling sample, not a natural-prior deployment estimate.

Under the natural class-prior stress test, precision drops sharply, showing
that alert-budget thresholding matters more than accuracy. DBSCAN's maximum
risk of 0.848 corresponds to label `-1`, the noise/outlier group in the
modeling account sample; it is an investigation signal, not accuracy, purity,
or natural-prior deployment precision.

## Limitations

The dataset is synthetic. Retrospective random-split graph features may contain
transductive information. The strict chronological experiment removes
future-edge leakage in the sampled stream, but its class prior is still about
5.32%, its accounts are not fully independent, and its history is limited by
sampling. The natural-prior stress test isolates base-rate effects but is not
a fully independent temporal deployment test. Production validation still
requires the unsampled stream, out-of-time and account-disjoint testing,
calibration, and thresholds tied to investigation capacity.

## Contributions and GenAI Disclosure

Team contributions: Minghua XIONG was responsible for data processing. Bokang
GAO and Zi Xuan SZETO were responsible for the remaining project components,
including exploratory analysis, feature engineering, clustering, supervised
modeling, model and feature ablation, natural class-prior stress testing,
leakage auditing, report preparation, presentation design, and notebook
integration.

Codex/ChatGPT was used for planning, debugging, implementation assistance for
model-selection and temporal-history experiments, visualization/report
structuring, and language polishing. The team verified and reran the code,
checked the numerical outputs, and reviewed the final interpretation. Estimated
GenAI-assisted content: 25%.
