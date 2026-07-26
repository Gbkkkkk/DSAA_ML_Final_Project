# Final Validation Report

Validation date: 2026-07-24

## Status

PASS. The final submission files were repaired, rebuilt, and independently
validated from a freshly extracted ZIP.

This report is intentionally outside `zxszeto_group_IBM_AML.zip`.

## Modified Files

- `report_zxszeto_group_IBM_AML.tex`
- `report_zxszeto_group_IBM_AML.pdf`
- `presentation_zxszeto_group_IBM_AML.pdf`
- `project_zxszeto_group_IBM_AML.ipynb`
- `requirements_zxszeto_group_IBM_AML.txt`
- `README.md`
- `zxszeto_group_IBM_AML.zip`
- `src/create_latex_report_source.py` in the working project, for consistency
  with the final report source

## Contribution and GenAI Checks

PASS. The contribution statement is consistent in the report and README:

> Team contributions: Minghua XIONG was responsible for data processing.
> Bokang GAO and Zi Xuan SZETO were responsible for the remaining project
> components, including exploratory analysis, feature engineering, clustering,
> supervised modeling, model and feature ablation, natural class-prior stress
> testing, leakage auditing, report preparation, presentation design, and
> notebook integration.

No percentage contribution was assigned to any member.

PASS. The report and README retain the required GenAI disclosure and the
estimated GenAI-assisted content remains 25 percent.

## DBSCAN Wording

PASS. The value 0.848 is unchanged and was traced to
`account_cluster_profiles.csv`, where the maximum laundering-account rate
corresponds to `dbscan_cluster = -1`.

- Report: identifies it as the small high-risk DBSCAN noise/outlier group,
  label -1, in the modeling account sample.
- Presentation slide 5: uses `DBSCAN noise/outlier-group maximum risk`.
- Notebook: explains that it is not accuracy, purity, or natural-prior
  deployment precision.
- README: gives the same bounded interpretation.

The low NMI and near-zero ARI remain visible. The clustering result is presented
as an investigation signal, not as a classifier or deployment estimate.

## Requirements

Versions were queried from the actual Python 3.13 project environment. The
final requirements file contains:

```text
ipykernel==7.3.0
ipython==9.15.0
jupyter-client==8.9.1
jupyter-core==5.9.1
matplotlib==3.10.8
nbclient==0.11.0
nbformat==5.10.4
networkx==3.6.1
numpy==2.4.4
pandas==3.0.2
pyarrow==23.0.1
reportlab==4.4.10
scikit-learn==1.8.0
seaborn==0.13.2
```

PASS. A new virtual environment was created at
`tmp/clean_env_20260724_2248`. Installation with
`python -m pip install -r requirements_zxszeto_group_IBM_AML.txt` completed
successfully.

PASS. All notebook imports and both core scripts,
`src/aml_project_pipeline.py` and `src/aml_project_enhancements.py`, imported
successfully in that clean environment.

## Notebook

PASS. The notebook was executed in quick mode with the clean environment, first
from a clean staging root and then from the freshly extracted final ZIP root.

- Total cells: 31
- Code cells: 20
- Code cells with execution counts: 20
- Code cells with preserved outputs: 18
- `RUN_FULL_PIPELINE = False`: confirmed
- Required extracted-root assertion: confirmed
- Relative project paths: confirmed
- Quick/full mode and artifact-generation explanations: confirmed
- t-SNE, clustering, supervised comparisons, split evaluation, ROC/PR,
  confusion matrices, graph ablation, top-k MI, natural-prior stress, and
  time-based leakage audit: preserved

The full pipeline was not rerun during this repair because that would repeat
the multi-million-row experiment and was not needed to validate the requested
submission fixes. The saved experiment artifacts and executed quick-mode
notebook were retained.

## Report

PASS. The report uses the course-provided `neurips_2025.sty` in final,
non-preprint mode:

```tex
\usepackage[final,main,nonatbib]{neurips_2025}
```

- Compiled twice from the report workspace: PASS
- Compiled twice from the clean staging root: PASS
- Compiled twice from the freshly extracted ZIP root: PASS
- Pages: 6
- LaTeX errors: none
- Undefined controls: none
- Undefined citations/references: none
- Overfull boxes: none
- Preprint/anonymous/under-review PDF text: none

All six final pages were rendered and inspected. No clipping, overlap, broken
figures, or unreadable contribution text was found.

## Presentation

PASS. The presentation remains 10 pages and retains the requested narrative:
rare-event problem, accuracy failure, transaction/account features,
t-SNE/DBSCAN exploration, clustering interpretation, failure-driven path,
PR-AUC comparison, top-k MI trade-off, natural-prior alert pressure, and the
random-versus-time leakage audit.

- Slide 5 DBSCAN wording repaired: PASS
- Slide 10 random graph gain +0.166 PR-AUC retained: PASS
- Slide 10 time-based graph change -0.224 PR-AUC retained: PASS
- Bounded deployment conclusion retained: PASS
- All ten PDF pages rendered and inspected: PASS
- Clipping, overlap, stretched images, or missing page numbers: none found

## ZIP

The final ZIP was created with Python `zipfile` and
`path.relative_to(root).as_posix()`.

- Entries: 93
- CRC test with `ZipFile.testzip()`: PASS
- Entries containing Windows backslashes: 0
- Required forward-slash examples: confirmed
- Extra top-level project folder: none
- Fresh extraction to a new temporary directory: PASS
- Notebook quick execution after extraction: PASS
- Report compilation after extraction: PASS

The local shell did not provide `unzip` or `zipinfo`; the equivalent standard
library checks used `ZipFile.testzip()` and `ZipFile.namelist()`. The complete
entry list was inspected programmatically.

The ZIP excludes validation reports, audit files, old PDFs, nested ZIPs,
temporary renders, `__pycache__`, bytecode, checkpoints, the clean virtual
environment, cached files, and filenames containing `(1)` or `(2)`.

## Numeric Consistency

PASS. The report and presentation values were checked against
`results/*.csv` and `results/*.json`.

- HGB plus graph: F1 0.628 and PR-AUC 0.721
- HGB graph PR-AUC gain: +0.166
- Random Forest graph PR-AUC gain: +0.131
- Natural-prior stress: 0.1034 percent positive rate, 99,113 false positives,
  and 665.6 alerts per 10,000
- Time audit: base PR-AUC 0.683 versus graph PR-AUC 0.459
- DBSCAN maximum risk: 0.848 at label -1
- DBSCAN ARI: -0.008

The random/transductive graph gain, negative time-based graph result, and the
distinction among the three evaluation protocols are all preserved.

## SHA-256 Identity

PASS. The following standalone files are byte-identical to the same-named ZIP
entries:

- Report PDF: `b7b131deec8cde50e014db7c9694150194280666f4e3c7e62945101b672a8d76`
- Report TEX: `e2889c1597faef99f0716846bfcc7df5dd9bf01606a3b900da0d370ff070275b`
- Presentation PDF: `93e7d5f55b72dc108b913f61feff037aa0ce3c080de01cc203f6979e90dde177`
- Notebook: `c5a81b3b5c947b0c41d01117a1dede69107a628af4ec2928cf3d909816a529ae`
- Requirements: `fd3a81a79984cd59e0db9276c44a7c0e3b989c8b1b42475ee301861729280b1d`
- README: `7f4dfffa5f363373247d8175abe49cd3ecb5006d68e602b473b3a981e48a8971`

## Recursive Cleanup

PASS outside the course style file. No obsolete placeholder, preprint option,
group ID, student placeholder, replacement marker, TODO, FIXME, blocked-file
marker, or `data cleaning` contribution wording remains in submission-facing
text or source files.

The unmodified course style contains nine internal occurrences of `preprint`
as part of its option implementation. The report does not select that option,
and the compiled PDF contains no preprint notice.

## Residual Notes

No submission blocker remains.

- MiKTeX prints its local update reminder after successful compilation.
- Jupyter on Windows prints benign ZMQ transport/event-loop warnings while the
  notebook still exits successfully.
- The artifact-tool process returns a nonzero Windows status after writing the
  PPTX and all preview files. The PowerPoint-exported final PDF was therefore
  used for full ten-page visual validation.

No experimental result was fabricated, removed, or silently altered during
this repair.
