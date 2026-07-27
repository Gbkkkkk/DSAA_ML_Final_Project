# Final Validation Report

Validation date: 2026-07-28

## Status

PASS. The final submission was rebuilt after the formal cross-validation,
grid-search, and strict expanding-history temporal repair, followed by a final
dataset-provenance and code-readability repair. Validation was repeated from a
freshly extracted ZIP. This report is intentionally kept outside
`zxszeto_group_IBM_AML.zip`.

## Modified Scope

- Reworked `project_zxszeto_group_IBM_AML.ipynb` into an executed research
  narrative with motivations, code explanations, observations, failure
  diagnosis, and implementation appendices.
- Added `src/aml_project_model_selection.py`.
- Documented and clarified `src/aml_project_pipeline.py` and
  `src/aml_project_enhancements.py`.
- Added formal CV/grid-search and strict temporal result CSVs and figures.
- Corrected the data instructions to distinguish the official Kaggle package
  from the transaction-only Hugging Face mirror.
- Made the loader accept the plain transaction CSV, its single-file ZIP, or a
  full Kaggle download ZIP without manual renaming.
- Simplified source-level function signatures and helper names so the submitted
  code is easier for course markers and group members to read.
- Updated and rebuilt the report and presentation PDFs.
- Updated `README.md` and rebuilt `zxszeto_group_IBM_AML.zip`.

## Dataset Provenance and Loader Repair

PASS. The official IBM AML Kaggle file list contains two files for each
size/illicitness variant: a transaction CSV such as `HI-Small_Trans.csv` and a
pattern-description file such as `HI-Small_Patterns.txt`. It does not provide a
separate account table. This project uses the transaction CSV for modeling;
the pattern text is optional documentation and is not parsed. Account-level
features are derived from the sender and receiver identifiers in `Account` and
`Account.1`.

The Hugging Face fallback contains only `HI-Small_Trans.csv.zip`, so the README
now labels it as a fallback rather than implying that it is identical to the
complete Kaggle package. `src/aml_project_pipeline.py` now discovers and streams
all of the following layouts:

- `data/HI-Small_Trans.csv`;
- `data/HI-Small_Trans.csv.zip`;
- a full Kaggle ZIP containing `HI-Small_Trans.csv` at any internal path.

The common streaming helper is reused by the enhancement script. A five-row
loader smoke test passed against the local transaction ZIP with all 11 expected
columns and the `Is Laundering` target.

## Formal Model Selection

PASS. Model selection uses three-fold `GridSearchCV`, average precision as the
refit metric, preprocessing inside each fold, out-of-fold threshold selection,
and one untouched 30 percent test split. Five model/feature configurations were
searched on the complete 165,177-row modeling sample.

- Logistic Regression base: CV PR-AUC 0.329 +/- 0.009; test PR-AUC 0.346.
- Decision Tree base: CV PR-AUC 0.468 +/- 0.011; test PR-AUC 0.477.
- Random Forest base: CV PR-AUC 0.553 +/- 0.011; test PR-AUC 0.553.
- HGB base: CV PR-AUC 0.561 +/- 0.009; test PR-AUC 0.556.
- HGB retrospective graph: CV PR-AUC 0.721 +/- 0.005; test PR-AUC
  0.723 and F1 0.633.

The candidate-level search table, selected parameters, fold dispersion,
out-of-fold threshold, and held-out metrics are saved in
`results/grid_search_candidates.csv` and
`results/cross_validation_summary.csv`.

## Temporal Audit and Repair

PASS. The original negative time audit is retained as a failure mode. A frozen
training-period account snapshot scored PR-AUC 0.459 versus 0.683 for base
features. This exposed a representation mismatch because later activity could
not update the account state.

The repair sorts all modeling rows chronologically and records every account,
pair, reverse-pair, recency, rolling-activity, counterparty, and amount-history
feature before the current transaction updates state. Base and strict-history
HGB models each receive an independent 18-candidate search on the chronological
validation period; the final test period is evaluated once.

- Chronological split: 60/20/20, with 99,106 train, 33,035 validation, and
  33,036 test rows.
- Tuned base: test PR-AUC 0.713 and F1 0.662.
- Tuned strict expanding history: test PR-AUC 0.829 and F1 0.723.
- Paired PR-AUC gain: +0.116.
- Paired 500-resample bootstrap 95% interval: [+0.100, +0.130], entirely above
  zero.

This validates feature availability on the sampled chronological stream. It is
not presented as a natural-prior or deployment-ready estimate.

## Notebook

PASS. The notebook was executed from the final worktree and again on 2026-07-28
from the freshly extracted ZIP using the project Python environment.

- Total cells: 76
- Markdown cells: 50
- Code cells: 26
- Code cells with execution counts: 26
- Code cells with preserved outputs: 24
- Error outputs: 0
- `RUN_FULL_PIPELINE = False` for fast reviewer execution: confirmed
- Full regeneration path invokes all three source scripts: confirmed
- Extracted-root assertion and relative project paths: confirmed

The notebook now covers preprocessing, EDA, PCA/t-SNE, KMeans and DBSCAN,
multiple clustering metrics, supervised baselines, train/test/full confusion
matrices, ROC/PR analysis, mutual-information selection, graph ablation,
natural-prior stress, formal CV/grid search, the frozen-snapshot failure,
strict historical repair, and bootstrap uncertainty. Markdown before and after
experiment cells states the question, explains the code, interprets the result,
and limits the claim. A final code appendix displays every line of all three
source modules so the submitted notebook contains the complete implementation.

The new model-selection and temporal experiments were actually run on all
165,177 rows in the established modeling sample. The earlier multi-million-row
core pipeline was not repeated in this repair; its saved full-data and
natural-prior artifacts remain unchanged.

## Report

PASS. The report uses the provided `neurips_2025.sty` in final mode and was
compiled twice in both the final worktree and the freshly extracted ZIP.

- PDF pages: 12 total.
- Main report: pages 1-10.
- References begin: page 11.
- Credit and GenAI disclosure: page 12.
- Course 5-10 page recommendation excluding references: satisfied.
- LaTeX errors, undefined controls/references, and overfull boxes: none.
- Preprint, anonymous, or under-review notice in the PDF: none.

All pages were rendered and inspected. The new CV figure, temporal ablation
figure, tables, references, and disclosure are readable with no clipping or
overlap.

## Presentation

PASS. The presentation remains 10 pages. Slides 6, 7, and 10 were updated in
the existing visual template.

- Slide 6 preserves the failure-driven development path and adds the temporal
  repair.
- Slide 7 reports the five formal CV comparisons and fold variability.
- Slide 10 contrasts frozen-snapshot failure with strict-history recovery and
  reports the paired interval.
- All ten PDF pages were rendered and visually inspected with no clipping,
  overlap, or missing page numbers.

## Clustering and Natural-prior Wording

PASS. DBSCAN maximum risk 0.848 is consistently identified as the label `-1`
noise/outlier group in the modeling account sample, not accuracy, purity, or
deployment precision. Low NMI and near-zero ARI remain visible.

PASS. The natural-prior stress test remains separate from temporal validation.
Its 0.1034 percent positive rate, 99,113 false positives, and 665.6 alerts per
10,000 are retained as workload evidence, not a prospective performance claim.

## Contribution and GenAI Checks

PASS. No percentage contribution is assigned to any member. The report and
README state that Minghua XIONG handled data processing and Bokang GAO and Zi
Xuan SZETO handled the remaining project components.

PASS. The disclosure now explicitly includes implementation assistance for the
model-selection and temporal-history experiments, along with planning,
debugging, visualization/report structuring, and language polishing. The stated
estimate remains 25 percent and the team-verification statement is retained.

## Clean Environment

Installation from `requirements_zxszeto_group_IBM_AML.txt` previously completed
successfully in the clean Python 3.13 validation environment.

PASS. On 2026-07-28 all three source modules compiled and imported after fresh
extraction to `tmp/fresh_zip_validation_20260728_final`. Jupyter execution from
that extracted root completed all 26 code cells successfully. Windows printed
benign ZMQ event-loop warnings; no notebook error resulted.

## ZIP

The final ZIP was generated with an explicit allowlist and POSIX archive paths.

- Entries: 103.
- CRC test with `ZipFile.testzip()`: PASS.
- Entries containing Windows backslashes: 0.
- Extra top-level folder: none.
- Fresh extraction: PASS.
- Notebook quick execution after extraction: PASS.
- Three-module import after extraction: PASS.
- Report PDF is byte-identical to the previously compiled and visually
  validated final report: PASS.

The ZIP excludes the raw Kaggle archive, Parquet feature caches, validation
reports, LaTeX intermediates, temporary renders, `__pycache__`, bytecode,
checkpoints, environments, old PDFs, and nested ZIPs.

## SHA-256 Identity

PASS. Each standalone submission file is byte-identical to its ZIP entry.

- Report PDF: `49f1ecb373b5829edf6e18921adc4628c8668df272b8e320245bd26715aee902`
- Report TEX: `2efa51c2b71f66d6720c0c0a41442ac63a6e9a03c23e796bdb294e7812014c71`
- Presentation PDF: `8516150c0653e4126c60d1b97eff8eb3bf456568f2b254450a67fbe51d7fff95`
- Notebook: `c85c23f96ba97ff0ca65d1d6c3dbd14368214cce1f3036bbdcff5869ac3bc658`
- Requirements: `fd3a81a79984cd59e0db9276c44a7c0e3b989c8b1b42475ee301861729280b1d`
- README: `dc5f9b90e2133fd7d37cfe75f345f405a14b2b6782e71187785d931b2780243f`
- Submission ZIP: `6fcef402b879ae5f19f590fd6198830eac1ec90c7bcb3299c2e4f9da87218516`

## Residual Notes

No submission blocker remains. The final code repair changes input discovery,
documentation, and readability; it does not alter saved experimental results.
No experimental result was fabricated or silently removed.
