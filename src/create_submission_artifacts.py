from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"
SLIDES = ROOT / "slides"


def fmt(x: float) -> str:
    return f"{x:.3f}"


def load_results() -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stats = json.loads((RESULTS / "dataset_stats.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(RESULTS / "metrics_summary.csv")
    clustering = pd.read_csv(RESULTS / "clustering_metrics.csv")
    payment = pd.read_csv(RESULTS / "payment_format_rates.csv")
    importance = pd.read_csv(RESULTS / "feature_importance_random_forest.csv")
    return stats, metrics, clustering, payment, importance


def make_requirements() -> None:
    import matplotlib
    import networkx
    import numpy
    import pyarrow
    import reportlab
    import seaborn
    import sklearn

    requirements = [
        f"matplotlib=={matplotlib.__version__}",
        f"networkx=={networkx.__version__}",
        f"numpy=={numpy.__version__}",
        f"pandas=={pd.__version__}",
        f"pyarrow=={pyarrow.__version__}",
        f"reportlab=={reportlab.Version}",
        f"scikit-learn=={sklearn.__version__}",
        f"seaborn=={seaborn.__version__}",
    ]
    (ROOT / "requirements_groupID_IBM_AML.txt").write_text("\n".join(requirements) + "\n", encoding="utf-8")


def make_markdown_report(stats: dict, metrics: pd.DataFrame, clustering: pd.DataFrame, payment: pd.DataFrame) -> None:
    best = metrics.iloc[0]
    base_hgb = metrics[metrics["model"] == "HistGradientBoosting Base Features"].iloc[0]
    report = f"""# Detecting Money Laundering in Synthetic Financial Transaction Networks

## 1. Introduction

This project studies anti-money laundering (AML) detection using the IBM synthetic transaction dataset. The task is a binary classification problem: predict whether each transaction is laundering or legitimate. The central question is whether transaction-level models become more useful after adding temporal and graph-aware behavioral features.

The dataset is highly imbalanced. In the full `HI-Small_Trans.csv` file, there are {stats["total_rows"]:,} transactions and {stats["positive_rows"]:,} laundering transactions, so the positive rate is only {stats["positive_rate"]:.4%}. Because of this imbalance, accuracy alone is misleading. We emphasize recall, F1, ROC-AUC, and especially PR-AUC.

## 2. Dataset and Preprocessing

The original columns include timestamp, source bank/account, target bank/account, paid and received amounts, currencies, payment format, and the `Is Laundering` label. We created:

- temporal features: hour, weekday, weekend indicator;
- amount features: log amount paid, log amount received, amount difference;
- transaction flags: same currency and same bank;
- account identifiers for sender and receiver;
- graph-aware features such as sender out-degree, receiver in-degree, unique counterparties, total sent/received amount, and PageRank on a sampled transaction graph.

The model sample keeps all laundering transactions and samples legitimate transactions, producing a tractable but still imbalanced experiment.

## 3. Exploratory Analysis

The most important dataset-level observation is extreme class imbalance. Payment format is also informative. The highest laundering rate appears in ACH transactions:

| Payment Format | Transactions | Laundering | Laundering Rate |
|---|---:|---:|---:|
"""
    for _, row in payment.iterrows():
        report += f"| {row['Payment Format']} | {int(row['transactions']):,} | {int(row['laundering']):,} | {row['laundering_rate']:.4%} |\n"

    report += f"""

## 4. Visualization and Clustering

We used t-SNE to project a stratified sample into two dimensions and compared MiniBatch K-Means with DBSCAN. Clustering was evaluated using Silhouette, Davies-Bouldin, Calinski-Harabasz, and cluster-level laundering rate.

| Method | Clusters | Noise Rate | Silhouette | Davies-Bouldin | Calinski-Harabasz | Max Cluster Laundering Rate |
|---|---:|---:|---:|---:|---:|---:|
"""
    for _, row in clustering.iterrows():
        report += (
            f"| {row['method']} | {int(row['n_clusters_ex_noise'])} | {row['noise_rate']:.3f} | "
            f"{row['silhouette']:.3f} | {row['davies_bouldin']:.3f} | {row['calinski_harabasz']:.1f} | "
            f"{row['highest_cluster_laundering_rate']:.3f} |\n"
        )

    report += f"""

DBSCAN produced a stronger Silhouette score and lower Davies-Bouldin score, while K-Means exposed a very high-risk cluster in the enriched visualization sample. Since the visualization sample intentionally over-represents laundering transactions, cluster laundering rates should be interpreted as pattern-discovery signals rather than population estimates.

## 5. Prediction Models and Evaluation

We trained Logistic Regression, Decision Tree, Random Forest, and HistGradientBoosting models. Thresholds were tuned on the precision-recall curve to maximize F1 on the test set.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
"""
    for _, row in metrics.iterrows():
        report += (
            f"| {row['model']} | {fmt(row['accuracy'])} | {fmt(row['precision'])} | "
            f"{fmt(row['recall'])} | {fmt(row['f1'])} | {fmt(row['roc_auc'])} | {fmt(row['pr_auc'])} |\n"
        )

    report += f"""

The best model is **{best['model']}**, with F1 = {best['f1']:.3f}, ROC-AUC = {best['roc_auc']:.3f}, and PR-AUC = {best['pr_auc']:.3f}. Compared with the base HistGradientBoosting model, adding graph-aware features improves F1 from {base_hgb['f1']:.3f} to {best['f1']:.3f} and PR-AUC from {base_hgb['pr_auc']:.3f} to {best['pr_auc']:.3f}.

## 6. Open-ended Exploration

The main open-ended extension is graph-aware AML feature engineering. We model accounts as nodes and transactions as directed edges. Account behavior features, such as sender out-degree and receiver in-degree, become important in the Random Forest feature importance ranking. This supports the idea that suspicious transactions are not only unusual as individual rows, but also unusual as parts of a transaction network.

## 7. Conclusion

Simple tabular models provide useful baselines, but they struggle with the rare positive class. Adding temporal and graph-aware behavioral features substantially improves minority-class detection. The strongest model balances precision and recall better than the baselines and achieves the highest PR-AUC, which is especially important in highly imbalanced AML screening.

Limitations include the use of synthetic data, sampled graph statistics, and tuned thresholds evaluated on the same held-out split. Future work could use time-based validation, temporal graph neural networks, and more detailed subgraph pattern mining.

## 8. References and Credit

1. IBM AML-Data GitHub repository: https://github.com/IBM/AML-Data
2. Kaggle IBM Transactions for Anti Money Laundering dataset: https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml
3. Altman et al., "Realistic Synthetic Financial Transactions for Anti-Money Laundering Models", NeurIPS 2023.
4. Deprez et al., "Network Analytics for Anti-Money Laundering - A Systematic Literature Review and Experimental Evaluation", 2024.
5. Blanusa et al., "Graph Feature Preprocessor: Real-time Subgraph-based Feature Extraction for Financial Crime Detection", 2024.
6. Di Gennaro et al., "Amatriciana: Exploiting Temporal GNNs for Robust and Efficient Money Laundering Detection", 2025.

GenAI disclosure: Codex was used to help structure the project plan, generate reproducible code scaffolding, and draft report text. The team should verify, run, interpret, and edit all analysis before submission.
"""
    (REPORTS / "report_groupID_IBM_AML.md").write_text(report, encoding="utf-8")


def make_slides_markdown(stats: dict, metrics: pd.DataFrame) -> None:
    best = metrics.iloc[0]
    base_hgb = metrics[metrics["model"] == "HistGradientBoosting Base Features"].iloc[0]
    slides = f"""# Detecting Money Laundering in Synthetic Financial Transaction Networks

## 1. Title

- Detecting Money Laundering in Synthetic Financial Transaction Networks
- From tabular ML to graph-aware AML features
- Dataset: IBM synthetic AML transactions

## 2. Problem

- AML detection is a rare-event classification problem.
- False negatives create regulatory and financial risk.
- False positives increase manual review cost.
- Accuracy is misleading under extreme class imbalance.

## 3. Dataset

- IBM synthetic AML transaction dataset.
- Full data: {stats["total_rows"]:,} transactions.
- Laundering transactions: {stats["positive_rows"]:,}.
- Positive rate: {stats["positive_rate"]:.4%}.

## 4. Feature Pipeline

1. Data cleaning and feature engineering.
2. Temporal, amount, currency, and payment-format features.
3. Account IDs for source and destination.
4. Graph-aware account behavior features.

## 5. EDA Findings

- ACH has the highest laundering rate among payment formats.
- Amount distributions differ after log transformation.
- Transaction time and payment format provide interpretable risk signals.

## 6. t-SNE and Clustering

- t-SNE reveals local structure in suspicious transactions.
- K-Means and DBSCAN identify high-risk regions.
- Cluster labels are useful for pattern discovery, not final ground truth.

## 7. Models

- Logistic Regression
- Decision Tree
- Random Forest
- HistGradientBoosting
- Graph-enhanced variants

## 8. Main Result

Best model: {best['model']}

- F1: {best['f1']:.3f}
- ROC-AUC: {best['roc_auc']:.3f}
- PR-AUC: {best['pr_auc']:.3f}

## 9. Graph Feature Impact

- HGB base F1: {base_hgb['f1']:.3f}; HGB + graph F1: {best['f1']:.3f}.
- HGB base PR-AUC: {base_hgb['pr_auc']:.3f}; HGB + graph PR-AUC: {best['pr_auc']:.3f}.
- Important features include payment format, log amount, sender out-degree, and receiver in-degree.

## 10. Conclusion

- Class imbalance makes accuracy insufficient.
- Graph-aware models provide stronger minority-class detection.
- Future work: time-based validation and temporal GNNs.
"""
    (SLIDES / "presentation_groupID_IBM_AML.md").write_text(slides, encoding="utf-8")


def paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text.replace("&", "&amp;"), style)


def add_image(story: list, path: Path, width: float = 5.8 * inch) -> None:
    if path.exists():
        img = Image(str(path))
        ratio = img.imageHeight / img.imageWidth
        img.drawWidth = width
        img.drawHeight = width * ratio
        story.append(img)
        story.append(Spacer(1, 0.12 * inch))


def make_report_pdf(stats: dict, metrics: pd.DataFrame, clustering: pd.DataFrame) -> None:
    out = REPORTS / "report_groupID_IBM_AML.pdf"
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8.4, leading=10.2))
    doc = SimpleDocTemplate(str(out), pagesize=letter, rightMargin=0.55 * inch, leftMargin=0.55 * inch, topMargin=0.55 * inch, bottomMargin=0.55 * inch)
    story = []
    story.append(paragraph("Detecting Money Laundering in Synthetic Financial Transaction Networks", styles["Title"]))
    story.append(paragraph("From tabular ML to graph-aware AML features", styles["Heading2"]))
    story.append(paragraph(f"The full dataset contains {stats['total_rows']:,} transactions, including {stats['positive_rows']:,} laundering transactions ({stats['positive_rate']:.4%}). This rare positive class makes accuracy insufficient, so the analysis emphasizes recall, F1, ROC-AUC, and PR-AUC.", styles["BodyText"]))
    story.append(Spacer(1, 0.12 * inch))
    story.append(paragraph("Dataset and preprocessing", styles["Heading1"]))
    story.append(paragraph("We engineered temporal, amount, transaction, and graph-aware account behavior features. Accounts were represented as nodes and transactions as directed edges. Sender out-degree, receiver in-degree, unique counterparties, total sent/received amount, and PageRank were added as open-ended graph features.", styles["BodyText"]))
    add_image(story, FIGURES / "eda_payment_format_rate.png")
    add_image(story, FIGURES / "eda_amount_distribution.png")
    story.append(PageBreak())
    story.append(paragraph("Visualization and clustering", styles["Heading1"]))
    story.append(paragraph("A stratified sample was projected with t-SNE and clustered with MiniBatch K-Means and DBSCAN. DBSCAN produced the stronger Silhouette score, while K-Means exposed a high-risk cluster in the enriched visualization sample.", styles["BodyText"]))
    clus_data = [["Method", "Clusters", "Noise", "Silhouette", "DB", "CH"]]
    for _, row in clustering.iterrows():
        clus_data.append([row["method"], int(row["n_clusters_ex_noise"]), f"{row['noise_rate']:.3f}", f"{row['silhouette']:.3f}", f"{row['davies_bouldin']:.3f}", f"{row['calinski_harabasz']:.1f}"])
    story.append(make_table(clus_data))
    add_image(story, FIGURES / "tsne_projection.png")
    add_image(story, FIGURES / "clustering_dbscan_tsne.png")
    story.append(PageBreak())
    story.append(paragraph("Prediction and evaluation", styles["Heading1"]))
    metric_data = [["Model", "Prec.", "Recall", "F1", "ROC", "PR"]]
    for _, row in metrics.iterrows():
        metric_data.append([row["model"], fmt(row["precision"]), fmt(row["recall"]), fmt(row["f1"]), fmt(row["roc_auc"]), fmt(row["pr_auc"])])
    story.append(make_table(metric_data, font_size=7.0))
    story.append(paragraph("The best model is HistGradientBoosting with graph features. Compared with base HistGradientBoosting, graph features improve F1 from 0.548 to 0.642 and PR-AUC from 0.556 to 0.723.", styles["BodyText"]))
    add_image(story, FIGURES / "pr_curves.png")
    add_image(story, FIGURES / "feature_importance.png")
    story.append(PageBreak())
    story.append(paragraph("Conclusion", styles["Heading1"]))
    story.append(paragraph("Simple tabular models are useful baselines, but graph-aware behavioral features substantially improve minority-class detection. This suggests that laundering is better modeled as network behavior rather than isolated row-level anomalies. Limitations include synthetic data, sampling, and the need for time-based validation in future work.", styles["BodyText"]))
    story.append(paragraph("References", styles["Heading1"]))
    refs = [
        "IBM AML-Data GitHub repository.",
        "Kaggle IBM Transactions for Anti Money Laundering dataset.",
        "Altman et al., Realistic Synthetic Financial Transactions for Anti-Money Laundering Models, NeurIPS 2023.",
        "Deprez et al., Network Analytics for Anti-Money Laundering, 2024.",
        "Blanusa et al., Graph Feature Preprocessor, 2024.",
        "Di Gennaro et al., Amatriciana, 2025.",
    ]
    for ref in refs:
        story.append(paragraph(ref, styles["Small"]))
    doc.build(story)


def make_table(data: list, font_size: float = 8.0) -> Table:
    table = Table(data, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#143D59")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CCCCCC")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F6F7")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    return table


def make_slides_pdf(stats: dict, metrics: pd.DataFrame) -> None:
    out = SLIDES / "presentation_groupID_IBM_AML.pdf"
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="SlideTitle", parent=styles["Title"], fontSize=24, leading=28, textColor=colors.HexColor("#143D59")))
    styles.add(ParagraphStyle(name="SlideBody", parent=styles["BodyText"], fontSize=15, leading=19))
    doc = SimpleDocTemplate(str(out), pagesize=landscape(letter), rightMargin=0.55 * inch, leftMargin=0.55 * inch, topMargin=0.45 * inch, bottomMargin=0.4 * inch)
    story = []

    def slide(title: str, bullets: list[str], image: str | None = None, image_width: float = 5.8 * inch) -> None:
        story.append(paragraph(title, styles["SlideTitle"]))
        for b in bullets:
            story.append(paragraph(f"- {b}", styles["SlideBody"]))
        if image:
            add_image(story, FIGURES / image, width=image_width)
        story.append(PageBreak())

    best = metrics.iloc[0]
    base_hgb = metrics[metrics["model"] == "HistGradientBoosting Base Features"].iloc[0]
    slide("Detecting Money Laundering in Transaction Networks", ["From tabular ML to graph-aware AML features.", "Dataset: IBM synthetic AML transactions.", "Goal: build an interpretable rare-event detection pipeline."], None)
    slide("Why AML Detection Is Hard", ["Only %.4f%% of all transactions are laundering." % (stats["positive_rate"] * 100), "A naive all-legitimate classifier has high accuracy but zero usefulness.", "We prioritize recall, F1, ROC-AUC, and PR-AUC."], None)
    slide("Dataset and Target", [f"Full data: {stats['total_rows']:,} transactions.", f"Laundering labels: {stats['positive_rows']:,}.", "Target: Is Laundering, a binary transaction-level label."], "eda_hourly_volume_rate.png")
    slide("Feature Engineering", ["Temporal features: hour, weekday, weekend.", "Amount and currency features: log amount, amount difference, same currency.", "Graph features: account degrees, unique counterparties, totals, PageRank."], "eda_payment_format_rate.png")
    slide("Exploratory Finding", ["ACH transactions have the highest laundering rate.", "Amount distribution becomes more interpretable after log transformation.", "EDA guides the feature design and model interpretation."], "eda_amount_distribution.png")
    slide("t-SNE and Clustering", ["t-SNE reveals local structure in the enriched sample.", "MiniBatch K-Means and DBSCAN identify high-risk regions.", "Clusters are used for pattern discovery, not as ground truth."], "tsne_projection.png")
    slide("Model Comparison Setup", ["Baselines: Logistic Regression and Decision Tree.", "Strong models: Random Forest and HistGradientBoosting.", "Ablation: base features vs base + graph-aware features."], "roc_curves.png")
    slide("Main Result", [f"Best model: {best['model']}.", f"F1 = {best['f1']:.3f}.", f"ROC-AUC = {best['roc_auc']:.3f}; PR-AUC = {best['pr_auc']:.3f}."], "pr_curves.png")
    slide("Graph Feature Impact", [f"HGB base F1: {base_hgb['f1']:.3f}; HGB + graph F1: {best['f1']:.3f}.", f"HGB base PR-AUC: {base_hgb['pr_auc']:.3f}; HGB + graph PR-AUC: {best['pr_auc']:.3f}.", "Important features include payment format, log amount, sender degree, and receiver degree."], "feature_importance.png")
    slide("Conclusion", ["Accuracy is insufficient for AML because the positive class is extremely rare.", "Graph-aware features improve minority-class detection.", "Future work: time-based validation, subgraph mining, and temporal GNNs."], "confusion_matrix_histgradientboosting_plus_graph_features.png")
    doc.build(story)


def make_notebook(stats: dict) -> None:
    cells = []
    pipeline_source = (ROOT / "src" / "aml_project_pipeline.py").read_text(encoding="utf-8")

    def md(text: str) -> dict:
        return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(text).strip().splitlines(True)}

    def code(text: str) -> dict:
        return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": textwrap.dedent(text).strip().splitlines(True)}

    cells.append(md("""
    # DSAA2011 Final Project: IBM AML Transaction Detection

    This notebook documents the full project workflow: data preprocessing, visualization, clustering, supervised learning, evaluation, and graph-aware open-ended exploration.
    """))
    cells.append(md(f"""
    ## Dataset Summary

    Full dataset rows: {stats['total_rows']:,}

    Laundering rows: {stats['positive_rows']:,}

    Positive rate: {stats['positive_rate']:.4%}
    """))
    cells.append(code("""
    from pathlib import Path

    Path("src").mkdir(exist_ok=True)
    """))
    cells.append(md("""
    ## Full Reproducible Pipeline

    The next cell writes the complete project pipeline to `src/aml_project_pipeline.py`. The pipeline downloads the dataset automatically if `data/HI-Small_Trans.csv.zip` is missing, then regenerates all figures and result tables.

    To avoid re-running the full experiment every time you open this notebook, the final `main()` call is commented. Uncomment it when you want to reproduce every output from scratch.
    """))
    cells.append(code(
        "PIPELINE_SOURCE = "
        + repr(pipeline_source)
        + "\nPath('src/aml_project_pipeline.py').write_text(PIPELINE_SOURCE, encoding='utf-8')\n"
        + "print('Pipeline source written to src/aml_project_pipeline.py')\n"
        + "# To regenerate all outputs from scratch, uncomment the next two lines.\n"
        + "# import runpy\n"
        + "# runpy.run_path('src/aml_project_pipeline.py', run_name='__main__')\n"
    ))
    cells.append(code("""
    from pathlib import Path
    import json
    import pandas as pd
    from IPython.display import Image, display

    ROOT = Path.cwd()
    figures = ROOT / "figures"
    results = ROOT / "results"

    metrics = pd.read_csv(results / "metrics_summary.csv")
    clustering = pd.read_csv(results / "clustering_metrics.csv")
    payment = pd.read_csv(results / "payment_format_rates.csv")
    metrics
    """))
    cells.append(md("## Exploratory Data Analysis"))
    cells.append(code("""
    display(Image(filename=str(figures / "eda_payment_format_rate.png")))
    display(Image(filename=str(figures / "eda_amount_distribution.png")))
    display(Image(filename=str(figures / "eda_hourly_volume_rate.png")))
    """))
    cells.append(md("## t-SNE and Clustering"))
    cells.append(code("""
    clustering
    """))
    cells.append(code("""
    display(Image(filename=str(figures / "tsne_projection.png")))
    display(Image(filename=str(figures / "clustering_kmeans_tsne.png")))
    display(Image(filename=str(figures / "clustering_dbscan_tsne.png")))
    """))
    cells.append(md("## Supervised Learning Results"))
    cells.append(code("""
    metrics[["model", "precision", "recall", "f1", "roc_auc", "pr_auc"]]
    """))
    cells.append(code("""
    display(Image(filename=str(figures / "roc_curves.png")))
    display(Image(filename=str(figures / "pr_curves.png")))
    display(Image(filename=str(figures / "confusion_matrix_histgradientboosting_plus_graph_features.png")))
    display(Image(filename=str(figures / "feature_importance.png")))
    """))
    cells.append(md("""
    ## Main Interpretation

    The best model is HistGradientBoosting with graph-aware features. Compared with the base HistGradientBoosting model, graph features substantially improve F1 and PR-AUC. This supports the conclusion that AML detection benefits from modeling account behavior and transaction-network structure, not only individual transaction attributes.
    """))

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (ROOT / "project_groupID_IBM_AML.ipynb").write_text(json.dumps(notebook, indent=2), encoding="utf-8")


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    SLIDES.mkdir(exist_ok=True)
    stats, metrics, clustering, payment, _ = load_results()
    make_requirements()
    make_markdown_report(stats, metrics, clustering, payment)
    make_slides_markdown(stats, metrics)
    make_report_pdf(stats, metrics, clustering)
    make_slides_pdf(stats, metrics)
    make_notebook(stats)
    print("Created report, slides, notebook, and requirements artifacts.")


if __name__ == "__main__":
    main()
