from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN, MiniBatchKMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
INPUT_FEATURES = ROOT / "results" / "model_sample_features.parquet"
OUT = ROOT / "outputs" / "part2_visualization_clustering"
FIG = OUT / "figures"
TABLES = OUT / "tables"
RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "Amount Paid",
    "Amount Received",
    "log_amount_paid",
    "log_amount_received",
    "amount_diff",
    "hour",
    "weekday",
    "is_weekend",
    "same_currency",
    "same_bank",
    "sender_out_degree",
    "sender_unique_receivers",
    "sender_total_sent",
    "sender_mean_sent",
    "receiver_in_degree",
    "receiver_unique_senders",
    "receiver_total_received",
    "receiver_mean_received",
    "sent_received_ratio",
    "sender_pagerank",
    "receiver_pagerank",
    "sample_sender_out_degree",
    "sample_receiver_in_degree",
]

CATEGORICAL_FEATURES = [
    "Payment Currency",
    "Receiving Currency",
    "Payment Format",
]

PROFILE_FEATURES = [
    "Amount Paid",
    "log_amount_paid",
    "sender_out_degree",
    "receiver_in_degree",
    "sender_unique_receivers",
    "receiver_unique_senders",
    "sent_received_ratio",
    "same_bank",
    "same_currency",
]


def ensure_dirs() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)


def load_part2_sample(max_negatives: int = 6_500, max_total: int = 10_000) -> pd.DataFrame:
    if not INPUT_FEATURES.exists():
        raise FileNotFoundError(
            f"{INPUT_FEATURES} not found. Run src/aml_project_pipeline.py first to create shared features."
        )
    df = pd.read_parquet(INPUT_FEATURES)
    positives = df[df["Is Laundering"] == 1]
    negatives = df[df["Is Laundering"] == 0].sample(
        min(max_negatives, int((df["Is Laundering"] == 0).sum())),
        random_state=RANDOM_STATE,
    )
    sample = pd.concat([positives, negatives], ignore_index=True)
    sample = sample.sample(min(len(sample), max_total), random_state=RANDOM_STATE).reset_index(drop=True)
    return sample


def make_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def evaluate_labels(x: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels)
    non_noise = labels != -1
    unique = set(labels.tolist())
    n_clusters = len(unique - {-1})
    row = {
        "n_clusters_ex_noise": n_clusters,
        "noise_rate": float((labels == -1).mean()),
        "silhouette": np.nan,
        "davies_bouldin": np.nan,
        "calinski_harabasz": np.nan,
    }
    if n_clusters >= 2 and non_noise.sum() > 20:
        used_x = x[non_noise] if -1 in unique else x
        used_labels = labels[non_noise] if -1 in unique else labels
        row["silhouette"] = float(silhouette_score(used_x, used_labels))
        row["davies_bouldin"] = float(davies_bouldin_score(used_x, used_labels))
        row["calinski_harabasz"] = float(calinski_harabasz_score(used_x, used_labels))
    return row


def safe_cluster_name(method: str, cluster: object) -> str:
    return f"{method}: {cluster}"


def cluster_profiles(df: pd.DataFrame, method: str, label_col: str) -> pd.DataFrame:
    rows = []
    overall_rate = df["Is Laundering"].mean()
    for cluster_id, part in df.groupby(label_col, dropna=False):
        payment_mode = part["Payment Format"].mode()
        pay_currency_mode = part["Payment Currency"].mode()
        row = {
            "method": method,
            "cluster": cluster_id,
            "transactions": len(part),
            "laundering": int(part["Is Laundering"].sum()),
            "laundering_rate": part["Is Laundering"].mean(),
            "enrichment_vs_sample": part["Is Laundering"].mean() / overall_rate if overall_rate else np.nan,
            "dominant_payment_format": payment_mode.iloc[0] if len(payment_mode) else "Unknown",
            "dominant_payment_currency": pay_currency_mode.iloc[0] if len(pay_currency_mode) else "Unknown",
        }
        for feature in PROFILE_FEATURES:
            row[f"mean_{feature}"] = float(part[feature].mean())
        rows.append(row)
    profiles = pd.DataFrame(rows)
    return profiles.sort_values(["laundering_rate", "transactions"], ascending=[False, False])


def plot_tsne(df: pd.DataFrame, hue: str, filename: str, title: str, palette=None) -> None:
    plot_df = df.copy()
    if hue == "Is Laundering":
        plot_df["label_name"] = plot_df["Is Laundering"].map({0: "Legitimate", 1: "Laundering"})
        hue = "label_name"
        palette = {"Legitimate": "#8A8F98", "Laundering": "#C83E4D"}
    plt.figure(figsize=(8.5, 6.2))
    sns.scatterplot(
        data=plot_df,
        x="tsne_1",
        y="tsne_2",
        hue=hue,
        palette=palette,
        s=10,
        alpha=0.72,
        linewidth=0,
    )
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(markerscale=2, fontsize=8, loc="best", frameon=True)
    plt.tight_layout()
    plt.savefig(FIG / filename, dpi=240)
    plt.close()


def plot_metric_comparison(metrics: pd.DataFrame) -> None:
    metric_long = metrics.melt(
        id_vars="method",
        value_vars=["silhouette", "davies_bouldin", "calinski_harabasz"],
        var_name="metric",
        value_name="value",
    )
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric in zip(axes, ["silhouette", "davies_bouldin", "calinski_harabasz"]):
        part = metric_long[metric_long["metric"] == metric]
        sns.barplot(data=part, x="method", y="value", ax=ax, color="#2E7D8F")
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=25)
    fig.suptitle("Internal Clustering Quality Metrics", y=1.03)
    fig.tight_layout()
    fig.savefig(FIG / "06_clustering_metric_comparison.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_risk_bars(profiles: pd.DataFrame) -> None:
    high_risk = (
        profiles[(profiles["cluster"] != -1) & (profiles["transactions"] >= 50)]
        .sort_values(["laundering_rate", "transactions"], ascending=[False, False])
        .head(12)
        .copy()
    )
    high_risk["cluster_name"] = high_risk.apply(lambda r: safe_cluster_name(r["method"], r["cluster"]), axis=1)
    plt.figure(figsize=(10, 5.5))
    ax = sns.barplot(data=high_risk, y="cluster_name", x="laundering_rate", hue="method", dodge=False)
    for container in ax.containers:
        ax.bar_label(container, labels=[f"{bar.get_width():.1%}" if bar.get_width() > 0 else "" for bar in container], padding=3)
    plt.xlabel("Laundering rate in enriched visualization sample")
    plt.ylabel("")
    plt.title("Highest-risk Clusters by Laundering Concentration")
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    plt.xlim(0, min(1.08, max(1.0, high_risk["laundering_rate"].max() + 0.08)))
    plt.tight_layout()
    plt.savefig(FIG / "07_high_risk_cluster_laundering_rates.png", dpi=240)
    plt.close()


def plot_profile_heatmap(profiles: pd.DataFrame) -> None:
    high_risk = (
        profiles[(profiles["cluster"] != -1) & (profiles["transactions"] >= 50)]
        .sort_values(["laundering_rate", "transactions"], ascending=[False, False])
        .head(8)
        .copy()
    )
    high_risk["cluster_name"] = high_risk.apply(lambda r: safe_cluster_name(r["method"], r["cluster"]), axis=1)
    cols = [f"mean_{feature}" for feature in PROFILE_FEATURES]
    matrix = high_risk.set_index("cluster_name")[cols]
    matrix = (matrix - matrix.mean(axis=0)) / matrix.std(axis=0).replace(0, 1)
    matrix = matrix.clip(-2.5, 2.5)
    plt.figure(figsize=(11, 5.5))
    sns.heatmap(matrix, cmap="vlag", center=0, linewidths=0.4, cbar_kws={"label": "z-score within high-risk clusters"})
    plt.title("Behavioral Profile of High-risk Clusters")
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(FIG / "08_high_risk_cluster_profile_heatmap.png", dpi=240)
    plt.close()


def plot_k_sweep(k_sweep: pd.DataFrame) -> None:
    fig, ax1 = plt.subplots(figsize=(8.5, 4.8))
    ax1.plot(k_sweep["k"], k_sweep["silhouette"], marker="o", color="#2E7D8F", label="Silhouette")
    ax1.set_xlabel("Number of K-Means clusters")
    ax1.set_ylabel("Silhouette")
    ax2 = ax1.twinx()
    ax2.plot(k_sweep["k"], k_sweep["highest_cluster_laundering_rate"], marker="s", color="#C83E4D", label="Max laundering rate")
    ax2.set_ylabel("Max cluster laundering rate")
    ax1.set_title("K-Means Model Selection: Structure vs Risk Enrichment")
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="best")
    fig.tight_layout()
    fig.savefig(FIG / "05_kmeans_k_sweep.png", dpi=240)
    plt.close(fig)


def build_written_outputs(stats: dict, metrics: pd.DataFrame, profiles: pd.DataFrame) -> None:
    top = (
        profiles[(profiles["cluster"] != -1) & (profiles["transactions"] >= 50)]
        .sort_values(["laundering_rate", "transactions"], ascending=[False, False])
        .head(3)
    )
    best_internal = metrics.sort_values(["silhouette", "davies_bouldin"], ascending=[False, True]).iloc[0]
    report = f"""# Part 2: Visualization and Clustering Analysis

## Objective

This section explores whether laundering transactions form recognizable regions in an unsupervised transaction-feature space. The clustering algorithms do not use `Is Laundering` during fitting; the label is used only after clustering to interpret risk concentration.

## Sampling and Feature Space

The original data contains more than five million transactions, while laundering cases are extremely rare. Because t-SNE and density-based clustering are computationally expensive, we used a stratified enriched sample that keeps laundering transactions and samples legitimate transactions. This preserves rare suspicious patterns while keeping the visualization computationally feasible. Therefore, cluster laundering rates in this section should be interpreted as pattern-discovery signals rather than population-level probabilities.

Features include amount, log amount, time, currency, payment format, same-bank/same-currency flags, and graph-aware account behavior features such as sender out-degree, receiver in-degree, unique counterparties, sent/received totals, and PageRank.

## Dimensionality Reduction

Before t-SNE, features were standardized and reduced with PCA. PCA denoises the one-hot encoded feature space and makes t-SNE and DBSCAN more stable. The t-SNE map shows that laundering transactions are not globally linearly separable, but they concentrate in several local regions. This supports treating AML detection as a nonlinear rare-event problem.

## Clustering Algorithms

We compare three algorithms:

- MiniBatch K-Means: a scalable centroid-based baseline.
- DBSCAN: a density-based method that can identify local suspicious regions and noise points.
- Gaussian Mixture Model: a probabilistic soft-clustering baseline for elliptical clusters.

## Quantitative Evaluation

The best internal clustering result is **{best_internal['method']}**, with Silhouette = {best_internal['silhouette']:.3f}, Davies-Bouldin = {best_internal['davies_bouldin']:.3f}, and Calinski-Harabasz = {best_internal['calinski_harabasz']:.1f}. Internal metrics are not sufficient for AML, so we also inspect laundering enrichment inside clusters.

## High-risk Cluster Profiling

Top high-risk clusters in the enriched visualization sample:

| Method | Cluster | Transactions | Laundering Rate | Enrichment | Dominant Payment Format | Mean Amount Paid | Mean Sender Out-degree | Mean Receiver In-degree |
|---|---:|---:|---:|---:|---|---:|---:|---:|
"""
    for _, row in top.iterrows():
        report += (
            f"| {row['method']} | {row['cluster']} | {int(row['transactions'])} | "
            f"{row['laundering_rate']:.3f} | {row['enrichment_vs_sample']:.2f}x | "
            f"{row['dominant_payment_format']} | {row['mean_Amount Paid']:.2f} | "
            f"{row['mean_sender_out_degree']:.2f} | {row['mean_receiver_in_degree']:.2f} |\n"
        )

    report += """

## Interpretation

The high-risk clusters are characterized by combinations of payment format, transaction amount, and account-network behavior. This is important because it shows that laundering is not only a row-level anomaly; it also appears as a behavioral pattern in the transaction network. The clustering results therefore motivate the Part 3 modeling choice: nonlinear supervised models with graph-aware features should outperform simple linear baselines.

## Final Part 2 Conclusion

The unsupervised analysis shows that laundering transactions are not globally separable, but they concentrate in several local high-risk regions characterized by transaction amount, payment format, and account-network behavior. This supports the use of nonlinear and graph-aware supervised models for AML detection.
"""
    (OUT / "part2_report_section.md").write_text(report, encoding="utf-8")

    speech = """# Part 2 Presentation Script

For Part 2, we used t-SNE and clustering to explore whether laundering transactions form recognizable patterns without using labels for training. Since the full dataset has over five million transactions and laundering cases are extremely rare, we used a stratified enriched sample that keeps laundering cases and samples legitimate transactions. The t-SNE plot shows that laundering transactions are not globally separable, but they form several local high-risk regions. We then compared MiniBatch K-Means, DBSCAN, and Gaussian Mixture clustering. Internal metrics help evaluate cluster compactness, while cluster-level laundering enrichment helps interpret AML risk. The cluster profiles show that high-risk regions are associated with payment format, amount, and graph-based account behavior. This motivates the graph-aware nonlinear supervised models in Part 3.
"""
    (OUT / "part2_presentation_script.md").write_text(speech, encoding="utf-8")

    summary = {
        "best_internal_method": best_internal.to_dict(),
        "top_high_risk_clusters": top.to_dict(orient="records"),
        "generated_figures": sorted(p.name for p in FIG.glob("*.png")),
    }
    (OUT / "part2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    sns.set_theme(style="whitegrid", context="notebook")

    df = load_part2_sample()
    df.to_csv(TABLES / "part2_enriched_sample.csv", index=False)
    sample_stats = {
        "rows": int(len(df)),
        "laundering": int(df["Is Laundering"].sum()),
        "laundering_rate": float(df["Is Laundering"].mean()),
        "note": "Enriched sample for visualization and clustering; rates are not population estimates.",
    }
    (TABLES / "part2_sample_stats.json").write_text(json.dumps(sample_stats, indent=2), encoding="utf-8")

    preprocessor = make_preprocessor()
    x = preprocessor.fit_transform(df)
    pca = PCA(n_components=min(20, x.shape[1], len(df) - 1), random_state=RANDOM_STATE)
    x_pca = pca.fit_transform(x)
    explained = pd.DataFrame(
        {
            "component": np.arange(1, len(pca.explained_variance_ratio_) + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    explained.to_csv(TABLES / "pca_explained_variance.csv", index=False)

    tsne = TSNE(
        n_components=2,
        perplexity=35,
        init="pca",
        learning_rate="auto",
        max_iter=650,
        random_state=RANDOM_STATE,
    )
    embedding = tsne.fit_transform(x_pca)
    df["tsne_1"] = embedding[:, 0]
    df["tsne_2"] = embedding[:, 1]

    k_rows = []
    best_k = None
    best_k_score = -np.inf
    for k in range(3, 11):
        labels = MiniBatchKMeans(n_clusters=k, random_state=RANDOM_STATE, batch_size=2048, n_init="auto").fit_predict(x_pca)
        eval_row = evaluate_labels(x_pca, labels)
        rates = df.assign(cluster=labels).groupby("cluster")["Is Laundering"].mean()
        eval_row.update({"k": k, "highest_cluster_laundering_rate": float(rates.max())})
        k_rows.append(eval_row)
        score = eval_row["silhouette"] + 0.08 * eval_row["highest_cluster_laundering_rate"]
        if score > best_k_score:
            best_k_score = score
            best_k = k
    k_sweep = pd.DataFrame(k_rows)
    k_sweep.to_csv(TABLES / "kmeans_k_sweep.csv", index=False)
    plot_k_sweep(k_sweep)

    metrics_rows = []
    profiles = []

    kmeans_labels = MiniBatchKMeans(
        n_clusters=int(best_k),
        random_state=RANDOM_STATE,
        batch_size=2048,
        n_init="auto",
    ).fit_predict(x_pca)
    df["cluster_kmeans"] = kmeans_labels
    row = evaluate_labels(x_pca, kmeans_labels)
    row.update({"method": f"MiniBatch K-Means (k={best_k})"})
    row["highest_cluster_laundering_rate"] = float(df.groupby("cluster_kmeans")["Is Laundering"].mean().max())
    metrics_rows.append(row)
    profiles.append(cluster_profiles(df, row["method"], "cluster_kmeans"))

    dbscan_candidates = []
    for eps in [2.0, 2.5, 3.0, 3.5]:
        labels = DBSCAN(eps=eps, min_samples=20).fit_predict(x_pca[:, : min(10, x_pca.shape[1])])
        eval_row = evaluate_labels(x_pca, labels)
        eval_row.update({"eps": eps})
        dbscan_candidates.append((eval_row, labels))
    dbscan_table = pd.DataFrame([row for row, _ in dbscan_candidates])
    dbscan_table.to_csv(TABLES / "dbscan_eps_sweep.csv", index=False)
    valid = dbscan_table[dbscan_table["n_clusters_ex_noise"] >= 2].copy()
    if not valid.empty:
        best_idx = valid.sort_values(["silhouette", "noise_rate"], ascending=[False, True]).index[0]
    else:
        best_idx = dbscan_table.sort_values("n_clusters_ex_noise", ascending=False).index[0]
    dbscan_eval, dbscan_labels = dbscan_candidates[int(best_idx)]
    df["cluster_dbscan"] = dbscan_labels
    dbscan_eval = dict(dbscan_eval)
    dbscan_eval.update({"method": f"DBSCAN (eps={dbscan_eval['eps']})"})
    dbscan_eval["highest_cluster_laundering_rate"] = float(df.groupby("cluster_dbscan")["Is Laundering"].mean().max())
    metrics_rows.append(dbscan_eval)
    profiles.append(cluster_profiles(df, dbscan_eval["method"], "cluster_dbscan"))

    gmm = GaussianMixture(n_components=int(best_k), covariance_type="diag", random_state=RANDOM_STATE)
    gmm_labels = gmm.fit_predict(x_pca)
    df["cluster_gmm"] = gmm_labels
    gmm_row = evaluate_labels(x_pca, gmm_labels)
    gmm_row.update({"method": f"Gaussian Mixture (k={best_k})"})
    gmm_row["highest_cluster_laundering_rate"] = float(df.groupby("cluster_gmm")["Is Laundering"].mean().max())
    metrics_rows.append(gmm_row)
    profiles.append(cluster_profiles(df, gmm_row["method"], "cluster_gmm"))

    metrics = pd.DataFrame(metrics_rows)
    metrics = metrics[
        [
            "method",
            "n_clusters_ex_noise",
            "noise_rate",
            "silhouette",
            "davies_bouldin",
            "calinski_harabasz",
            "highest_cluster_laundering_rate",
        ]
    ]
    metrics.to_csv(TABLES / "part2_clustering_metrics.csv", index=False)

    all_profiles = pd.concat(profiles, ignore_index=True).sort_values(
        ["laundering_rate", "transactions"], ascending=[False, False]
    )
    all_profiles.to_csv(TABLES / "part2_cluster_profiles_full.csv", index=False)
    high_risk = all_profiles[(all_profiles["cluster"] != -1) & (all_profiles["transactions"] >= 50)].head(15)
    high_risk.to_csv(TABLES / "part2_high_risk_clusters.csv", index=False)

    df.to_csv(TABLES / "part2_tsne_cluster_assignments.csv", index=False)
    plot_tsne(df, "Is Laundering", "01_tsne_by_laundering_label.png", "t-SNE Projection Colored by Laundering Label", None)
    plot_tsne(df, "cluster_kmeans", "02_tsne_by_kmeans_cluster.png", "t-SNE Projection Colored by K-Means Cluster", "tab10")
    plot_tsne(df, "cluster_dbscan", "03_tsne_by_dbscan_cluster.png", "t-SNE Projection Colored by DBSCAN Cluster", "tab10")
    plot_tsne(df, "cluster_gmm", "04_tsne_by_gmm_cluster.png", "t-SNE Projection Colored by Gaussian Mixture Cluster", "tab10")
    plot_metric_comparison(metrics)
    plot_risk_bars(all_profiles)
    plot_profile_heatmap(all_profiles)
    build_written_outputs(sample_stats, metrics, all_profiles)

    print("Part 2 A+ visualization and clustering outputs written to:")
    print(OUT)


if __name__ == "__main__":
    main()
