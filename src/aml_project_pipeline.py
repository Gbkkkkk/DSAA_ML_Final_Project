from __future__ import annotations

import json
import math
import urllib.request
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN, MiniBatchKMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    calinski_harabasz_score,
    classification_report,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA_ZIP = ROOT / "data" / "HI-Small_Trans.csv.zip"
DATA_URL = (
    "https://huggingface.co/datasets/eexzzm/"
    "IBM-Transactions-for-Anti-Money-Laundering-HI-Small-Trans/resolve/main/"
    "HI-Small_Trans.csv.zip?download=true"
)
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
RANDOM_STATE = 42


def ensure_dirs() -> None:
    for path in [ROOT / "data", FIGURES, RESULTS]:
        path.mkdir(parents=True, exist_ok=True)


def ensure_data() -> None:
    if DATA_ZIP.exists():
        return
    print(f"Downloading dataset to {DATA_ZIP}...")
    urllib.request.urlretrieve(DATA_URL, DATA_ZIP)


def csv_name_in_zip() -> str:
    with zipfile.ZipFile(DATA_ZIP) as zf:
        names = [name for name in zf.namelist() if name.lower().endswith(".csv")]
    if not names:
        raise FileNotFoundError(f"No CSV file found in {DATA_ZIP}")
    return names[0]


def first_pass_stats(chunksize: int = 500_000) -> dict:
    total_rows = 0
    positive_rows = 0
    payment_counts = {}
    payment_pos = {}
    currency_counts = {}
    currency_pos = {}
    hourly_counts = {}
    hourly_pos = {}
    amount_by_label = {0: [], 1: []}

    name = csv_name_in_zip()
    with zipfile.ZipFile(DATA_ZIP) as zf:
        for chunk in pd.read_csv(zf.open(name), chunksize=chunksize):
            chunk["Timestamp"] = pd.to_datetime(chunk["Timestamp"], errors="coerce")
            chunk["hour"] = chunk["Timestamp"].dt.hour
            total_rows += len(chunk)
            positive_rows += int(chunk["Is Laundering"].sum())

            for col, count_map, pos_map in [
                ("Payment Format", payment_counts, payment_pos),
                ("Payment Currency", currency_counts, currency_pos),
                ("hour", hourly_counts, hourly_pos),
            ]:
                counts = chunk.groupby(col, dropna=False).size()
                poss = chunk.groupby(col, dropna=False)["Is Laundering"].sum()
                for key, value in counts.items():
                    count_map[str(key)] = count_map.get(str(key), 0) + int(value)
                for key, value in poss.items():
                    pos_map[str(key)] = pos_map.get(str(key), 0) + int(value)

            for label in [0, 1]:
                vals = chunk.loc[chunk["Is Laundering"] == label, "Amount Paid"].dropna()
                if len(vals):
                    amount_by_label[label].append(vals.sample(min(len(vals), 3000), random_state=RANDOM_STATE))

    stats = {
        "total_rows": total_rows,
        "positive_rows": positive_rows,
        "positive_rate": positive_rows / total_rows,
        "payment_counts": payment_counts,
        "payment_pos": payment_pos,
        "currency_counts": currency_counts,
        "currency_pos": currency_pos,
        "hourly_counts": hourly_counts,
        "hourly_pos": hourly_pos,
    }
    for label in [0, 1]:
        if amount_by_label[label]:
            sampled = pd.concat(amount_by_label[label], ignore_index=True)
            stats[f"amount_label_{label}_sample"] = sampled.sample(
                min(len(sampled), 50_000), random_state=RANDOM_STATE
            ).tolist()
    return stats


def plot_eda(stats: dict) -> None:
    sns.set_theme(style="whitegrid", context="notebook")

    with open(RESULTS / "dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    amount_df = pd.DataFrame(
        {
            "Amount Paid": stats.get("amount_label_0_sample", []) + stats.get("amount_label_1_sample", []),
            "Label": ["Legitimate"] * len(stats.get("amount_label_0_sample", []))
            + ["Laundering"] * len(stats.get("amount_label_1_sample", [])),
        }
    )
    amount_df["log10_amount"] = np.log10(amount_df["Amount Paid"].clip(lower=1e-4))
    plt.figure(figsize=(8, 5))
    sns.histplot(data=amount_df, x="log10_amount", hue="Label", bins=60, stat="density", common_norm=False)
    plt.title("Transaction Amount Distribution by Label")
    plt.xlabel("log10(Amount Paid)")
    plt.tight_layout()
    plt.savefig(FIGURES / "eda_amount_distribution.png", dpi=220)
    plt.close()

    payment = pd.DataFrame(
        [
            {
                "Payment Format": key,
                "transactions": stats["payment_counts"][key],
                "laundering": stats["payment_pos"].get(key, 0),
            }
            for key in stats["payment_counts"]
        ]
    )
    payment["laundering_rate"] = payment["laundering"] / payment["transactions"]
    payment = payment.sort_values("laundering_rate", ascending=False)
    payment.to_csv(RESULTS / "payment_format_rates.csv", index=False)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=payment, y="Payment Format", x="laundering_rate", color="#2E7D8F")
    plt.title("Laundering Rate by Payment Format")
    plt.xlabel("Laundering rate")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(FIGURES / "eda_payment_format_rate.png", dpi=220)
    plt.close()

    hourly = pd.DataFrame(
        [
            {
                "hour": int(float(key)) if key != "nan" else -1,
                "transactions": stats["hourly_counts"][key],
                "laundering": stats["hourly_pos"].get(key, 0),
            }
            for key in stats["hourly_counts"]
        ]
    ).sort_values("hour")
    hourly["laundering_rate"] = hourly["laundering"] / hourly["transactions"]
    hourly.to_csv(RESULTS / "hourly_rates.csv", index=False)
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.bar(hourly["hour"], hourly["transactions"], color="#B7B7B7", label="Transactions")
    ax1.set_ylabel("Transaction count")
    ax2 = ax1.twinx()
    ax2.plot(hourly["hour"], hourly["laundering_rate"], color="#C83E4D", marker="o", label="Laundering rate")
    ax2.set_ylabel("Laundering rate")
    ax1.set_xlabel("Hour of day")
    plt.title("Transaction Volume and Laundering Rate by Hour")
    fig.tight_layout()
    plt.savefig(FIGURES / "eda_hourly_volume_rate.png", dpi=220)
    plt.close()


def load_model_sample(stats: dict, target_negatives: int = 160_000, chunksize: int = 500_000) -> pd.DataFrame:
    total_negatives = stats["total_rows"] - stats["positive_rows"]
    neg_frac = min(1.0, target_negatives / total_negatives)
    frames = []
    name = csv_name_in_zip()
    with zipfile.ZipFile(DATA_ZIP) as zf:
        for chunk_index, chunk in enumerate(pd.read_csv(zf.open(name), chunksize=chunksize)):
            positives = chunk[chunk["Is Laundering"] == 1]
            negatives = chunk[chunk["Is Laundering"] == 0]
            neg_sample = negatives.sample(frac=neg_frac, random_state=RANDOM_STATE + chunk_index)
            frames.append(pd.concat([positives, neg_sample], ignore_index=True))
    sample = pd.concat(frames, ignore_index=True)
    sample = sample.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    sample.to_csv(RESULTS / "model_sample_preview.csv", index=False)
    return sample


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["hour"] = df["Timestamp"].dt.hour.fillna(-1).astype(int)
    df["weekday"] = df["Timestamp"].dt.weekday.fillna(-1).astype(int)
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
    df["log_amount_paid"] = np.log1p(df["Amount Paid"].clip(lower=0))
    df["log_amount_received"] = np.log1p(df["Amount Received"].clip(lower=0))
    df["amount_diff"] = df["Amount Paid"] - df["Amount Received"]
    df["same_currency"] = (df["Payment Currency"] == df["Receiving Currency"]).astype(int)
    df["same_bank"] = (df["From Bank"] == df["To Bank"]).astype(int)
    df["from_account_id"] = df["From Bank"].astype(str) + "_" + df["Account"].astype(str)
    df["to_account_id"] = df["To Bank"].astype(str) + "_" + df["Account.1"].astype(str)
    return df


def add_graph_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    sent = df.groupby("from_account_id").agg(
        sender_out_degree=("to_account_id", "size"),
        sender_unique_receivers=("to_account_id", "nunique"),
        sender_total_sent=("Amount Paid", "sum"),
        sender_mean_sent=("Amount Paid", "mean"),
    )
    received = df.groupby("to_account_id").agg(
        receiver_in_degree=("from_account_id", "size"),
        receiver_unique_senders=("from_account_id", "nunique"),
        receiver_total_received=("Amount Received", "sum"),
        receiver_mean_received=("Amount Received", "mean"),
    )
    df = df.merge(sent, left_on="from_account_id", right_index=True, how="left")
    df = df.merge(received, left_on="to_account_id", right_index=True, how="left")
    df["sent_received_ratio"] = df["sender_total_sent"] / (df["receiver_total_received"] + 1e-6)

    graph_sample = df.sample(min(len(df), 60_000), random_state=RANDOM_STATE)
    graph = nx.from_pandas_edgelist(
        graph_sample,
        source="from_account_id",
        target="to_account_id",
        edge_attr=None,
        create_using=nx.DiGraph(),
    )
    pagerank = nx.pagerank(graph, alpha=0.85, max_iter=80)
    in_degree = dict(graph.in_degree())
    out_degree = dict(graph.out_degree())
    df["sender_pagerank"] = df["from_account_id"].map(pagerank).fillna(0.0)
    df["receiver_pagerank"] = df["to_account_id"].map(pagerank).fillna(0.0)
    df["sample_sender_out_degree"] = df["from_account_id"].map(out_degree).fillna(0.0)
    df["sample_receiver_in_degree"] = df["to_account_id"].map(in_degree).fillna(0.0)
    return df


BASE_NUMERIC = [
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
]

GRAPH_NUMERIC = [
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

CATEGORICAL = [
    "Payment Currency",
    "Receiving Currency",
    "Payment Format",
]


def make_preprocessor(numeric_features: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, CATEGORICAL),
        ],
        remainder="drop",
    )


def run_clustering_and_tsne(df: pd.DataFrame) -> None:
    pos = df[df["Is Laundering"] == 1]
    neg = df[df["Is Laundering"] == 0].sample(min(8000, (df["Is Laundering"] == 0).sum()), random_state=RANDOM_STATE)
    viz = pd.concat([pos, neg], ignore_index=True).sample(frac=1.0, random_state=RANDOM_STATE)
    viz = viz.sample(min(len(viz), 12_000), random_state=RANDOM_STATE)

    preprocessor = make_preprocessor(BASE_NUMERIC + GRAPH_NUMERIC)
    x = preprocessor.fit_transform(viz)
    pca_dims = min(20, x.shape[1], x.shape[0] - 1)
    x_pca = PCA(n_components=pca_dims, random_state=RANDOM_STATE).fit_transform(x)

    tsne = TSNE(
        n_components=2,
        perplexity=35,
        init="pca",
        learning_rate="auto",
        max_iter=700,
        random_state=RANDOM_STATE,
    )
    emb = tsne.fit_transform(x_pca)
    viz["tsne_1"] = emb[:, 0]
    viz["tsne_2"] = emb[:, 1]
    viz.to_csv(RESULTS / "tsne_sample.csv", index=False)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=viz,
        x="tsne_1",
        y="tsne_2",
        hue="Is Laundering",
        palette={0: "#8A8F98", 1: "#C83E4D"},
        s=9,
        alpha=0.7,
        linewidth=0,
    )
    plt.title("t-SNE Projection Colored by Laundering Label")
    plt.tight_layout()
    plt.savefig(FIGURES / "tsne_projection.png", dpi=220)
    plt.close()

    cluster_rows = []
    kmeans = MiniBatchKMeans(n_clusters=6, random_state=RANDOM_STATE, batch_size=2048, n_init="auto")
    viz["kmeans_cluster"] = kmeans.fit_predict(x_pca)
    cluster_rows.append(evaluate_clusters("MiniBatchKMeans", x_pca, viz["kmeans_cluster"].to_numpy(), viz))

    dbscan = DBSCAN(eps=2.8, min_samples=20)
    viz["dbscan_cluster"] = dbscan.fit_predict(x_pca[:, : min(10, x_pca.shape[1])])
    cluster_rows.append(evaluate_clusters("DBSCAN", x_pca, viz["dbscan_cluster"].to_numpy(), viz))

    pd.DataFrame(cluster_rows).to_csv(RESULTS / "clustering_metrics.csv", index=False)

    for col, fname, title in [
        ("kmeans_cluster", "clustering_kmeans_tsne.png", "K-Means Clusters on t-SNE Projection"),
        ("dbscan_cluster", "clustering_dbscan_tsne.png", "DBSCAN Clusters on t-SNE Projection"),
    ]:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=viz, x="tsne_1", y="tsne_2", hue=col, palette="tab10", s=9, alpha=0.75, linewidth=0)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(FIGURES / fname, dpi=220)
        plt.close()

    profiles = []
    for method_col in ["kmeans_cluster", "dbscan_cluster"]:
        prof = (
            viz.groupby(method_col)
            .agg(
                transactions=("Is Laundering", "size"),
                laundering=("Is Laundering", "sum"),
                avg_amount=("Amount Paid", "mean"),
                avg_sender_out_degree=("sender_out_degree", "mean"),
            )
            .reset_index()
        )
        prof["laundering_rate"] = prof["laundering"] / prof["transactions"]
        prof["method"] = method_col
        profiles.append(prof)
    pd.concat(profiles, ignore_index=True).to_csv(RESULTS / "cluster_profiles.csv", index=False)


def evaluate_clusters(method: str, x: np.ndarray, labels: np.ndarray, viz: pd.DataFrame) -> dict:
    unique = set(labels)
    non_noise = labels != -1
    valid = len(unique - {-1}) >= 2 and non_noise.sum() > 10
    row = {
        "method": method,
        "n_clusters_ex_noise": len(unique - {-1}),
        "noise_rate": float((labels == -1).mean()),
        "silhouette": np.nan,
        "davies_bouldin": np.nan,
        "calinski_harabasz": np.nan,
        "highest_cluster_laundering_rate": np.nan,
    }
    if valid:
        used_x = x[non_noise] if -1 in unique else x
        used_labels = labels[non_noise] if -1 in unique else labels
        row["silhouette"] = float(silhouette_score(used_x, used_labels))
        row["davies_bouldin"] = float(davies_bouldin_score(used_x, used_labels))
        row["calinski_harabasz"] = float(calinski_harabasz_score(used_x, used_labels))
    rates = viz.assign(cluster=labels).groupby("cluster")["Is Laundering"].mean()
    if len(rates):
        row["highest_cluster_laundering_rate"] = float(rates.max())
    return row


def evaluate_classifier(name: str, model: Pipeline, x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> dict:
    model.fit(x_train, y_train)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(x_test)[:, 1]
    else:
        y_score = model.decision_function(x_test)
    y_pred_default = (y_score >= 0.5).astype(int)

    precision, recall, thresholds = precision_recall_curve(y_test, y_score)
    f1_values = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    best_idx = int(np.nanargmax(f1_values))
    best_threshold = thresholds[max(best_idx - 1, 0)] if len(thresholds) else 0.5
    y_pred_tuned = (y_score >= best_threshold).astype(int)

    report = classification_report(y_test, y_pred_tuned, output_dict=True, zero_division=0)
    row = {
        "model": name,
        "threshold": float(best_threshold),
        "accuracy": float((y_pred_tuned == y_test).mean()),
        "precision": float(precision_score(y_test, y_pred_tuned, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred_tuned, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred_tuned, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_score)),
        "pr_auc": float(average_precision_score(y_test, y_score)),
        "positive_precision": float(report["1"]["precision"]),
        "positive_recall": float(report["1"]["recall"]),
        "positive_f1": float(report["1"]["f1-score"]),
    }

    cm = confusion_matrix(y_test, y_pred_tuned)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legitimate", "Laundering"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix: {name}")
    plt.tight_layout()
    safe_name = name.lower().replace(" ", "_").replace("+", "plus")
    plt.savefig(FIGURES / f"confusion_matrix_{safe_name}.png", dpi=220)
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, y_score)
    pr_precision, pr_recall, _ = precision_recall_curve(y_test, y_score)
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(RESULTS / f"roc_{safe_name}.csv", index=False)
    pd.DataFrame({"precision": pr_precision, "recall": pr_recall}).to_csv(RESULTS / f"pr_{safe_name}.csv", index=False)
    return row


def run_supervised_models(df: pd.DataFrame) -> None:
    y = df["Is Laundering"].astype(int)
    x = df.drop(columns=["Is Laundering"])
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )

    base_pre = make_preprocessor(BASE_NUMERIC)
    graph_pre = make_preprocessor(BASE_NUMERIC + GRAPH_NUMERIC)

    models = [
        (
            "Logistic Regression",
            Pipeline(
                [
                    ("preprocess", base_pre),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=1000,
                            class_weight="balanced",
                            n_jobs=-1,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
        ),
        (
            "Decision Tree",
            Pipeline(
                [
                    ("preprocess", base_pre),
                    ("model", DecisionTreeClassifier(max_depth=9, class_weight="balanced", random_state=RANDOM_STATE)),
                ]
            ),
        ),
        (
            "Random Forest + Graph Features",
            Pipeline(
                [
                    ("preprocess", graph_pre),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=140,
                            max_depth=16,
                            min_samples_leaf=5,
                            class_weight="balanced_subsample",
                            n_jobs=-1,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
        ),
        (
            "Random Forest Base Features",
            Pipeline(
                [
                    ("preprocess", make_preprocessor(BASE_NUMERIC)),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=140,
                            max_depth=16,
                            min_samples_leaf=5,
                            class_weight="balanced_subsample",
                            n_jobs=-1,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
        ),
        (
            "HistGradientBoosting Base Features",
            Pipeline(
                [
                    ("preprocess", make_preprocessor(BASE_NUMERIC)),
                    (
                        "model",
                        HistGradientBoostingClassifier(
                            max_iter=120,
                            learning_rate=0.08,
                            max_leaf_nodes=31,
                            random_state=RANDOM_STATE,
                            class_weight="balanced",
                        ),
                    ),
                ]
            ),
        ),
        (
            "HistGradientBoosting + Graph Features",
            Pipeline(
                [
                    ("preprocess", make_preprocessor(BASE_NUMERIC + GRAPH_NUMERIC)),
                    (
                        "model",
                        HistGradientBoostingClassifier(
                            max_iter=120,
                            learning_rate=0.08,
                            max_leaf_nodes=31,
                            random_state=RANDOM_STATE,
                            class_weight="balanced",
                        ),
                    ),
                ]
            ),
        ),
    ]

    rows = []
    fitted = {}
    for name, model in models:
        print(f"Training {name}...")
        rows.append(evaluate_classifier(name, model, x_train, x_test, y_train, y_test))
        fitted[name] = model
    metrics = pd.DataFrame(rows).sort_values(["positive_f1", "pr_auc"], ascending=False)
    metrics.to_csv(RESULTS / "metrics_summary.csv", index=False)

    plt.figure(figsize=(8, 6))
    for _, row in metrics.iterrows():
        safe_name = row["model"].lower().replace(" ", "_").replace("+", "plus")
        curve = pd.read_csv(RESULTS / f"roc_{safe_name}.csv")
        plt.plot(curve["fpr"], curve["tpr"], label=f"{row['model']} AUC={row['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "roc_curves.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 6))
    for _, row in metrics.iterrows():
        safe_name = row["model"].lower().replace(" ", "_").replace("+", "plus")
        curve = pd.read_csv(RESULTS / f"pr_{safe_name}.csv")
        plt.plot(curve["recall"], curve["precision"], label=f"{row['model']} AP={row['pr_auc']:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "pr_curves.png", dpi=220)
    plt.close()

    best_rf = fitted.get("Random Forest + Graph Features")
    if best_rf is not None:
        pre = best_rf.named_steps["preprocess"]
        rf = best_rf.named_steps["model"]
        names = pre.get_feature_names_out()
        importances = pd.DataFrame({"feature": names, "importance": rf.feature_importances_})
        importances = importances.sort_values("importance", ascending=False).head(20)
        importances.to_csv(RESULTS / "feature_importance_random_forest.csv", index=False)
        plt.figure(figsize=(9, 6))
        sns.barplot(data=importances, y="feature", x="importance", color="#2E7D8F")
        plt.title("Top Random Forest Feature Importances")
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig(FIGURES / "feature_importance.png", dpi=220)
        plt.close()


def main() -> None:
    ensure_dirs()
    ensure_data()
    print("Computing full-dataset statistics...")
    stats = first_pass_stats()
    plot_eda(stats)

    print("Loading stratified model sample...")
    sample = load_model_sample(stats)
    sample = add_features(sample)
    sample = add_graph_features(sample)
    sample.to_parquet(RESULTS / "model_sample_features.parquet", index=False)
    print(f"Model sample shape: {sample.shape}")
    print(sample["Is Laundering"].value_counts(normalize=True).rename("rate"))

    print("Running t-SNE and clustering...")
    run_clustering_and_tsne(sample)

    print("Training and evaluating supervised models...")
    run_supervised_models(sample)
    print("Done. Outputs written to figures/ and results/.")


if __name__ == "__main__":
    main()
