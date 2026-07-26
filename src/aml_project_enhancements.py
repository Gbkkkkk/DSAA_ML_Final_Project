"""Open-ended AML experiments built on the core course pipeline.

This module contains account-level clustering, mutual-information feature
selection, feature-set ablations, the natural class-prior stress test, and the
legacy frozen-snapshot time audit retained as a documented failure mode.
"""

from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    adjusted_rand_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    normalized_mutual_info_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from aml_project_pipeline import (
    BASE_NUMERIC,
    CATEGORICAL,
    DATA_ZIP,
    FIGURES,
    GRAPH_NUMERIC,
    RANDOM_STATE,
    RESULTS,
    add_features,
    add_graph_features,
    csv_name_in_zip,
    make_preprocessor,
)


RAW_COLUMNS = [
    "Timestamp",
    "From Bank",
    "Account",
    "To Bank",
    "Account.1",
    "Amount Received",
    "Receiving Currency",
    "Amount Paid",
    "Payment Currency",
    "Payment Format",
    "Is Laundering",
]


ACCOUNT_FEATURES = [
    "sent_count",
    "received_count",
    "total_sent",
    "total_received",
    "mean_sent",
    "mean_received",
    "unique_receivers",
    "unique_senders",
    "in_out_count_ratio",
    "in_out_amount_ratio",
    "net_amount_flow",
    "fan_out_score",
    "fan_in_score",
    "scatter_gather_score",
    "max_sent_per_hour",
    "max_received_per_hour",
]


# ---------- Account-level clustering ----------


def purity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_pred != -1
    if mask.sum() == 0:
        return 0.0
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    total = 0
    for cluster in np.unique(y_pred):
        labels, counts = np.unique(y_true[y_pred == cluster], return_counts=True)
        if len(counts):
            total += int(counts.max())
    return total / len(y_true)


def build_account_features(df: pd.DataFrame) -> pd.DataFrame:
    sent = df.groupby("from_account_id").agg(
        sent_count=("to_account_id", "size"),
        total_sent=("Amount Paid", "sum"),
        mean_sent=("Amount Paid", "mean"),
        unique_receivers=("to_account_id", "nunique"),
        sent_laundering=("Is Laundering", "max"),
    )
    received = df.groupby("to_account_id").agg(
        received_count=("from_account_id", "size"),
        total_received=("Amount Received", "sum"),
        mean_received=("Amount Received", "mean"),
        unique_senders=("from_account_id", "nunique"),
        received_laundering=("Is Laundering", "max"),
    )
    accounts = sent.join(received, how="outer").fillna(0)
    accounts.index.name = "account_id"
    accounts = accounts.reset_index()

    sent_hour = (
        df.groupby(["from_account_id", "hour"]).size().groupby(level=0).max().rename("max_sent_per_hour")
    )
    received_hour = (
        df.groupby(["to_account_id", "hour"]).size().groupby(level=0).max().rename("max_received_per_hour")
    )
    accounts = accounts.merge(sent_hour, left_on="account_id", right_index=True, how="left")
    accounts = accounts.merge(received_hour, left_on="account_id", right_index=True, how="left")
    accounts[["max_sent_per_hour", "max_received_per_hour"]] = accounts[
        ["max_sent_per_hour", "max_received_per_hour"]
    ].fillna(0)

    accounts["account_laundering"] = (
        (accounts["sent_laundering"] > 0) | (accounts["received_laundering"] > 0)
    ).astype(int)
    accounts["in_out_count_ratio"] = accounts["received_count"] / (accounts["sent_count"] + 1)
    accounts["in_out_amount_ratio"] = accounts["total_received"] / (accounts["total_sent"] + 1)
    accounts["net_amount_flow"] = accounts["total_received"] - accounts["total_sent"]
    accounts["fan_out_score"] = accounts["unique_receivers"] / (accounts["sent_count"] + 1)
    accounts["fan_in_score"] = accounts["unique_senders"] / (accounts["received_count"] + 1)
    accounts["scatter_gather_score"] = (accounts["unique_receivers"] + 1) / (accounts["unique_senders"] + 1)

    for col in [
        "total_sent",
        "total_received",
        "mean_sent",
        "mean_received",
        "net_amount_flow",
    ]:
        accounts[f"log_abs_{col}"] = np.log1p(np.abs(accounts[col]))
    return accounts


def evaluate_cluster_labels(method: str, x: np.ndarray, labels: np.ndarray, y: np.ndarray) -> dict:
    mask = labels != -1
    row = {
        "method": method,
        "n_clusters_ex_noise": int(len(set(labels) - {-1})),
        "noise_rate": float((labels == -1).mean()),
        "silhouette": np.nan,
        "nmi": float(normalized_mutual_info_score(y[mask], labels[mask])) if mask.sum() else 0.0,
        "ari": float(adjusted_rand_score(y[mask], labels[mask])) if mask.sum() else 0.0,
        "purity": float(purity_score(y, labels)),
        "cluster_positive_rate_max": np.nan,
    }
    if row["n_clusters_ex_noise"] >= 2 and mask.sum() > 20:
        row["silhouette"] = float(silhouette_score(x[mask], labels[mask]))
    profile = pd.DataFrame({"cluster": labels, "label": y})
    if len(profile):
        row["cluster_positive_rate_max"] = float(profile.groupby("cluster")["label"].mean().max())
    return row


def run_account_level_clustering(df: pd.DataFrame) -> None:
    accounts = build_account_features(df)
    accounts.to_csv(RESULTS / "account_features.csv", index=False)

    positive_accounts = accounts[accounts["account_laundering"] == 1]
    negative_accounts = accounts[accounts["account_laundering"] == 0].sample(
        min(20000, (accounts["account_laundering"] == 0).sum()), random_state=RANDOM_STATE
    )
    account_sample = pd.concat([positive_accounts, negative_accounts], ignore_index=True).sample(
        frac=1, random_state=RANDOM_STATE
    )

    feature_cols = ACCOUNT_FEATURES + [
        "log_abs_total_sent",
        "log_abs_total_received",
        "log_abs_mean_sent",
        "log_abs_mean_received",
        "log_abs_net_amount_flow",
    ]
    x = account_sample[feature_cols].replace([np.inf, -np.inf], 0).fillna(0)
    x_scaled = StandardScaler().fit_transform(x)
    x_pca = PCA(n_components=min(12, x_scaled.shape[1]), random_state=RANDOM_STATE).fit_transform(x_scaled)
    y = account_sample["account_laundering"].to_numpy()

    kmeans = MiniBatchKMeans(n_clusters=6, random_state=RANDOM_STATE, batch_size=2048, n_init="auto")
    kmeans_labels = kmeans.fit_predict(x_pca)
    dbscan = DBSCAN(eps=1.8, min_samples=25)
    dbscan_labels = dbscan.fit_predict(x_pca[:, :8])

    account_sample["kmeans_cluster"] = kmeans_labels
    account_sample["dbscan_cluster"] = dbscan_labels
    account_sample.to_csv(RESULTS / "account_clustering_sample.csv", index=False)

    rows = [
        evaluate_cluster_labels("Account KMeans", x_pca, kmeans_labels, y),
        evaluate_cluster_labels("Account DBSCAN", x_pca, dbscan_labels, y),
    ]
    pd.DataFrame(rows).to_csv(RESULTS / "account_clustering_metrics.csv", index=False)

    profiles = []
    for col in ["kmeans_cluster", "dbscan_cluster"]:
        prof = (
            account_sample.groupby(col)
            .agg(
                accounts=("account_laundering", "size"),
                laundering_accounts=("account_laundering", "sum"),
                avg_sent_count=("sent_count", "mean"),
                avg_received_count=("received_count", "mean"),
                avg_fan_out=("fan_out_score", "mean"),
                avg_fan_in=("fan_in_score", "mean"),
                avg_scatter_gather=("scatter_gather_score", "mean"),
            )
            .reset_index()
        )
        prof["laundering_account_rate"] = prof["laundering_accounts"] / prof["accounts"]
        prof["method"] = col
        profiles.append(prof)
    pd.concat(profiles, ignore_index=True).to_csv(RESULTS / "account_cluster_profiles.csv", index=False)

    emb = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(x_scaled)
    account_sample["pca_1"] = emb[:, 0]
    account_sample["pca_2"] = emb[:, 1]
    for hue, fname, title in [
        ("account_laundering", "account_pca_by_laundering.png", "Account Behavior PCA Colored by Laundering Label"),
        ("kmeans_cluster", "account_kmeans_clusters.png", "Account-level KMeans Clusters"),
        ("dbscan_cluster", "account_dbscan_clusters.png", "Account-level DBSCAN Clusters"),
    ]:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=account_sample,
            x="pca_1",
            y="pca_2",
            hue=hue,
            s=9,
            alpha=0.7,
            linewidth=0,
            palette="tab10" if hue != "account_laundering" else {0: "#8A8F98", 1: "#C83E4D"},
        )
        plt.title(title)
        plt.tight_layout()
        plt.savefig(FIGURES / fname, dpi=220)
        plt.close()


def run_feature_selection(df: pd.DataFrame) -> None:
    """Rank encoded transaction and graph features by mutual information."""
    y = df["Is Laundering"].astype(int)
    x = df.drop(columns=["Is Laundering"])
    feature_set = BASE_NUMERIC + GRAPH_NUMERIC
    preprocessor = make_preprocessor(feature_set)
    x_trans = preprocessor.fit_transform(x)
    feature_names = preprocessor.get_feature_names_out()
    sample_size = min(60000, x_trans.shape[0])
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(x_trans.shape[0], size=sample_size, replace=False)
    mi = mutual_info_classif(x_trans[idx], y.iloc[idx], discrete_features=False, random_state=RANDOM_STATE)
    scores = pd.DataFrame({"feature": feature_names, "mutual_information": mi}).sort_values(
        "mutual_information", ascending=False
    )
    scores.to_csv(RESULTS / "mutual_information_features.csv", index=False)
    plt.figure(figsize=(9, 6))
    sns.barplot(data=scores.head(18), y="feature", x="mutual_information", color="#5B8E7D")
    plt.title("Top Features by Mutual Information")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(FIGURES / "mutual_information_features.png", dpi=220)
    plt.close()


def _score_from_scores(y_true: pd.Series, y_score: np.ndarray, threshold: float) -> dict:
    pred = (y_score >= threshold).astype(int)
    return {
        "accuracy": float((pred == y_true).mean()),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
    }


def _best_f1_threshold(y_val: pd.Series, val_score: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_val, val_score)
    if not len(thresholds):
        return 0.5
    f1_values = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1_values))])


def run_feature_selection_ablation(df: pd.DataFrame) -> None:
    """Retrain HGB on top-k feature subsets to quantify the complexity trade-off."""
    y = df["Is Laundering"].astype(int)
    x = df.drop(columns=["Is Laundering"])
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )
    x_fit, x_val, y_fit, y_val = train_test_split(
        x_train, y_train, test_size=0.20, random_state=RANDOM_STATE, stratify=y_train
    )

    preprocessor = make_preprocessor(BASE_NUMERIC + GRAPH_NUMERIC)
    x_fit_t = preprocessor.fit_transform(x_fit)
    x_val_t = preprocessor.transform(x_val)
    x_test_t = preprocessor.transform(x_test)
    feature_names = preprocessor.get_feature_names_out()

    mi = pd.read_csv(RESULTS / "mutual_information_features.csv")
    rows = []

    def train_selected(label: str, indices: np.ndarray) -> None:
        start = time.perf_counter()
        model = HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.08,
            max_leaf_nodes=31,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        )
        model.fit(x_fit_t[:, indices], y_fit)
        val_score = model.predict_proba(x_val_t[:, indices])[:, 1]
        threshold = _best_f1_threshold(y_val, val_score)
        test_score = model.predict_proba(x_test_t[:, indices])[:, 1]
        scores = _score_from_scores(y_test, test_score, threshold)
        rows.append(
            {
                "feature_set": label,
                "n_features": int(len(indices)),
                "threshold": threshold,
                "training_seconds": float(time.perf_counter() - start),
                **scores,
            }
        )

    all_indices = np.arange(len(feature_names))
    train_selected("All base+graph encoded features", all_indices)
    for k in [5, 10, 20]:
        selected_names = mi["feature"].head(k).tolist()
        indices = np.array([int(np.where(feature_names == name)[0][0]) for name in selected_names if name in feature_names])
        train_selected(f"Top-{k} mutual information features", indices)

    base_metrics = pd.read_csv(RESULTS / "metrics_summary.csv")
    for model_name, label in [
        ("HistGradientBoosting Base Features", "All base features (pipeline result)"),
        ("HistGradientBoosting + Graph Features", "All graph features (pipeline result)"),
    ]:
        row = base_metrics[base_metrics["model"] == model_name].iloc[0]
        rows.append(
            {
                "feature_set": label,
                "n_features": np.nan,
                "threshold": float(row["threshold"]),
                "training_seconds": np.nan,
                "accuracy": float(row["accuracy"]),
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "f1": float(row["f1"]),
                "roc_auc": float(row["roc_auc"]),
                "pr_auc": float(row["pr_auc"]),
            }
        )
    pd.DataFrame(rows).to_csv(RESULTS / "feature_selection_ablation.csv", index=False)


def add_graph_features_from_history(target: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Map one frozen history snapshot onto target rows (legacy audit design)."""
    target = target.copy()
    sent = history.groupby("from_account_id").agg(
        sender_out_degree=("to_account_id", "size"),
        sender_unique_receivers=("to_account_id", "nunique"),
        sender_total_sent=("Amount Paid", "sum"),
        sender_mean_sent=("Amount Paid", "mean"),
    )
    received = history.groupby("to_account_id").agg(
        receiver_in_degree=("from_account_id", "size"),
        receiver_unique_senders=("from_account_id", "nunique"),
        receiver_total_received=("Amount Received", "sum"),
        receiver_mean_received=("Amount Received", "mean"),
    )
    target = target.merge(sent, left_on="from_account_id", right_index=True, how="left")
    target = target.merge(received, left_on="to_account_id", right_index=True, how="left")
    for col in [
        "sender_out_degree",
        "sender_unique_receivers",
        "sender_total_sent",
        "sender_mean_sent",
        "receiver_in_degree",
        "receiver_unique_senders",
        "receiver_total_received",
        "receiver_mean_received",
    ]:
        target[col] = target[col].fillna(0.0)
    target["sent_received_ratio"] = target["sender_total_sent"] / (target["receiver_total_received"] + 1e-6)

    graph_sample = history.sample(min(len(history), 60_000), random_state=RANDOM_STATE)
    graph = __import__("networkx").from_pandas_edgelist(
        graph_sample,
        source="from_account_id",
        target="to_account_id",
        create_using=__import__("networkx").DiGraph(),
    )
    pagerank = __import__("networkx").pagerank(graph, alpha=0.85, max_iter=80)
    out_degree = dict(graph.out_degree())
    in_degree = dict(graph.in_degree())
    target["sender_pagerank"] = target["from_account_id"].map(pagerank).fillna(0.0)
    target["receiver_pagerank"] = target["to_account_id"].map(pagerank).fillna(0.0)
    target["sample_sender_out_degree"] = target["from_account_id"].map(out_degree).fillna(0.0)
    target["sample_receiver_in_degree"] = target["to_account_id"].map(in_degree).fillna(0.0)
    return target


def evaluate_temporal_protocol(df: pd.DataFrame) -> None:
    """Reproduce the frozen-snapshot audit that motivated strict online history."""
    raw = df[RAW_COLUMNS].copy()
    raw["Timestamp"] = pd.to_datetime(raw["Timestamp"], errors="coerce")
    raw = raw.sort_values("Timestamp").reset_index(drop=True)
    n = len(raw)
    train_raw = raw.iloc[: int(n * 0.60)].copy()
    val_raw = raw.iloc[int(n * 0.60) : int(n * 0.80)].copy()
    test_raw = raw.iloc[int(n * 0.80) :].copy()

    train_base = add_features(train_raw)
    val_base = add_features(val_raw)
    test_base = add_features(test_raw)
    train_graph = add_graph_features_from_history(train_base, train_base)
    val_graph = add_graph_features_from_history(val_base, train_base)
    test_graph = add_graph_features_from_history(test_base, train_base)

    rows = []
    for label, features, train_df, val_df, test_df in [
        ("HGB Base, time split", BASE_NUMERIC, train_base, val_base, test_base),
        ("HGB Graph, frozen train snapshot", BASE_NUMERIC + GRAPH_NUMERIC, train_graph, val_graph, test_graph),
    ]:
        y_train = train_df["Is Laundering"].astype(int)
        y_val = val_df["Is Laundering"].astype(int)
        y_test = test_df["Is Laundering"].astype(int)
        x_train = train_df.drop(columns=["Is Laundering"])
        x_val = val_df.drop(columns=["Is Laundering"])
        x_test = test_df.drop(columns=["Is Laundering"])

        model = Pipeline(
            [
                ("preprocess", make_preprocessor(features)),
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
        )
        model.fit(x_train, y_train)
        val_score = model.predict_proba(x_val)[:, 1]
        threshold = _best_f1_threshold(y_val, val_score)
        test_score = model.predict_proba(x_test)[:, 1]
        scores = _score_from_scores(y_test, test_score, threshold)
        rows.append(
            {
                "protocol": "time_based_60_20_20_frozen_snapshot",
                "model": label,
                "train_rows": len(train_df),
                "validation_rows": len(val_df),
                "test_rows": len(test_df),
                "train_positive_rate": float(y_train.mean()),
                "validation_positive_rate": float(y_val.mean()),
                "test_positive_rate": float(y_test.mean()),
                "threshold": threshold,
                **scores,
            }
        )

    pd.DataFrame(rows).to_csv(RESULTS / "time_based_leakage_check.csv", index=False)


def load_negative_natural_sample(needed_negatives: int, chunksize: int = 500_000) -> pd.DataFrame:
    frames = []
    # Over-sample chunkwise very lightly, then trim. This keeps memory controlled.
    total_negative = 5_078_345 - 5_177
    frac = min(1.0, needed_negatives / total_negative * 1.25)
    with zipfile.ZipFile(DATA_ZIP) as zf:
        name = csv_name_in_zip()
        for i, chunk in enumerate(pd.read_csv(zf.open(name), chunksize=chunksize)):
            neg = chunk[chunk["Is Laundering"] == 0]
            if len(neg):
                frames.append(neg.sample(frac=frac, random_state=RANDOM_STATE + i))
            if sum(len(f) for f in frames) >= needed_negatives * 1.05:
                break
    negs = pd.concat(frames, ignore_index=True)
    return negs.sample(min(len(negs), needed_negatives), random_state=RANDOM_STATE).reset_index(drop=True)


def run_natural_distribution_stress_test(df: pd.DataFrame) -> None:
    """Restore the full-data class prior and translate scores into alert load."""
    raw_cols = [
        "Timestamp",
        "From Bank",
        "Account",
        "To Bank",
        "Account.1",
        "Amount Received",
        "Receiving Currency",
        "Amount Paid",
        "Payment Currency",
        "Payment Format",
        "Is Laundering",
    ]
    y = df["Is Laundering"].astype(int)
    x = df.drop(columns=["Is Laundering"])
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )
    x_fit, x_val, y_fit, y_val = train_test_split(
        x_train, y_train, test_size=0.20, random_state=RANDOM_STATE, stratify=y_train
    )

    model = Pipeline(
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
    )
    model.fit(x_fit, y_fit)
    val_score = model.predict_proba(x_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, val_score)
    f1_values = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    threshold = float(thresholds[int(np.nanargmax(f1_values[:-1]))])

    positive_holdout = x_test[y_test == 1].copy()
    positive_holdout["Is Laundering"] = 1
    positive_holdout = positive_holdout[raw_cols]
    full_rate = json.loads((RESULTS / "dataset_stats.json").read_text(encoding="utf-8"))["positive_rate"]
    needed_negatives = int(round(len(positive_holdout) * (1 - full_rate) / full_rate))
    needed_negatives = min(needed_negatives, 1_500_000)
    negatives = load_negative_natural_sample(needed_negatives)
    negatives = negatives[raw_cols]
    natural_eval = pd.concat([positive_holdout, negatives], ignore_index=True).sample(
        frac=1, random_state=RANDOM_STATE
    )
    natural_eval = add_features(natural_eval)
    natural_eval = add_graph_features(natural_eval)

    y_eval = natural_eval["Is Laundering"].astype(int)
    x_eval = natural_eval.drop(columns=["Is Laundering"])
    score = model.predict_proba(x_eval)[:, 1]
    pred = (score >= threshold).astype(int)
    cm = confusion_matrix(y_eval, pred)
    row = {
        "model": "HistGradientBoosting + Graph Features",
        "evaluation": "natural_class_prior_stress_test",
        "rows": int(len(natural_eval)),
        "positive_rows": int(y_eval.sum()),
        "positive_rate": float(y_eval.mean()),
        "threshold_from_validation": threshold,
        "accuracy": float((pred == y_eval).mean()),
        "precision": float(precision_score(y_eval, pred, zero_division=0)),
        "recall": float(recall_score(y_eval, pred, zero_division=0)),
        "f1": float(f1_score(y_eval, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_eval, score)),
        "pr_auc": float(average_precision_score(y_eval, score)),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }
    pd.DataFrame([row]).to_csv(RESULTS / "natural_distribution_evaluation.csv", index=False)

    sweep_rows = []
    candidate_thresholds = [threshold, 0.95, 0.98, 0.99, 0.995, 0.999]
    for alert_rate in [0.10, 0.05, 0.02, 0.01, 0.005, 0.001]:
        candidate_thresholds.append(float(np.quantile(score, 1 - alert_rate)))
    for th in sorted(set(round(t, 10) for t in candidate_thresholds)):
        pred_sweep = (score >= th).astype(int)
        cm_sweep = confusion_matrix(y_eval, pred_sweep)
        sweep_rows.append(
            {
                "threshold": th,
                "alert_rate": float(pred_sweep.mean()),
                "alerts": int(pred_sweep.sum()),
                "precision": float(precision_score(y_eval, pred_sweep, zero_division=0)),
                "recall": float(recall_score(y_eval, pred_sweep, zero_division=0)),
                "f1": float(f1_score(y_eval, pred_sweep, zero_division=0)),
                "tn": int(cm_sweep[0, 0]),
                "fp": int(cm_sweep[0, 1]),
                "fn": int(cm_sweep[1, 0]),
                "tp": int(cm_sweep[1, 1]),
            }
        )
    pd.DataFrame(sweep_rows).sort_values("threshold").to_csv(
        RESULTS / "natural_threshold_sweep.csv", index=False
    )

    plt.figure(figsize=(5.5, 4.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Legitimate", "Laundering"], yticklabels=["Legitimate", "Laundering"])
    plt.title("Natural Class-prior Stress Test Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(FIGURES / "confusion_matrix_natural_distribution.png", dpi=220)
    plt.close()


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    df = pd.read_parquet(RESULTS / "model_sample_features.parquet")
    run_account_level_clustering(df)
    run_feature_selection(df)
    run_feature_selection_ablation(df)
    evaluate_temporal_protocol(df)
    run_natural_distribution_stress_test(df)
    print("Enhanced AML analyses completed.")


if __name__ == "__main__":
    main()
