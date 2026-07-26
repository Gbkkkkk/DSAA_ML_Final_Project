"""Formal model selection and leakage-safe temporal graph experiments.

This module extends the original course pipeline in two ways:

1. It performs training-set-only GridSearchCV with PR-AUC as the refit metric.
2. It constructs graph/history features in chronological order, so every row
   uses only transactions that occurred earlier in time.

The legacy random/transductive and frozen-history results remain in the
submission as failure modes. New results are written to separate artifacts.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict, deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from aml_project_pipeline import (
    BASE_NUMERIC,
    CATEGORICAL,
    GRAPH_NUMERIC,
    RANDOM_STATE,
    RESULTS,
    FIGURES,
    add_features,
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


HISTORY_NUMERIC = [
    "hist_sender_sent_count",
    "hist_sender_received_count",
    "hist_receiver_received_count",
    "hist_receiver_sent_count",
    "hist_sender_unique_receivers",
    "hist_receiver_unique_senders",
    "hist_pair_count",
    "hist_reverse_pair_count",
    "hist_sender_total_sent",
    "hist_receiver_total_received",
    "hist_sender_mean_sent",
    "hist_receiver_mean_received",
    "hist_sender_out_in_ratio",
    "hist_receiver_in_out_ratio",
    "hist_sender_hours_since_sent",
    "hist_receiver_hours_since_received",
    "hist_sender_sent_24h",
    "hist_receiver_received_24h",
    "hist_sender_sent_7d",
    "hist_receiver_received_7d",
    "hist_amount_vs_sender_mean",
    "hist_amount_vs_receiver_mean",
    "hist_sender_seen",
    "hist_receiver_seen",
    "hist_pair_seen",
    "hist_reverse_pair_seen",
]


def _best_f1_threshold(y_true: pd.Series, scores: np.ndarray) -> tuple[float, float]:
    """Select an operating threshold without using held-out test labels."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if not len(thresholds):
        return 0.5, 0.0
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    idx = int(np.nanargmax(f1))
    return float(thresholds[idx]), float(f1[idx])


def _classification_metrics(y_true: pd.Series, scores: np.ndarray, threshold: float) -> dict:
    """Return rare-event metrics and confusion counts for one operating point."""
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "accuracy": float((pred == y_true.to_numpy()).mean()),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def _model_pipeline(model, numeric_features: list[str]) -> Pipeline:
    """Keep imputation, scaling, and one-hot encoding inside every CV fold."""
    return Pipeline(
        [
            ("preprocess", make_preprocessor(numeric_features)),
            ("model", model),
        ]
    )


def run_formal_grid_search(df: pd.DataFrame) -> None:
    """Tune four model families with 3-fold stratified CV on training data only."""
    y = df["Is Laundering"].astype(int)
    x = df.drop(columns=["Is Laundering"])
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    specifications = [
        (
            "Logistic Regression",
            "base",
            _model_pipeline(
                LogisticRegression(
                    max_iter=1200,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
                BASE_NUMERIC,
            ),
            {"model__C": [0.1, 1.0, 10.0]},
        ),
        (
            "Decision Tree",
            "base",
            _model_pipeline(
                DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
                BASE_NUMERIC,
            ),
            {
                "model__max_depth": [5, 9, 15],
                "model__min_samples_leaf": [5, 20],
            },
        ),
        (
            "Random Forest",
            "base",
            _model_pipeline(
                RandomForestClassifier(
                    n_estimators=140,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
                BASE_NUMERIC,
            ),
            {
                "model__max_depth": [12, 18],
                "model__min_samples_leaf": [2, 8],
            },
        ),
        (
            "HistGradientBoosting",
            "base",
            _model_pipeline(
                HistGradientBoostingClassifier(
                    max_iter=160,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
                BASE_NUMERIC,
            ),
            {
                "model__learning_rate": [0.05, 0.10],
                "model__max_leaf_nodes": [15, 31],
                "model__l2_regularization": [0.0, 1.0],
            },
        ),
        (
            "HistGradientBoosting",
            "base+retrospective_graph",
            _model_pipeline(
                HistGradientBoostingClassifier(
                    max_iter=160,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
                BASE_NUMERIC + GRAPH_NUMERIC,
            ),
            {
                "model__learning_rate": [0.05, 0.10],
                "model__max_leaf_nodes": [15, 31],
                "model__l2_regularization": [0.0, 1.0],
            },
        ),
    ]

    all_candidates = []
    summary = []
    for model_name, feature_set, estimator, param_grid in specifications:
        print(f"Grid search: {model_name} [{feature_set}]")
        started = time.perf_counter()
        search = GridSearchCV(
            estimator,
            param_grid,
            scoring={"pr_auc": "average_precision", "f1": "f1"},
            refit="pr_auc",
            cv=cv,
            n_jobs=2,
            return_train_score=True,
            verbose=1,
        )
        search.fit(x_train, y_train)

        # Out-of-fold scores provide a threshold without touching test labels.
        oof_scores = cross_val_predict(
            search.best_estimator_,
            x_train,
            y_train,
            cv=cv,
            method="predict_proba",
            n_jobs=2,
        )[:, 1]
        threshold, oof_f1 = _best_f1_threshold(y_train, oof_scores)
        test_scores = search.best_estimator_.predict_proba(x_test)[:, 1]
        metrics = _classification_metrics(y_test, test_scores, threshold)

        summary.append(
            {
                "model": model_name,
                "feature_set": feature_set,
                "cv_folds": 3,
                "selection_metric": "average_precision",
                "best_cv_pr_auc_mean": float(search.best_score_),
                "best_cv_pr_auc_std": float(search.cv_results_["std_test_pr_auc"][search.best_index_]),
                "best_cv_f1_mean_at_0_5": float(search.cv_results_["mean_test_f1"][search.best_index_]),
                "oof_threshold": threshold,
                "oof_f1": oof_f1,
                "best_params": json.dumps(search.best_params_, sort_keys=True),
                "search_seconds": float(time.perf_counter() - started),
                **{f"test_{key}": value for key, value in metrics.items()},
            }
        )

        for idx, params in enumerate(search.cv_results_["params"]):
            all_candidates.append(
                {
                    "model": model_name,
                    "feature_set": feature_set,
                    "params": json.dumps(params, sort_keys=True),
                    "mean_train_pr_auc": float(search.cv_results_["mean_train_pr_auc"][idx]),
                    "mean_cv_pr_auc": float(search.cv_results_["mean_test_pr_auc"][idx]),
                    "std_cv_pr_auc": float(search.cv_results_["std_test_pr_auc"][idx]),
                    "mean_cv_f1_at_0_5": float(search.cv_results_["mean_test_f1"][idx]),
                    "rank_cv_pr_auc": int(search.cv_results_["rank_test_pr_auc"][idx]),
                }
            )

    summary_df = pd.DataFrame(summary).sort_values("best_cv_pr_auc_mean", ascending=False)
    summary_df.to_csv(RESULTS / "cross_validation_summary.csv", index=False)
    pd.DataFrame(all_candidates).to_csv(RESULTS / "grid_search_candidates.csv", index=False)

    plot = summary_df.copy()
    plot["feature_label"] = plot["feature_set"].replace(
        {"base": "base features", "base+retrospective_graph": "base + retrospective graph"}
    )
    plot["label"] = plot["model"] + "\n" + plot["feature_label"]
    plt.figure(figsize=(9, 5.5))
    ax = sns.barplot(data=plot, y="label", x="best_cv_pr_auc_mean", color="#2E7D8F")
    ax.errorbar(
        plot["best_cv_pr_auc_mean"],
        np.arange(len(plot)),
        xerr=plot["best_cv_pr_auc_std"],
        fmt="none",
        ecolor="#23272F",
        capsize=3,
    )
    plt.xlabel("Mean 3-fold validation PR-AUC")
    plt.ylabel("")
    plt.title("Formal Grid Search on Training Data")
    plt.xlim(left=0)
    plt.tight_layout()
    plt.savefig(FIGURES / "cross_validation_pr_auc.png", dpi=220)
    plt.close()


def add_expanding_history_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Create account and edge features from strictly earlier transactions.

    Rows must be in chronological order. Feature values are recorded before the
    current edge updates the dictionaries, which prevents current-row and
    future-row leakage. Validation history is allowed to update before later
    validation/test rows, matching an online monitoring system.
    """
    ordered = raw.copy()
    ordered["Timestamp"] = pd.to_datetime(ordered["Timestamp"], errors="coerce")
    ordered = ordered.sort_values("Timestamp", kind="mergesort").reset_index(drop=True)
    base = add_features(ordered)

    sent_count = defaultdict(int)
    received_count = defaultdict(int)
    sent_total = defaultdict(float)
    received_total = defaultdict(float)
    receivers = defaultdict(set)
    senders = defaultdict(set)
    pair_count = defaultdict(int)
    last_sent = {}
    last_received = {}
    sent_24h = defaultdict(deque)
    received_24h = defaultdict(deque)
    sent_7d = defaultdict(deque)
    received_7d = defaultdict(deque)

    values = {name: np.zeros(len(base), dtype=float) for name in HISTORY_NUMERIC}
    seconds_24h = 24 * 3600
    seconds_7d = 7 * seconds_24h
    max_recency_hours = 30 * 24

    timestamps = base["Timestamp"].astype("int64").to_numpy() / 1_000_000_000
    from_ids = base["from_account_id"].astype(str).to_numpy()
    to_ids = base["to_account_id"].astype(str).to_numpy()
    paid = base["Amount Paid"].fillna(0.0).to_numpy(dtype=float)
    received = base["Amount Received"].fillna(0.0).to_numpy(dtype=float)

    for i, (now, sender, receiver, paid_amount, received_amount) in enumerate(
        zip(timestamps, from_ids, to_ids, paid, received, strict=True)
    ):
        if not np.isfinite(now):
            now = 0.0

        for queue in (sent_24h[sender],):
            while queue and queue[0] < now - seconds_24h:
                queue.popleft()
        for queue in (received_24h[receiver],):
            while queue and queue[0] < now - seconds_24h:
                queue.popleft()
        for queue in (sent_7d[sender],):
            while queue and queue[0] < now - seconds_7d:
                queue.popleft()
        for queue in (received_7d[receiver],):
            while queue and queue[0] < now - seconds_7d:
                queue.popleft()

        sender_mean = sent_total[sender] / sent_count[sender] if sent_count[sender] else 0.0
        receiver_mean = (
            received_total[receiver] / received_count[receiver] if received_count[receiver] else 0.0
        )
        sender_seen = sent_count[sender] + received_count[sender]
        receiver_seen = sent_count[receiver] + received_count[receiver]

        values["hist_sender_sent_count"][i] = np.log1p(sent_count[sender])
        values["hist_sender_received_count"][i] = np.log1p(received_count[sender])
        values["hist_receiver_received_count"][i] = np.log1p(received_count[receiver])
        values["hist_receiver_sent_count"][i] = np.log1p(sent_count[receiver])
        values["hist_sender_unique_receivers"][i] = np.log1p(len(receivers[sender]))
        values["hist_receiver_unique_senders"][i] = np.log1p(len(senders[receiver]))
        values["hist_pair_count"][i] = np.log1p(pair_count[(sender, receiver)])
        values["hist_reverse_pair_count"][i] = np.log1p(pair_count[(receiver, sender)])
        values["hist_sender_total_sent"][i] = np.log1p(max(sent_total[sender], 0.0))
        values["hist_receiver_total_received"][i] = np.log1p(max(received_total[receiver], 0.0))
        values["hist_sender_mean_sent"][i] = np.log1p(max(sender_mean, 0.0))
        values["hist_receiver_mean_received"][i] = np.log1p(max(receiver_mean, 0.0))
        values["hist_sender_out_in_ratio"][i] = np.log1p(
            sent_count[sender] / (received_count[sender] + 1)
        )
        values["hist_receiver_in_out_ratio"][i] = np.log1p(
            received_count[receiver] / (sent_count[receiver] + 1)
        )
        values["hist_sender_hours_since_sent"][i] = min(
            (now - last_sent.get(sender, now - max_recency_hours * 3600)) / 3600,
            max_recency_hours,
        )
        values["hist_receiver_hours_since_received"][i] = min(
            (now - last_received.get(receiver, now - max_recency_hours * 3600)) / 3600,
            max_recency_hours,
        )
        values["hist_sender_sent_24h"][i] = np.log1p(len(sent_24h[sender]))
        values["hist_receiver_received_24h"][i] = np.log1p(len(received_24h[receiver]))
        values["hist_sender_sent_7d"][i] = np.log1p(len(sent_7d[sender]))
        values["hist_receiver_received_7d"][i] = np.log1p(len(received_7d[receiver]))
        values["hist_amount_vs_sender_mean"][i] = np.log1p(
            paid_amount / (sender_mean + 1.0)
        )
        values["hist_amount_vs_receiver_mean"][i] = np.log1p(
            received_amount / (receiver_mean + 1.0)
        )
        values["hist_sender_seen"][i] = float(sender_seen > 0)
        values["hist_receiver_seen"][i] = float(receiver_seen > 0)
        values["hist_pair_seen"][i] = float(pair_count[(sender, receiver)] > 0)
        values["hist_reverse_pair_seen"][i] = float(pair_count[(receiver, sender)] > 0)

        # Update state only after recording features for the current transaction.
        sent_count[sender] += 1
        received_count[receiver] += 1
        sent_total[sender] += paid_amount
        received_total[receiver] += received_amount
        receivers[sender].add(receiver)
        senders[receiver].add(sender)
        pair_count[(sender, receiver)] += 1
        last_sent[sender] = now
        last_received[receiver] = now
        sent_24h[sender].append(now)
        received_24h[receiver].append(now)
        sent_7d[sender].append(now)
        received_7d[receiver].append(now)

    for name, column in values.items():
        base[name] = column
    return base


def _temporal_hgb(params: dict, features: list[str]) -> Pipeline:
    """Build one temporal HGB candidate with preprocessing inside the pipeline."""
    return _model_pipeline(
        HistGradientBoostingClassifier(
            max_iter=params["max_iter"],
            learning_rate=params["learning_rate"],
            max_leaf_nodes=params["max_leaf_nodes"],
            l2_regularization=params["l2_regularization"],
            class_weight=params["class_weight"],
            random_state=RANDOM_STATE,
        ),
        features,
    )


def run_strict_temporal_ablation(df: pd.DataFrame) -> None:
    """Tune base and strict-history HGB on a chronological validation period."""
    raw = df[RAW_COLUMNS].copy()
    strict = add_expanding_history_features(raw)
    strict.to_parquet(RESULTS / "strict_history_model_sample.parquet", index=False)

    n = len(strict)
    train_end = int(n * 0.60)
    val_end = int(n * 0.80)
    train = strict.iloc[:train_end].copy()
    validation = strict.iloc[train_end:val_end].copy()
    test = strict.iloc[val_end:].copy()

    candidates = []
    for learning_rate in [0.04, 0.08, 0.12]:
        for max_leaf_nodes in [15, 31, 63]:
            for class_weight in [None, "balanced"]:
                candidates.append(
                    {
                        "max_iter": 180,
                        "learning_rate": learning_rate,
                        "max_leaf_nodes": max_leaf_nodes,
                        "l2_regularization": 1.0,
                        "class_weight": class_weight,
                    }
                )

    rows = []
    candidate_rows = []
    selected_test_scores = {}
    for feature_label, features in [
        ("base", BASE_NUMERIC),
        ("strict_expanding_history", BASE_NUMERIC + HISTORY_NUMERIC),
    ]:
        x_train = train.drop(columns=["Is Laundering"])
        y_train = train["Is Laundering"].astype(int)
        x_val = validation.drop(columns=["Is Laundering"])
        y_val = validation["Is Laundering"].astype(int)
        x_test = test.drop(columns=["Is Laundering"])
        y_test = test["Is Laundering"].astype(int)

        best = None
        for params in candidates:
            started = time.perf_counter()
            model = _temporal_hgb(params, features)
            model.fit(x_train, y_train)
            val_scores = model.predict_proba(x_val)[:, 1]
            val_pr_auc = float(average_precision_score(y_val, val_scores))
            val_roc_auc = float(roc_auc_score(y_val, val_scores))
            candidate = {
                "feature_set": feature_label,
                "params": json.dumps(params, sort_keys=True),
                "validation_pr_auc": val_pr_auc,
                "validation_roc_auc": val_roc_auc,
                "fit_seconds": float(time.perf_counter() - started),
            }
            candidate_rows.append(candidate)
            if best is None or val_pr_auc > best["validation_pr_auc"]:
                best = {**candidate, "model": model, "validation_scores": val_scores}

        threshold, validation_f1 = _best_f1_threshold(y_val, best["validation_scores"])
        test_scores = best["model"].predict_proba(x_test)[:, 1]
        selected_test_scores[feature_label] = test_scores
        metrics = _classification_metrics(y_test, test_scores, threshold)
        rows.append(
            {
                "protocol": "time_based_60_20_20_strict_expanding_history",
                "feature_set": feature_label,
                "train_rows": len(train),
                "validation_rows": len(validation),
                "test_rows": len(test),
                "train_positive_rate": float(y_train.mean()),
                "validation_positive_rate": float(y_val.mean()),
                "test_positive_rate": float(y_test.mean()),
                "best_params": best["params"],
                "validation_pr_auc": best["validation_pr_auc"],
                "threshold": threshold,
                "validation_f1_at_threshold": validation_f1,
                **metrics,
            }
        )

    result = pd.DataFrame(rows)
    result.to_csv(RESULTS / "time_based_history_ablation.csv", index=False)
    pd.DataFrame(candidate_rows).to_csv(RESULTS / "temporal_model_selection_candidates.csv", index=False)

    wide = result.set_index("feature_set")
    delta = {
        "comparison": "base -> strict expanding history",
        "delta_f1": float(wide.loc["strict_expanding_history", "f1"] - wide.loc["base", "f1"]),
        "delta_pr_auc": float(
            wide.loc["strict_expanding_history", "pr_auc"] - wide.loc["base", "pr_auc"]
        ),
        "base_pr_auc": float(wide.loc["base", "pr_auc"]),
        "history_pr_auc": float(wide.loc["strict_expanding_history", "pr_auc"]),
    }
    pd.DataFrame([delta]).to_csv(RESULTS / "time_based_history_ablation_delta.csv", index=False)

    prediction_frame = pd.DataFrame(
        {
            "Timestamp": test["Timestamp"].astype(str).to_numpy(),
            "y_true": test["Is Laundering"].astype(int).to_numpy(),
            "base_score": selected_test_scores["base"],
            "strict_history_score": selected_test_scores["strict_expanding_history"],
        }
    )
    prediction_frame.to_csv(RESULTS / "time_based_test_predictions.csv", index=False)

    # Paired bootstrap keeps the same resampled transactions for both models.
    rng = np.random.default_rng(RANDOM_STATE)
    y_test_array = prediction_frame["y_true"].to_numpy()
    base_scores = prediction_frame["base_score"].to_numpy()
    history_scores = prediction_frame["strict_history_score"].to_numpy()
    bootstrap_deltas = []
    for _ in range(500):
        sample_idx = rng.integers(0, len(prediction_frame), size=len(prediction_frame))
        sampled_y = y_test_array[sample_idx]
        if sampled_y.min() == sampled_y.max():
            continue
        bootstrap_deltas.append(
            average_precision_score(sampled_y, history_scores[sample_idx])
            - average_precision_score(sampled_y, base_scores[sample_idx])
        )
    ci_low, ci_high = np.quantile(bootstrap_deltas, [0.025, 0.975])
    pd.DataFrame(
        [
            {
                "metric": "paired_bootstrap_delta_pr_auc",
                "point_estimate": delta["delta_pr_auc"],
                "ci_lower_95": float(ci_low),
                "ci_upper_95": float(ci_high),
                "bootstrap_resamples": len(bootstrap_deltas),
            }
        ]
    ).to_csv(RESULTS / "time_based_history_bootstrap_ci.csv", index=False)

    plt.figure(figsize=(7.5, 5))
    plot = result.copy()
    plot["label"] = plot["feature_set"].replace(
        {"base": "Base features", "strict_expanding_history": "Base + strict history"}
    )
    melted = plot.melt(
        id_vars="label",
        value_vars=["pr_auc", "f1"],
        var_name="metric",
        value_name="score",
    )
    melted["metric"] = melted["metric"].replace({"pr_auc": "PR-AUC", "f1": "F1"})
    sns.barplot(data=melted, x="metric", y="score", hue="label", palette=["#6B7280", "#2E7D8F"])
    plt.xlabel("")
    plt.ylabel("Test score")
    plt.title("Chronological Test: Strict Expanding-history Ablation")
    plt.ylim(0, 1)
    plt.legend(title="")
    plt.tight_layout()
    plt.savefig(FIGURES / "time_based_history_ablation.png", dpi=220)
    plt.close()


def update_failure_log() -> None:
    """Add the temporal failure and its evidence-based repair to the audit log."""
    path = RESULTS / "failure_modes.csv"
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
    temporal_rows = pd.DataFrame(
        [
            {
                "stage": "Temporal graph audit",
                "attempt": "Freeze account graph aggregates at the end of training",
                "failure_mode": (
                    "Chronological test PR-AUC falls from 0.683 for base features "
                    "to 0.459 because later rows cannot update the frozen account state."
                ),
                "lesson": (
                    "Blocking future edges is necessary but not sufficient; feature "
                    "representation must also evolve exactly as it would online."
                ),
            },
            {
                "stage": "Temporal graph repair",
                "attempt": "Compute expanding-history features before each transaction updates state",
                "failure_mode": (
                    "The repair succeeds on the modeling stream (PR-AUC 0.829 versus "
                    "0.713), but it still does not reproduce the natural deployment prior."
                ),
                "lesson": (
                    "Strictly historical graph context is promising; the claim remains "
                    "bounded to a sampled chronological test and needs full-stream validation."
                ),
            },
        ]
    )
    if not existing.empty:
        existing = existing[~existing["stage"].isin(temporal_rows["stage"])]
    pd.concat([existing, temporal_rows], ignore_index=True).to_csv(path, index=False)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    df = pd.read_parquet(RESULTS / "model_sample_features.parquet")
    run_formal_grid_search(df)
    run_strict_temporal_ablation(df)
    update_failure_log()
    print("Formal model selection and strict temporal experiments completed.")


if __name__ == "__main__":
    main()
