# Detecting Money Laundering in Synthetic Financial Transaction Networks

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
- Full data: 5,078,345 transactions.
- Laundering transactions: 5,177.
- Positive rate: 0.1019%.

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

Best model: HistGradientBoosting + Graph Features

- F1: 0.642
- ROC-AUC: 0.976
- PR-AUC: 0.723

## 9. Graph Feature Impact

- HGB base F1: 0.548; HGB + graph F1: 0.642.
- HGB base PR-AUC: 0.556; HGB + graph PR-AUC: 0.723.
- Important features include payment format, log amount, sender out-degree, and receiver in-degree.

## 10. Conclusion

- Class imbalance makes accuracy insufficient.
- Graph-aware models provide stronger minority-class detection.
- Future work: time-based validation and temporal GNNs.
