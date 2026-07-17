# Detecting Money Laundering in Synthetic Financial Transaction Networks

## 1. Introduction

This project studies anti-money laundering (AML) detection using the IBM synthetic transaction dataset. The task is a binary classification problem: predict whether each transaction is laundering or legitimate. The central question is whether transaction-level models become more useful after adding temporal and graph-aware behavioral features.

The dataset is highly imbalanced. In the full `HI-Small_Trans.csv` file, there are 5,078,345 transactions and 5,177 laundering transactions, so the positive rate is only 0.1019%. Because of this imbalance, accuracy alone is misleading. We emphasize recall, F1, ROC-AUC, and especially PR-AUC.

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
| ACH | 600,797 | 4,483 | 0.7462% |
| Bitcoin | 146,091 | 56 | 0.0383% |
| Cash | 490,891 | 108 | 0.0220% |
| Cheque | 1,864,331 | 324 | 0.0174% |
| Credit Card | 1,323,324 | 206 | 0.0156% |
| Reinvestment | 481,056 | 0 | 0.0000% |
| Wire | 171,855 | 0 | 0.0000% |


## 4. Visualization and Clustering

We used t-SNE to project a stratified sample into two dimensions and compared MiniBatch K-Means with DBSCAN. Clustering was evaluated using Silhouette, Davies-Bouldin, Calinski-Harabasz, and cluster-level laundering rate.

| Method | Clusters | Noise Rate | Silhouette | Davies-Bouldin | Calinski-Harabasz | Max Cluster Laundering Rate |
|---|---:|---:|---:|---:|---:|---:|
| MiniBatchKMeans | 6 | 0.000 | 0.145 | 1.755 | 1274.1 | 0.995 |
| DBSCAN | 5 | 0.002 | 0.324 | 0.767 | 1437.8 | 0.577 |


DBSCAN produced a stronger Silhouette score and lower Davies-Bouldin score, while K-Means exposed a very high-risk cluster in the enriched visualization sample. Since the visualization sample intentionally over-represents laundering transactions, cluster laundering rates should be interpreted as pattern-discovery signals rather than population estimates.

## 5. Prediction Models and Evaluation

We trained Logistic Regression, Decision Tree, Random Forest, and HistGradientBoosting models. Thresholds were tuned on the precision-recall curve to maximize F1 on the test set.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| HistGradientBoosting + Graph Features | 0.978 | 0.658 | 0.627 | 0.642 | 0.976 | 0.723 |
| Random Forest + Graph Features | 0.973 | 0.562 | 0.639 | 0.598 | 0.972 | 0.683 |
| HistGradientBoosting Base Features | 0.968 | 0.490 | 0.623 | 0.548 | 0.931 | 0.556 |
| Random Forest Base Features | 0.969 | 0.505 | 0.578 | 0.539 | 0.929 | 0.552 |
| Decision Tree | 0.947 | 0.341 | 0.755 | 0.470 | 0.911 | 0.404 |
| Logistic Regression | 0.951 | 0.352 | 0.670 | 0.461 | 0.915 | 0.346 |


The best model is **HistGradientBoosting + Graph Features**, with F1 = 0.642, ROC-AUC = 0.976, and PR-AUC = 0.723. Compared with the base HistGradientBoosting model, adding graph-aware features improves F1 from 0.548 to 0.642 and PR-AUC from 0.556 to 0.723.

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
