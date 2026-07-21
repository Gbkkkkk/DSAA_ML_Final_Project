# Part 2: Visualization and Clustering Analysis

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

The best internal clustering result is **DBSCAN (eps=3.5)**, with Silhouette = 0.530, Davies-Bouldin = 0.578, and Calinski-Harabasz = 1639.8. Internal metrics are not sufficient for AML, so we also inspect laundering enrichment inside clusters.

## High-risk Cluster Profiling

Top high-risk clusters in the enriched visualization sample:

| Method | Cluster | Transactions | Laundering Rate | Enrichment | Dominant Payment Format | Mean Amount Paid | Mean Sender Out-degree | Mean Receiver In-degree |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| MiniBatch K-Means (k=7) | 6 | 871 | 0.993 | 2.24x | ACH | 1520294.71 | 5.07 | 14.84 |
| Gaussian Mixture (k=7) | 5 | 1178 | 0.966 | 2.18x | ACH | 870238.83 | 6.32 | 12.50 |
| MiniBatch K-Means (k=7) | 3 | 1526 | 0.611 | 1.38x | ACH | 114537.77 | 17.13 | 1.99 |


## Interpretation

The high-risk clusters are characterized by combinations of payment format, transaction amount, and account-network behavior. This is important because it shows that laundering is not only a row-level anomaly; it also appears as a behavioral pattern in the transaction network. The clustering results therefore motivate the Part 3 modeling choice: nonlinear supervised models with graph-aware features should outperform simple linear baselines.

## Final Part 2 Conclusion

The unsupervised analysis shows that laundering transactions are not globally separable, but they concentrate in several local high-risk regions characterized by transaction amount, payment format, and account-network behavior. This supports the use of nonlinear and graph-aware supervised models for AML detection.
