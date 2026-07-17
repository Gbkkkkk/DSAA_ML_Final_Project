# DSAA2011 Final Project Plan: IBM AML Transaction Detection

## Project Title

**Detecting Money Laundering in Synthetic Financial Transaction Networks: From Tabular ML to Graph-aware AML Features**

## One-sentence Storyline

我们不只是预测 `Is Laundering`，而是把 IBM AML 数据集看成一个金融交易网络，比较“普通表格特征模型”和“图结构增强模型”在极度类别不平衡 AML 任务上的表现。

## Core Research Question

> Can transaction-level machine learning models detect money laundering more effectively after incorporating temporal, currency, and graph-based behavioral features?

## Dataset

Kaggle dataset:

https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml

Recommended files:

- `HI-Small_Trans.csv`: 主交易数据
- `HI-Small_Patterns.txt`: 洗钱模式解释，可用于 open-ended exploration

Main target:

- `Is Laundering`: binary classification target, where `1` means laundering transaction and `0` means legitimate transaction.

Expected columns:

- `Timestamp`
- `From Bank`
- `Account`
- `To Bank`
- `Account.1`
- `Amount Received`
- `Receiving Currency`
- `Amount Paid`
- `Payment Currency`
- `Payment Format`
- `Is Laundering`

## Why This Project Has A+ Potential

这个 IBM 数据集很适合做高质量 project。IBM 官方说明它是合成金融交易数据，包含银行转账、支付、信用卡、支票等交易，并提供 laundering 标签。数据由多智能体虚拟世界生成，不是匿名化真实个人数据。多数交易合法，少数交易是洗钱。

NeurIPS 2023 的数据集论文强调，真实 AML 数据通常难以获得且标签不完整，而这个合成数据的优势是 ground truth 标签完整，可用于模型比较。

本项目的亮点可以放在：

- Class imbalance
- Transaction network
- Graph-aware feature engineering
- Model interpretability
- False positive / false negative trade-off
- AML business interpretation

Recent AML research also supports this direction:

- Network analytics can improve AML predictive power, but results must be interpreted carefully under class imbalance and synthetic data settings.
- Graph/subgraph features combined with gradient boosting can improve minority-class F1.
- Temporal graph learning can help reduce false positives in AML detection.

## Overall Model Design

The final project should compare several levels of models:

1. **Baseline models**
   - Logistic Regression
   - Decision Tree

2. **Stronger classical models**
   - Random Forest
   - XGBoost or LightGBM

3. **A+ open-ended extension**
   - Graph-aware behavioral features
   - Account-level network statistics
   - Feature ablation study

4. **Optional bonus**
   - Small-scale GNN or node/edge embedding experiment, if time and computing resources allow

## Three-person Division of Work

## Part 1: Data Preprocessing, EDA, and Feature Engineering

**Owner:** Student 1

This part builds the foundation of the whole project.

### Tasks

- Load data and inspect schema.
- Check missing values.
- Check duplicates.
- Convert timestamp to datetime.
- Handle non-numeric categorical values.
- Standardize or normalize numerical features.
- Create account identifiers:
  - `from_account_id = From Bank + Account`
  - `to_account_id = To Bank + Account.1`
- Analyze class imbalance:
  - laundering count
  - non-laundering count
  - positive class ratio
  - baseline accuracy if predicting all transactions as non-laundering

### Feature Engineering

Temporal features:

- Hour
- Day
- Weekday
- Weekend indicator
- Time order

Amount features:

- `Amount Paid`
- `Amount Received`
- `log(Amount Paid)`
- `log(Amount Received)`
- Amount difference
- Same-currency indicator

Transaction features:

- Same-bank indicator
- Cross-bank indicator
- Same-account indicator
- Payment format encoding
- Currency encoding

### Required Visualizations

- Amount distribution for laundering vs non-laundering transactions
- Laundering rate by payment format
- Laundering rate by currency
- Transaction volume over time
- Suspicious rate over time
- Top sender/receiver banks or accounts

### A+ Focus

Do not only say "the data is imbalanced." Quantify the imbalance and explain why accuracy alone is misleading.

AML interpretation:

- False negative: laundering transaction missed, high regulatory risk.
- False positive: legitimate transaction flagged, higher manual review cost.

### Deliverables

- Notebook section: `1_data_preprocessing_eda`
- 3-5 polished figures
- Report sections:
  - Dataset
  - Preprocessing
  - Exploratory Data Analysis

## Part 2: Visualization, Clustering, and Pattern Discovery

**Owner:** Student 2

This part covers t-SNE, clustering analysis, and unsupervised pattern discovery.

### Input Features

Use the standardized and encoded features from Part 1.

For t-SNE and clustering, do not run directly on the full dataset if it is too large. Use stratified sampling:

- Include all laundering transactions if possible.
- Randomly sample non-laundering transactions.
- Recommended sample size: 50k-100k, depending on computer performance.

### t-SNE

Tasks:

- Apply t-SNE to reduce high-dimensional features to 2D.
- Color points by `Is Laundering`.
- Analyze whether laundering transactions form visible local patterns or overlap heavily with legitimate transactions.

### Clustering Algorithms

Use at least two clustering algorithms:

1. K-Means
   - Course-friendly
   - Easy to explain and evaluate

2. DBSCAN or HDBSCAN
   - Better for detecting sparse or anomalous clusters
   - More suitable for suspicious transaction discovery

Optional:

- Hierarchical clustering on a smaller sample

### Clustering Evaluation Metrics

Use multiple metrics:

- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Score
- Cluster-level laundering rate

### A+ Focus

Do not only report clustering scores. Answer these questions:

- Which clusters have unusually high laundering rates?
- What features characterize high-risk clusters?
- Are high-risk clusters associated with specific payment formats, currencies, cross-bank transfers, or time periods?
- Can cluster labels be used as additional features in supervised models?

### Deliverables

- t-SNE projection plot
- K-Means clustering plot
- DBSCAN/HDBSCAN clustering plot
- Cluster profiling table
- Report sections:
  - Data Visualization
  - Clustering Analysis

## Part 3: Prediction, Evaluation, and Model Selection

**Owner:** Student 3

This part is the supervised learning core of the project.

### Classification Target

Target variable:

```text
Is Laundering
```

Task:

```text
Predict whether a transaction is laundering or legitimate.
```

### Models

Required simple models:

- Logistic Regression
- Decision Tree

Stronger models for comparison:

- Random Forest
- XGBoost or LightGBM

Optional improvements:

- Class-weighted models
- Random undersampling
- SMOTE, if used carefully
- Threshold tuning

### Train-test Split

Use two evaluation settings:

1. Course-required split:
   - 70% training
   - 30% testing

2. A+ extension:
   - Time-based split
   - Earlier transactions for training
   - Later transactions for testing

This makes the evaluation closer to a real AML monitoring setting.

### Evaluation Metrics

Accuracy should be reported but not emphasized.

Main metrics:

- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion matrix

Why PR-AUC matters:

- The dataset is highly imbalanced.
- PR-AUC is more informative than ROC-AUC for rare positive classes.

### Required Plots

- Confusion matrix for each model
- ROC curve
- Precision-recall curve
- Feature importance plot
- Optional SHAP summary plot

### A+ Focus

Run an ablation study:

1. Original tabular features only
2. Original features + temporal features
3. Original features + temporal + account behavior features
4. Original features + temporal + account behavior + graph features

The strongest final conclusion would be:

> Adding temporal and graph-aware behavioral features improves minority-class detection compared with simple tabular baselines.

### Deliverables

- Model comparison table
- Confusion matrices
- ROC curves
- PR curves
- Feature importance or SHAP analysis
- Report sections:
  - Prediction
  - Evaluation
  - Model Choice

## Recommended Graph-aware Features

This is the key open-ended exploration that can make the project stand out.

Construct a transaction graph:

- Node: account
- Edge: transaction from source account to target account
- Edge attributes:
  - amount
  - currency
  - payment format
  - timestamp
  - laundering label

Recommended graph features:

- Sender out-degree
- Receiver in-degree
- Sender total sent amount
- Receiver total received amount
- Sender unique receivers
- Receiver unique senders
- Sent/received amount ratio
- Same-bank transaction flag
- Cross-bank transaction flag
- Short time-window transaction count
- PageRank
- Degree centrality
- Clustering coefficient, if computationally feasible

These features connect the project to recent AML research, where network analytics, subgraph patterns, and temporal graph behavior are central topics.

## Final Report Structure

Recommended length: 5-10 pages, excluding references.

### 1. Introduction

- AML background
- Why synthetic AML data is useful
- Research question
- Main findings preview

### 2. Dataset and Preprocessing

- Dataset source
- Field explanation
- Missing values
- Encoding
- Standardization
- Class imbalance
- Feature engineering

### 3. Visualization and Clustering

- EDA plots
- t-SNE projection
- K-Means result
- DBSCAN/HDBSCAN result
- Cluster laundering rate analysis

### 4. Prediction Models

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost/LightGBM
- Train/test split
- Class imbalance handling

### 5. Evaluation and Model Choice

- Confusion matrices
- Accuracy, precision, recall, F1
- ROC-AUC
- PR-AUC
- Threshold tuning
- Final model selection

### 6. Open-ended Exploration

- Graph-based features
- Ablation study
- Optional SHAP or feature importance
- Optional GNN or graph embedding experiment

### 7. Conclusion

- Main conclusion
- Limitations:
  - Synthetic data
  - Class imbalance
  - Computational constraints
- Future work:
  - Temporal GNN
  - Real-time streaming AML detection
  - More advanced subgraph pattern mining

### 8. References and Credit

- Cite Kaggle dataset
- Cite IBM GitHub
- Cite NeurIPS 2023 IBM AML paper
- Cite recent AML graph/network papers
- Disclose GenAI usage according to course policy

## 10-minute Presentation Structure

Recommended slide deck: 9-10 slides.

1. Title + team contribution
2. Problem: why AML detection is hard
3. Dataset: IBM synthetic AML data, labels, and imbalance
4. EDA: suspicious patterns by amount, time, currency, and payment format
5. t-SNE and clustering results
6. Model pipeline
7. Model comparison table
8. Graph feature ablation result
9. Final model choice and AML interpretation
10. Conclusion, limitations, and future work

## Recommended Project File Structure

```text
project_groupID_IBM_AML.ipynb
requirements_groupID_IBM_AML.txt
report_groupID_IBM_AML.pdf
presentation_groupID_IBM_AML.pdf
data/
  HI-Small_Trans.csv
  HI-Small_Patterns.txt
figures/
  eda_amount_distribution.png
  tsne_projection.png
  clustering_comparison.png
  confusion_matrices.png
  roc_pr_curves.png
  feature_importance.png
```

## Suggested Final Conclusion

The final report should aim to support this claim:

> In highly imbalanced AML detection, simple tabular models can achieve high accuracy but weak minority-class detection. Adding temporal and graph-aware behavioral features substantially improves recall, F1, and PR-AUC, making the model more useful for suspicious transaction screening.

This conclusion is strong because it combines:

- Course-required machine learning methods
- Realistic evaluation metrics
- Business interpretation
- Open-ended exploration
- Recent research direction

## References

1. IBM AML-Data GitHub repository: https://github.com/IBM/AML-Data
2. Kaggle IBM Transactions for Anti Money Laundering dataset: https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml
3. Altman et al., "Realistic Synthetic Financial Transactions for Anti-Money Laundering Models", NeurIPS 2023: https://proceedings.neurips.cc/paper_files/paper/2023/hash/5f38404edff6f3f642d6fa5892479c42-Abstract-Datasets_and_Benchmarks.html
4. Deprez et al., "Network Analytics for Anti-Money Laundering - A Systematic Literature Review and Experimental Evaluation": https://arxiv.org/abs/2405.19383
5. Blanusa et al., "Graph Feature Preprocessor: Real-time Subgraph-based Feature Extraction for Financial Crime Detection": https://arxiv.org/abs/2402.08593
6. Di Gennaro et al., "Amatriciana: Exploiting Temporal GNNs for Robust and Efficient Money Laundering Detection": https://arxiv.org/abs/2506.00654

