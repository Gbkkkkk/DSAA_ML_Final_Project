Official dataset: IBM Transactions for Anti Money Laundering on Kaggle
https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml

The Kaggle release has six independent variants (HI/LI and Small/Medium/Large).
Each variant has two files. HI-Small contains:
  - HI-Small_Trans.csv
  - HI-Small_Patterns.txt

This project reads only HI-Small_Trans.csv. Account-level features are derived
from its Account and Account.1 columns; no separate account file is required.
The Patterns file is optional and is not used in the submitted experiments.

Accepted data layouts:
  1. data/HI-Small_Trans.csv
  2. data/HI-Small_Trans.csv.zip
  3. a Kaggle download ZIP in data/ that contains HI-Small_Trans.csv

If none is present, src/aml_project_pipeline.py downloads a Hugging Face mirror
that contains only HI-Small_Trans.csv.zip. Kaggle remains the cited source.
