# GraphGuard – Graph-Based Financial Fraud Detection

## About the Project

I built GraphGuard as a machine learning project to explore how suspicious financial transactions can be identified by looking at both transaction details and the connections between bank accounts.

In a normal fraud detection system, we mainly look at details such as transaction amount and time. But sometimes, looking at a single transaction is not enough. The way money moves between different accounts can also reveal suspicious activity.

That's where graph analysis comes in.

In this project, I represent bank accounts as nodes and money transfers as directed edges. I then extract useful information from this network and use traditional machine learning algorithms to classify transactions as normal or potentially fraudulent.

## What I wanted to achieve

- Build a transaction network using Python and NetworkX.
- Understand how graph features can help in fraud detection.
- Train and compare different machine learning algorithms.
- Identify accounts involved in suspicious transactions.
- Create a dashboard where predictions and account connections can be explored.

## Tools and Libraries

| Technology | How I used it |
|---|---|
| Python | Main programming language |
| Pandas | Reading and processing transaction data |
| NumPy | Numerical operations |
| Scikit-learn | Training and evaluating ML models |
| NetworkX | Creating and analysing the transaction graph |
| Matplotlib | Creating evaluation plots |
| Seaborn | Confusion matrix visualization |
| Plotly | Interactive network visualization |
| Streamlit | Building the web dashboard |
| Joblib | Saving and loading the trained model |

## How the Project Works

The project follows these steps:

1. Generate a banking transaction dataset.
2. Represent accounts and transactions as a directed graph.
3. Extract graph-related features from the network.
4. Combine those features with transaction details.
5. Train multiple machine learning models.
6. Compare their performance.
7. Display predictions and account connections through a Streamlit dashboard.

## Graph Construction and Features

For the graph representation:

- Each bank account is treated as a node.
- Each money transfer is treated as a directed edge.
- The direction of an edge represents the movement of money from sender to receiver.

I extracted the following features:

| Feature | What it tells us |
|---|---|
| Sender Out-Degree | Number of different accounts receiving money from a sender |
| Sender In-Degree | Number of different accounts sending money to a sender |
| Receiver Out-Degree | Number of outgoing account connections |
| Receiver In-Degree | Number of incoming account connections |
| Weighted Degree | Number of transfers, including repeated transfers |
| PageRank | Importance of an account in the transaction network |
| Common Neighbours | Number of shared neighbouring accounts |
| Reciprocal Transfer | Whether a reverse transfer exists between two accounts |

Along with these graph features, I also used transaction amount, hour and day information.

## Machine Learning Models

I trained and compared three traditional machine learning algorithms:

1. Logistic Regression
2. Decision Tree
3. Random Forest

I used a train-test split and evaluated the models using accuracy, precision, recall, F1-score, PR-AUC and ROC-AUC.

I selected the primary model based on PR-AUC.

## Dataset

Since this is an academic prototype, I generated a synthetic banking transaction dataset using Python.

The dataset contains 1,000 accounts and 10,200 transactions.

I included different transaction patterns to simulate suspicious activity:

- Circular transactions – money moving between accounts in a cycle.
- Fan-in transactions – multiple accounts sending money to one account.
- Fan-out transactions – one account sending money to multiple accounts.
- High-value transfers – additional suspicious transfers.

The generated dataset contains:

| Transaction Type | Count |
|---|---:|
| Normal transactions | 9,500 |
| Fraud-labelled transactions | 700 |
| Total transactions | 10,200 |

These labels are based on the patterns injected during dataset generation. They are not confirmed cases of real financial fraud.

## Model Results

The following table shows the results from my current experiment:

| Model | Accuracy | Precision | Recall | F1-score | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 98.38% | 81.29% | 99.29% | 89.39% | 97.72% |
| Random Forest | 99.17% | 91.84% | 96.43% | 94.29% | 94.68% |
| Decision Tree | 98.53% | 85.71% | 94.29% | 89.80% | 90.80% |

Logistic Regression achieved the highest PR-AUC, so it was selected as the primary model.

Random Forest achieved the highest F1-score, which makes it an interesting alternative when balancing precision and recall.

## Dashboard

I used Streamlit to build a simple interactive dashboard with four sections.

### 1. Overview
Displays the total number of transactions, normal transactions, fraud-labelled transactions and accounts.

### 2. Fraud Detection
Allows the user to select a transaction and view its predicted fraud classification and estimated fraud score