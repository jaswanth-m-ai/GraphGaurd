
import pandas as pd
import networkx as nx
import numpy as np
import os

print("Loading transaction dataset...")

df = pd.read_csv("data/raw/transactions.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Build directed transaction graph
# Accounts = nodes, transactions = directed edges
G = nx.DiGraph()

for row in df.itertuples(index=False):
    sender = row.sender
    receiver = row.receiver

    if G.has_edge(sender, receiver):
        G[sender][receiver]["weight"] += 1
    else:
        G.add_edge(sender, receiver, weight=1)

print("Transaction graph created!")
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

# Convert to undirected graph for common-neighbour analysis
UG = G.to_undirected()

print("Calculating PageRank...")
pagerank = nx.pagerank(G, weight="weight")

# Extract graph features for every transaction
features = []

for row in df.itertuples(index=False):

    sender = row.sender
    receiver = row.receiver

    # Account connectivity
    sender_out = G.out_degree(sender)
    sender_in = G.in_degree(sender)

    receiver_out = G.out_degree(receiver)
    receiver_in = G.in_degree(receiver)

    # Weighted transaction connectivity
    sender_out_weight = G.out_degree(sender, weight="weight")
    receiver_in_weight = G.in_degree(receiver, weight="weight")

    # PageRank scores
    sender_pr = pagerank.get(sender, 0)
    receiver_pr = pagerank.get(receiver, 0)

    # Shared neighbours
    common = len(
        list(nx.common_neighbors(UG, sender, receiver))
    )

    # Check whether money also flows in the reverse direction
    reciprocal = int(G.has_edge(receiver, sender))

    features.append({
        "transaction_id": row.transaction_id,
        "amount": row.amount,
        "hour": pd.Timestamp(row.timestamp).hour,
        "day": pd.Timestamp(row.timestamp).dayofweek,
        "sender_out_degree": sender_out,
        "sender_in_degree": sender_in,
        "receiver_out_degree": receiver_out,
        "receiver_in_degree": receiver_in,
        "sender_out_weight": sender_out_weight,
        "receiver_in_weight": receiver_in_weight,
        "sender_pagerank": sender_pr,
        "receiver_pagerank": receiver_pr,
        "common_neighbors": common,
        "reciprocal_transfer": reciprocal,
        "is_fraud": row.is_fraud
    })

# Create feature dataset
feature_df = pd.DataFrame(features)

# Save processed data
os.makedirs("data/processed", exist_ok=True)

feature_df.to_csv(
    "data/processed/graph_features.csv",
    index=False
)

print("\nGraph feature extraction completed!")
print("------------------------------------")
print("Total transactions:", len(feature_df))
print("Total graph features:", len(feature_df.columns) - 2)
print("Fraud transactions:", feature_df["is_fraud"].sum())

print("\nFeature preview:")
print(feature_df.head())

print("\nSaved to data/processed/graph_features.csv")
