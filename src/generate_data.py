
import pandas as pd
import numpy as np
import random
import os

# Reproducibility
np.random.seed(42)
random.seed(42)

# Dataset configuration
NUM_ACCOUNTS = 1000
NUM_NORMAL = 9500
NUM_FRAUD = 500

accounts = [f"ACC_{i:04d}" for i in range(NUM_ACCOUNTS)]

transactions = []

start_date = pd.Timestamp("2025-01-01")


def create_transaction(sender, receiver, amount, is_fraud):
    timestamp = start_date + pd.Timedelta(
        seconds=random.randint(0, 30 * 24 * 60 * 60)
    )

    transactions.append({
        "sender": sender,
        "receiver": receiver,
        "amount": round(amount, 2),
        "timestamp": timestamp,
        "is_fraud": is_fraud
    })


# 1. Generate normal transactions
for _ in range(NUM_NORMAL):

    sender, receiver = random.sample(accounts, 2)

    amount = np.random.lognormal(mean=6, sigma=1)

    create_transaction(
        sender, receiver, amount, 0
    )


# 2. Generate suspicious circular transactions
# A -> B -> C -> A

for _ in range(50):

    group = random.sample(accounts, 3)

    for i in range(3):
        sender = group[i]
        receiver = group[(i + 1) % 3]

        create_transaction(
            sender, receiver,
            random.uniform(5000, 25000), 1
        )


# 3. Generate fan-in transactions
# Multiple accounts send money to one account

for _ in range(50):

    group = random.sample(accounts, 6)
    central_account = group[0]

    for sender in group[1:]:

        create_transaction(
            sender, central_account,
            random.uniform(3000, 15000), 1
        )


# 4. Generate fan-out transactions
# One account sends money to multiple accounts

for _ in range(50):

    group = random.sample(accounts, 6)
    central_account = group[0]

    for receiver in group[1:]:

        create_transaction(
            central_account, receiver,
            random.uniform(3000, 15000), 1
        )


# 5. Generate additional suspicious transfers
for _ in range(50):

    sender, receiver = random.sample(accounts, 2)

    create_transaction(
        sender, receiver,
        random.uniform(10000, 40000), 1
    )


# Convert to DataFrame
df = pd.DataFrame(transactions)

# Shuffle transactions
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Add transaction IDs
df.insert(
    0,
    "transaction_id",
    [f"TXN_{i:06d}" for i in range(len(df))]
)

# Create output directory
os.makedirs("data/raw", exist_ok=True)

# Save dataset
df.to_csv("data/raw/transactions.csv", index=False)

print("Dataset generated successfully!")
print("--------------------------------")
print("Total transactions:", len(df))
print("Total accounts:", NUM_ACCOUNTS)
print("Fraud transactions:", df["is_fraud"].sum())
print("Normal transactions:", (df["is_fraud"] == 0).sum())

print("\nDataset preview:")
print(df.head())

print("\nFraud distribution:")
print(df["is_fraud"].value_counts())
