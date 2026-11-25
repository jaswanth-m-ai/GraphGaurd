

import pandas as pd
import numpy as np
import os
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# 1. Load graph features
print("Loading graph features...")

df = pd.read_csv("data/processed/graph_features.csv")

# 2. Separate features and target
X = df.drop(columns=["transaction_id", "is_fraud"])
y = df["is_fraud"]

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Define traditional ML models
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ))
    ]),

    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced",
        max_depth=10,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )
}

results = []
trained_models = {}

# 5. Train and evaluate each model
for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "PR-AUC": average_precision_score(
            y_test, probabilities
        ),
        "ROC-AUC": roc_auc_score(
            y_test, probabilities
        )
    }

    results.append(metrics)
    trained_models[name] = model

    print(classification_report(
        y_test, predictions, zero_division=0
    ))

# 6. Compare model performance
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="PR-AUC",
    ascending=False
)

print("\nMODEL COMPARISON")
print("================")
print(results_df.to_string(index=False))

# 7. Select best model based on PR-AUC
best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

print("\nBest model:", best_model_name)

# 8. Save model and results
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

joblib.dump(best_model, "models/best_model.pkl")

results_df.to_csv(
    "outputs/model_comparison.csv",
    index=False
)

# 9. Generate confusion matrix for best model
best_predictions = best_model.predict(X_test)

cm = confusion_matrix(y_test, best_predictions)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Normal", "Fraud"],
    yticklabels=["Normal", "Fraud"]
)

plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=300)
plt.close()

# 10. Feature importance for tree-based models
if best_model_name in ["Decision Tree", "Random Forest"]:

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": best_model.feature_importances_
    }).sort_values(
        by="Importance",
        ascending=False
    )

    importance.to_csv(
        "outputs/feature_importance.csv",
        index=False
    )

    plt.figure(figsize=(9, 6))

    sns.barplot(
        data=importance.head(10),
        x="Importance",
        y="Feature"
    )

    plt.title("Top 10 Important Graph Features")
    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png", dpi=300)
    plt.close()

print("\nTraining completed successfully!")
print("Saved model: models/best_model.pkl")
print("Saved comparison: outputs/model_comparison.csv")
print("Saved confusion matrix: outputs/confusion_matrix.png")
