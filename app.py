
import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import joblib
import os

st.set_page_config(
    page_title="GraphGuard | Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

st.markdown("""
<style>
[data-testid="stMetric"] {
    background-color: #1c2333;
    padding: 18px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    transactions = pd.read_csv("data/raw/transactions.csv")
    features = pd.read_csv("data/processed/graph_features.csv")
    comparison = pd.read_csv("outputs/model_comparison.csv")
    return transactions, features, comparison


@st.cache_resource
def load_model():
    return joblib.load("models/best_model.pkl")


@st.cache_resource
def build_graph():
    graph = nx.DiGraph()

    for row in transactions.itertuples(index=False):
        sender = row.sender
        receiver = row.receiver

        if graph.has_edge(sender, receiver):
            graph[sender][receiver]["weight"] += 1
        else:
            graph.add_edge(sender, receiver, weight=1)

    return graph



transactions, features, comparison = load_data()
model = load_model()
G = build_graph()


model_features = list(model.feature_names_in_)

features["predicted_fraud"] = model.predict(
    features[model_features]
)




fraud_transactions = features[
    features["predicted_fraud"] == 1
]

fraud_transaction_ids = set(
    fraud_transactions["transaction_id"]
)

fraud_rows = transactions[
    transactions["transaction_id"].isin(fraud_transaction_ids)
]

fraud_accounts = set(fraud_rows["sender"])
fraud_accounts.update(fraud_rows["receiver"])

fraud_pairs = set(
    zip(fraud_rows["sender"], fraud_rows["receiver"])
)


st.title("🛡️ GraphGuard")
st.subheader("Graph-Based Intelligent Financial Fraud Detection")

st.write(
    "Detect suspicious financial transactions using "
    "machine learning and account network analysis."
)

st.divider()

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Dashboard",
    [
        "Overview",
        "Fraud Detection",
        "Transaction Network",
        "Model Performance"
    ]
)

if page == "Overview":

    st.header("System Overview")

    total_transactions = len(transactions)
    total_fraud = int(transactions["is_fraud"].sum())
    normal_transactions = total_transactions - total_fraud
    total_accounts = G.number_of_nodes()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Transactions", f"{total_transactions:,}")
    col2.metric("Fraud Transactions", f"{total_fraud:,}")
    col3.metric("Normal Transactions", f"{normal_transactions:,}")
    col4.metric("Accounts", f"{total_accounts:,}")

    st.subheader("Fraud Monitoring Alerts")

    flagged_count = len(fraud_transactions)
    flagged_account_count = len(fraud_accounts)

    col1, col2, col3 = st.columns(3)

    col1.metric("Predicted Fraud Transactions", flagged_count)
    col2.metric("Flagged Accounts", flagged_account_count)

    col3.metric(
        "Flagged Transaction Rate",
        f"{flagged_count / total_transactions * 100:.2f}%"
    )

    if flagged_count > 0:
        st.warning(
            f"⚠️ {flagged_count} transactions require investigation."
        )
    else:
        st.success("No suspicious transactions detected.")

    st.subheader("Recent Fraud Alerts")

    if not fraud_rows.empty:
        st.dataframe(
            fraud_rows.sort_values(
                "timestamp",
                ascending=False
            ).head(10),
            use_container_width=True
        )
    else:
        st.info("No fraud alerts available.")


    st.subheader("Transaction Classification")

    fig = go.Figure(data=[
        go.Pie(
            labels=["Normal", "Fraud"],
            values=[normal_transactions, total_fraud],
            hole=0.55
        )
    ])

    fig.update_layout(
        template="plotly_dark",
        title="Fraud vs Normal Transactions"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recent Transactions")

    st.dataframe(
        transactions.tail(10),
        use_container_width=True
    )


elif page == "Fraud Detection":

    st.header("Transaction Fraud Detection")

    st.write(
        "Select a transaction to calculate its fraud risk "
        "using the trained machine learning model."
    )

    transaction_ids = features["transaction_id"].tolist()

    selected_id = st.selectbox(
        "Select Transaction ID",
        transaction_ids
    )

    selected_row = features[
        features["transaction_id"] == selected_id
    ].iloc[0]

    X = features.drop(
        columns=["transaction_id", "is_fraud"]
    )
    input_data = selected_row[model_features].to_frame().T

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    col1, col2 = st.columns(2)

    col1.metric(
        "Fraud Probability",
        f"{probability * 100:.2f}%"
    )

    col2.metric(
        "Actual Dataset Label",
        "Fraud" if selected_row["is_fraud"] == 1 else "Normal"
    )

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("Suspicious Transaction Detected")
    else:
        st.success("Transaction Classified as Normal")

    st.progress(float(probability))

    st.caption(
        "Fraud probability is the model's estimated score, "
        "not a verified probability of real-world fraud."
    )

    st.subheader("Transaction Details")

    transaction_details = transactions[
        transactions["transaction_id"] == selected_id
    ]

    st.dataframe(
        transaction_details,
        use_container_width=True
    )

    st.subheader("Graph Features Used")

    st.dataframe(
        input_data.T.rename(columns={input_data.index[0]: "Value"}),
        use_container_width=True
    )



elif page == "Transaction Network":

    st.header("Financial Transaction Network")

    st.write(
        "Red accounts and connections indicate involvement "
        "in transactions predicted as fraudulent."
    )

    st.markdown(
        "🔴 Fraud-flagged account   🟠 Selected account   🔵 Normal account"
    )

    accounts = sorted(G.nodes())

    selected_account = st.selectbox(
        "Choose an Account",
        accounts
    )

    neighbours = set(G.predecessors(selected_account))
    neighbours.update(G.successors(selected_account))
    neighbours.add(selected_account)

    visible_nodes = list(neighbours)[:35]
    subgraph = G.subgraph(visible_nodes)

    positions = nx.spring_layout(
        subgraph,
        seed=42,
        k=0.7
    )

    normal_edge_x = []
    normal_edge_y = []
    fraud_edge_x = []
    fraud_edge_y = []

    for source, target in subgraph.edges():

        x0, y0 = positions[source]
        x1, y1 = positions[target]

        if (source, target) in fraud_pairs:
            fraud_edge_x.extend([x0, x1, None])
            fraud_edge_y.extend([y0, y1, None])
        else:
            normal_edge_x.extend([x0, x1, None])
            normal_edge_y.extend([y0, y1, None])

    normal_edges = go.Scatter(
        x=normal_edge_x,
        y=normal_edge_y,
        mode="lines",
        line=dict(width=1, color="#64748b"),
        hoverinfo="none",
        name="Normal connection"
    )

    fraud_edges = go.Scatter(
        x=fraud_edge_x,
        y=fraud_edge_y,
        mode="lines",
        line=dict(width=3, color="#ef4444"),
        hoverinfo="none",
        name="Fraud-flagged connection"
    )

    node_x = []
    node_y = []
    node_text = []
    node_colors = []

    for node in subgraph.nodes():

        x, y = positions[node]

        node_x.append(x)
        node_y.append(y)

        node_text.append(
            f"Account: {node}<br>"
            f"Incoming: {G.in_degree(node)}<br>"
            f"Outgoing: {G.out_degree(node)}<br>"
            f"Fraud flagged: {'Yes' if node in fraud_accounts else 'No'}"
        )

        if node == selected_account:
            node_colors.append("#f59e0b")
        elif node in fraud_accounts:
            node_colors.append("#ef4444")
        else:
            node_colors.append("#38bdf8")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=list(subgraph.nodes()),
        textposition="top center",
        hovertext=node_text,
        hoverinfo="text",
        marker=dict(
            size=15,
            color=node_colors,
            line=dict(width=1, color="white")
        ),
        name="Accounts"
    )

    fig = go.Figure(
        data=[normal_edges, fraud_edges, node_trace]
    )

    fig.update_layout(
        title=f"Network around {selected_account}",
        template="plotly_dark",
        height=650,
        legend=dict(
            orientation="h",
            y=-0.1
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            visible=False
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            visible=False
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Incoming Connections",
        G.in_degree(selected_account)
    )

    col2.metric(
        "Outgoing Connections",
        G.out_degree(selected_account)
    )

    col3.metric(
        "Fraud-Flagged Account",
        "Yes" if selected_account in fraud_accounts else "No"
    )

    st.subheader("Fraud-Flagged Accounts")

    flagged_visible = [
        account for account in visible_nodes
        if account in fraud_accounts
    ]

    if flagged_visible:
        st.dataframe(
            pd.DataFrame({
                "Account": flagged_visible,
                "Incoming": [
                    G.in_degree(account)
                    for account in flagged_visible
                ],
                "Outgoing": [
                    G.out_degree(account)
                    for account in flagged_visible
                ]
            }),
            use_container_width=True
        )
    else:
        st.success("No fraud-flagged accounts in this network view.")




elif page == "Model Performance":

    st.header("Machine Learning Model Evaluation")

    st.write(
        "Comparison of traditional machine learning algorithms "
        "trained using transaction and graph-derived features."
    )

    st.dataframe(
        comparison.style.format({
            "Accuracy": "{:.2%}",
            "Precision": "{:.2%}",
            "Recall": "{:.2%}",
            "F1 Score": "{:.2%}",
            "PR-AUC": "{:.2%}",
            "ROC-AUC": "{:.2%}"
        }),
        use_container_width=True
    )

    st.subheader("Model Comparison")

    metric = st.selectbox(
        "Choose Evaluation Metric",
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "PR-AUC",
            "ROC-AUC"
        ]
    )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=comparison["Model"],
            y=comparison[metric],
            text=[
                f"{value:.2%}"
                for value in comparison[metric]
            ],
            textposition="outside"
        )
    )

    fig.update_layout(
        title=f"Model Comparison - {metric}",
        yaxis_title=metric,
        template="plotly_dark",
        yaxis_range=[0, 1.1]
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Confusion Matrix")

    if os.path.exists("outputs/confusion_matrix.png"):
        st.image(
            "outputs/confusion_matrix.png",
            caption="Confusion Matrix of Selected Model"
        )

    if os.path.exists("outputs/feature_importance.png"):
        st.subheader("Important Graph Features")
        st.image("outputs/feature_importance.png")


st.divider()

st.caption(
    "GraphGuard | Academic ML Project | "
    "Synthetic transaction data for demonstration purposes"
)
