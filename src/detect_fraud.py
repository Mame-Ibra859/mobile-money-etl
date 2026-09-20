import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_fraud(df: pd.DataFrame, contamination: float = 0.001) -> pd.DataFrame:
    """Apply an Isolation Forest to flag anomalies as fraud."""
    feature_cols = [
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "balance_change_origin",
        "balance_change_destination",
        "amount_to_balance_ratio",
        "is_large_transaction",
    ]

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
    )

    predictions = model.fit_predict(df[feature_cols])
    df = df.copy()
    df["fraud_prediction"] = (predictions == -1).astype(int)

    return df
