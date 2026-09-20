import pandas as pd


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered features useful for fraud detection."""
    df = df.copy()

    df["balance_change_origin"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["balance_change_destination"] = df["newbalanceDest"] - df["oldbalanceDest"]
    df["amount_to_balance_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1)
    df["is_large_transaction"] = (df["amount"] > df["amount"].quantile(0.99)).astype(int)
    
    return df
