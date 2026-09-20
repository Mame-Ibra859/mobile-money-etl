import pandas as pd


def extract(path: str) -> pd.DataFrame:
    """Load raw PaySim dataset from CSV."""
    return pd.read_csv(path)
