# src/feature_engineering/df_drop.py

import pandas as pd
from typing import Optional, List

# List of columns to drop
COLUMNS_TO_DROP: List[str] = [
    "HasIframe",
    "NoOfiFrame_log",
    "Industry",
    "is_responsive",
]

# Columns that require one-hot encoding
ONE_HOT_COLUMNS: List[str] = [
    "Industry",
    "HostingProvider",
]

def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops specified columns and one-hot encodes selected categorical features.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame processed in the previous step (data/initial/).

    Returns
    -------
    pd.DataFrame
        The DataFrame with columns dropped and categorical features one-hot encoded.
    """
    
    # Start with a defensive copy
    df_engineered = df.copy()

    # 1. Drop the specified columns
    df_engineered = df_engineered.drop(
        columns=COLUMNS_TO_DROP, 
        errors='ignore'
    )

    # 2. One-hot encode the categorical columns
    # Only encode columns that exist (robust for different datasets)
    cols_to_encode = [col for col in ONE_HOT_COLUMNS if col in df_engineered.columns]

    if cols_to_encode:
        df_engineered = pd.get_dummies(
            df_engineered,
            columns=cols_to_encode,
            prefix=cols_to_encode,      # e.g., Industry_* , HostingProvider_*
            drop_first=True,           # keep all levels
            dummy_na=False              # do NOT encode NaN as separate category
        )

    return df_engineered
