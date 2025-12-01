# src/feature_engineering/df_ratio.py
import pandas as pd
import numpy as np
from typing import Optional, List

# Columns that require one-hot encoding
ONE_HOT_COLUMNS: List[str] = [
    "Industry",
    "HostingProvider",
]

def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates new features (ratios, interactions, binning) and
    one-hot encodes selected categorical features.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame processed in the previous step (data/initial/).

    Returns
    -------
    pd.DataFrame
        The DataFrame with new features engineered and categorical features 
        one-hot encoded.
    """
    
    # Start with a defensive copy
    df_engineered = df.copy()
    
    # Define a small epsilon for stable division
    epsilon = 1e-6 

    # --- 1. New Feature Creation ---
    
    # Check for the existence of columns before creating features
    
    # # 1.1 Total Redirects (Sum)
    # if all(col in df_engineered.columns for col in ['NoOfURLRedirect', 'NoOfSelfRedirect']):
    #     df_engineered['TotalRedirects'] = (
    #         df_engineered['NoOfURLRedirect'] + df_engineered['NoOfSelfRedirect']
    #     )
    
    # # 1.2 Suspicious Flag Count (Sum of binary features)
    # suspicious_cols = ['HasIframe', 'HasPopup', 'HasExternalRef', 'HasURLRedirect', 'HasSelfRef']
    # # Filter for columns actually present in the DataFrame
    # present_flags = [col for col in suspicious_cols if col in df_engineered.columns]
    
    # if present_flags:
    #     df_engineered['Suspicious_Flag_Count'] = df_engineered[present_flags].sum(axis=1)

    # 1.3 External to Self Reference Ratio (Ratio)
    if all(col in df_engineered.columns for col in ['ExternalRef_log1p', 'SelfRef_log1p']):
        # Use log1p values in the ratio, with epsilon for stable division
        df_engineered['External_to_Self_Ratio'] = (
            df_engineered['ExternalRef_log1p'] / (df_engineered['SelfRef_log1p'] + epsilon)
        )

    # 1.4 iFrame Density (Ratio)
    if all(col in df_engineered.columns for col in ['NoOfiFrame_log', 'LineOfCode_log']):
        df_engineered['iFrame_Density'] = (
            df_engineered['NoOfiFrame_log'] / (df_engineered['LineOfCode_log'] + epsilon)
        )
        
    # 1.5 Maximum Line Length Ratio (Ratio)
    if all(col in df_engineered.columns for col in ['LargestLineLength_log', 'LineOfCode_log']):
        df_engineered['MaxLineLength_Ratio'] = (
            df_engineered['LargestLineLength_log'] / (df_engineered['LineOfCode_log'] + epsilon)
        )
        
    # # 1.6 Domain Age Binning (Categorical)
    # if 'DomainAgeMonths' in df_engineered.columns:
    #     bins = [-np.inf, 3, 12, 36, np.inf]
    #     labels = ['Very New (0-3m)', 'New (4-12m)', 'Young (1-3y)', 'Established (>3y)']
    #     df_engineered['DomainAgeCategory'] = pd.cut(
    #         df_engineered['DomainAgeMonths'], 
    #         bins=bins, 
    #         labels=labels, 
    #         right=True
    #     )
    #     df_engineered = pd.get_dummies(df_engineered, columns=['DomainAgeCategory'], drop_first=True)

    # # 1.7 New Domain * High Redirect Interaction (Interaction)
    # if all(col in df_engineered.columns for col in ['NewDomains', 'NoOfURLRedirect']):
    #     df_engineered['NewDomain_HighRedirect'] = (
    #         df_engineered['NewDomains'] * df_engineered['NoOfURLRedirect']
    #     )

    
    # One-hot encode the categorical columns defined in ONE_HOT_COLUMNS
    cols_to_encode = [col for col in ONE_HOT_COLUMNS if col in df_engineered.columns]

    if cols_to_encode:
        df_engineered = pd.get_dummies(
            df_engineered,
            columns=cols_to_encode,
            prefix=cols_to_encode, 
            drop_first=True,
            dummy_na=False
        )

    return df_engineered