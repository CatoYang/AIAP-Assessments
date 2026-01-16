# src/preprocess/temp.py
# temp as in template

import pandas as pd
import numpy as np

def preprocess(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generalized preprocessing template. 
    Toggle sections or edit column lists as needed based on EDA

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe as loaded from loader

    Returns
    -------
    df : pd.DataFrame
        Cleaned dataframe with engineered features.
    """
    df = df.copy()

    # -----------------------------------------------------------------
    # 1. DROP JUNK COLUMNS
    # -----------------------------------------------------------------
    cols_to_drop = ["Unnamed: 0", "id", "timestamp"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # -----------------------------------------------------------------
    # 2. RENAME COLUMNS
    # -----------------------------------------------------------------
    rename_map = {
        "label": "is_legitimate",
        "Robots": "has_robotstxt",
        "IsResponsive": "is_responsive",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # -----------------------------------------------------------------
    # 3. STRING CLEANING (Global or Specific)
    # -----------------------------------------------------------------
    # Select specific object columns or all object columns
    str_cols = df.select_dtypes(include=['object', 'string']).columns
    
    for col in str_cols:
        if not df[col].isnull().all():
            df[col] = (
                df[col]
                .str.lower()
                .str.strip()                            # Remove all whitespace between characters
                .str.replace(r'\s+', '', regex=True)    # Remove underscores 
                .str.replace('_', '', regex=False)
                # .str.replace('.', '', regex=False)    # Remove dots (toggle carefully - can break URLs/Filenames)
                # .str.replace('-', '', regex=False)    # Remove hyphens
                # .str.replace('@', '', regex=False)    # Remove @ symbols
            )

    # -----------------------------------------------------------------
    # 4. NUMERIC: CLIPPING & LOG TRANSFORMS (log1p)
    # -----------------------------------------------------------------
    # # List columns that benefit from log(1+x) to squash outliers
    # log_transform_cols = [
    #     "NoOfPopup", "NoOfSelfRef", "NoOfExternalRef", 
    #     "NoOfiFrame", "LargestLineLength", "NoOfImage"
    # ]

    # for col in log_transform_cols:
    #     if col in df.columns:
    #         # Fix negatives (e.g., NoOfImage) and clip/winsorize if needed
    #         df[col] = df[col].abs().fillna(0)
            
    #         # Optional: Add Clipping (Winsorizing)
    #         # if col == "NoOfPopup": df[col] = df[col].clip(upper=31)
            
    #         df[f"{col}_log1p"] = np.log1p(df[col])
    #         # df = df.drop(columns=[col]) # Toggle to drop original

    # -----------------------------------------------------------------
    # 5. NUMERIC: BINARY FLAGS (Presence Check)
    # -----------------------------------------------------------------
    # # Create 1/0 flags if the value is > 0
    # binary_flag_cols = [
    #     "NoOfPopup", "NoOfSelfRef", "NoOfExternalRef", 
    #     "NoOfiFrame", "NoOfURLRedirect", "NoOfSelfRedirect"
    # ]

    # for col in binary_flag_cols:
    #     if col in df.columns:
    #         new_name = col.replace("NoOf", "Has") if "NoOf" in col else f"has_{col}"
    #         df[new_name] = (df[col] > 0).astype(int)

    # -----------------------------------------------------------------
    # 6. SPECIAL Handling
    # -----------------------------------------------------------------
    # # Domain Age logic
    # if "DomainAgeMonths" in df.columns:
    #     df["is_new_domain"] = (df["DomainAgeMonths"] < 1).astype(int)

    # # LineOfCode specific (Log + Sentinel for NaNs)
    # if "LineOfCode" in df.columns:
    #     df["LineOfCode_NaN"] = df["LineOfCode"].isna().astype(int)
    #     df["LineOfCode_log"] = np.where(
    #         df["LineOfCode"].notna() & (df["LineOfCode"] > 0),
    #         np.log(df["LineOfCode"]),
    #         -1 # Sentinel for missing/zero
    #     )

    # -----------------------------------------------------------------
    # 7. Infinite values and remainders
    # -----------------------------------------------------------------
    # Replace any inf/-inf resulting from math with NaN, then fill
    # df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    return df