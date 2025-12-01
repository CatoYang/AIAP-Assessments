# src/preprocess/basic1.py
import pandas as pd
import numpy as np

def preprocess(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Take the *raw* dataframe and return a cleaned / feature-engineered version
    matching the schema you've been using in EDA.

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
    # 0. Basic cleaning of string columns
    # -----------------------------------------------------------------
    for col in df.select_dtypes(include=['object', 'string']):
        # Ensure the column isn't all-NaN before trying string operations
        if not df[col].isnull().all():
            df[col] = (
                df[col]
                .astype(str) # Convert to string to prevent errors if mixed types exist
                .str.lower()
                .str.strip()
                .str.replace(r'\s+', '', regex=True)
                .str.replace('_', '', regex=False)
                #.str.replace('.', '', regex=False)   # this might be to aggresive if theres URLs in the future
            )

    # -----------------------------------------------------------------
    # 1. Drop junk index column if present
    # -----------------------------------------------------------------
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # -----------------------------------------------------------------
    # 2. Rename columns to target / flags
    # -----------------------------------------------------------------
    rename_map = {
        "label": "is_legitimate",
        "Robots": "has_robotstxt",
        "IsResponsive": "is_responsive",
    }
    # Only rename if present
    rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    # -----------------------------------------------------------------
    # 3. LineOfCode: NaN flag + log transform, drop raw
    # -----------------------------------------------------------------
    if "LineOfCode" in df.columns:
        # 1. Create NaN/Missingness Flag
        df["LineOfCode_NaN"] = df["LineOfCode"].isna().astype(int)
        
        # 2. Apply Log Transform and sentinel value
        df["LineOfCode_log"] = np.where(
                df["LineOfCode"].notna() & (df["LineOfCode"] > 0),
                np.log(df["LineOfCode"]),
                -1
        )
        
        # 3. Drop the original raw count column
        df = df.drop(columns=["LineOfCode"])

    # -----------------------------------------------------------------
    # 4. LargestLineLength: log transform, drop raw
    # -----------------------------------------------------------------
    if "LargestLineLength" in df.columns:
        mask_ll = df["LargestLineLength"].notna() & (df["LargestLineLength"] >= 0)
        df["LargestLineLength_log"] = np.nan
        # log1p is safer if there are any zeros; effect is tiny for large values
        df.loc[mask_ll, "LargestLineLength_log"] = np.log1p(
            df.loc[mask_ll, "LargestLineLength"]
        )
        df = df.drop(columns=["LargestLineLength"])

    # -----------------------------------------------------------------
    # 4.5 NoOfPopup: Flag + Winsorize (cap at 31)
    # -----------------------------------------------------------------
    if "NoOfPopup" in df.columns:
        # 1. Create Binary Flag for presence
        df["HasPopup"] = (df["NoOfPopup"] > 0).astype(int)
        
        # 2. Winsorize/Cap the raw count at 31
        capped_series = df["NoOfPopup"].clip(upper=31)
        
        # 3. Apply Log(1 + x) Transformation
        df["NoOfPopup_Log1p"] = np.log1p(capped_series)
        
        # 4. Drop the original column
        df = df.drop(columns=["NoOfPopup"])
    # -----------------------------------------------------------------
    # 5. Self / external refs: flags + log1p, drop raw counts
    # -----------------------------------------------------------------
    if "NoOfSelfRef" in df.columns:
        df["HasSelfRef"] = (df["NoOfSelfRef"] > 0).astype(int)
        df["SelfRef_log1p"] = np.log1p(df["NoOfSelfRef"].clip(lower=0))
        df = df.drop(columns=["NoOfSelfRef"])

    if "NoOfExternalRef" in df.columns:
        df["HasExternalRef"] = (df["NoOfExternalRef"] > 0).astype(int)
        df["ExternalRef_log1p"] = np.log1p(df["NoOfExternalRef"].clip(lower=0))
        df = df.drop(columns=["NoOfExternalRef"])

    # -----------------------------------------------------------------
    # 6. Images: fix negatives, flag, log1p, drop raw
    # -----------------------------------------------------------------
    if "NoOfImage" in df.columns:
        # fix data issue: negative counts
        df["NoOfImage"] = df["NoOfImage"].abs()
        df["Has_Images"] = (df["NoOfImage"] > 0).astype(int)
        df["NoOfImage_log1p"] = np.log1p(df["NoOfImage"])
        df = df.drop(columns=["NoOfImage"])

    # -----------------------------------------------------------------
    # 7. Iframe: flag + log1p count, drop raw
    # -----------------------------------------------------------------
    if "NoOfiFrame" in df.columns:
        df["HasIframe"] = (df["NoOfiFrame"] > 0).astype(int)
        df["NoOfiFrame_log"] = np.log1p(df["NoOfiFrame"].clip(lower=0))
        df = df.drop(columns=["NoOfiFrame"])

    # -----------------------------------------------------------------
    # 8. URL redirect helpers (keep original 0/1 + helper flags)
    # -----------------------------------------------------------------
    if "NoOfURLRedirect" in df.columns:
        df["HasURLRedirect"] = (df["NoOfURLRedirect"] > 0).astype(int)

    if "NoOfSelfRedirect" in df.columns:
        df["HasSelfRedirect"] = (df["NoOfSelfRedirect"] > 0).astype(int)

    # -----------------------------------------------------------------
    # 9. Domain age -> NewDomains flag
    # -----------------------------------------------------------------
    if "DomainAgeMonths" in df.columns:
        df["NewDomains"] = (df["DomainAgeMonths"] < 1).astype(int)

    return df