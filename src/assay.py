# src/assay.py
import click
import pandas as pd
from typing import Optional

# --- Import from data_loader.py ------------------------------------------------
try:
    from .data_loader import load_from_prompt
except ImportError:
    # Fallback for running the script directly (e.g., python src/assay.py)
    from data_loader import load_from_prompt

# --- EDA Core Logic (To be filled in later) -----------------------------------

def perform_quick_eda(
    df: pd.DataFrame, 
    file_stem: str, 
    table_name: Optional[str] = None
) -> None:
    """
    Placeholder for the comprehensive EDA logic.
    This function will receive the loaded DataFrame and file/table names.
    
    TODO: Implement the required EDA steps here, such as:
    1. Print summary statistics (df.describe()).
    2. Print data types and non-null counts (df.info()).
    3. Analyze missing values.
    4. Analyze unique values for categorical columns.
    5. Generate simple plots (optional, e.g., histograms).
    """
    
    click.echo("\n" + "="*50)
    click.echo(f"| Starting Quick EDA for: {file_stem}")
    if table_name:
        click.echo(f"| (Table: {table_name})")
    click.echo("="*50)
    
    # --- Basic Info Already Handled by load_from_prompt (Shape, Head) ---
    
    # Placeholder for df.info()
    click.echo("\n### 💾 Data Types and Non-Null Counts (df.info()) ###")
    df.info(verbose=False, memory_usage=False) # Use verbose=False to keep it quick
    
    # Placeholder for Missing Value Analysis
    click.echo("\n### ❓ Missing Values Analysis ###")
    missing_count = df.isnull().sum()
    missing_percent = (missing_count / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing_count, 
        'Missing %': missing_percent
    })
    
    # Filter to only show columns with missing values and sort
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values(
        by='Missing Count', ascending=False
    )
    
    if not missing_df.empty:
        click.echo(missing_df.to_markdown(floatfmt=".2f"))
    else:
        click.echo("✅ No missing values found in any column.")
        
    # Placeholder for Summary Statistics
    click.echo("\n### 🔢 Summary Statistics (df.describe()) ###")
    click.echo(df.describe(include='all').to_markdown(floatfmt=".2f"))
    
    click.echo("\n" + "="*50)
    click.echo("| Quick EDA Complete.")
    click.echo("="*50 + "\n")


# --- Click Command Definition --------------------------------------------------

@click.command(name="assay")
def assay_command() -> None:
    """
    Interactively load a dataset from the 'data/' folder and perform a quick EDA.
    """
    click.echo("\n--- 🔎 AI Assay: Quick Data Exploration ---")

    try:
        # load_from_prompt handles file discovery, selection, loading, and preview
        # It returns (DataFrame, file_stem, table_name)
        df, file_stem, table_name = load_from_prompt()
    except FileNotFoundError as e:
        click.echo(f"\n🛑 ERROR: {e}")
        return
    except ValueError as e:
        click.echo(f"\n🛑 ERROR: {e}")
        return
    except Exception as e:
        click.echo(f"\n🛑 An unexpected error occurred during data loading: {e}")
        return

    # Once the data is loaded, pass it to the EDA function
    perform_quick_eda(df, file_stem, table_name)