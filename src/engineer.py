# src/commands/engineer.py

from pathlib import Path
import importlib.util
from typing import Callable, Optional

import click
import pandas as pd

# -----------------------------------------------------------------------------
# 1. Robust Import and Fallback for Core Utilities (FIXED)
# Ensure data_loader, load_from_prompt, and find_project_root are imported
try:
    import data_loader
    from data_loader import load_from_prompt, find_project_root, discover_data_files, prompt_for_file_choice 
except ImportError as e:
    class MockDataLoader:
        BASE_PATH = Path()
    data_loader = MockDataLoader()
    def find_project_root():
        return Path.cwd() 
    def load_from_prompt(head=5):
        raise click.ClickException("Data loading utilities unavailable.")
    def discover_data_files():
        raise click.ClickException("Data loading utilities unavailable.")
    def prompt_for_file_choice(files):
        raise click.ClickException("Data loading utilities unavailable.")
# -----------------------------------------------------------------------------

# 2. Base Paths (FIXED)
BASE_PATH = find_project_root()
DATA_DIR = BASE_PATH / "data"
# NOTE: We assume the FE scripts are in src/feature_engineering
FE_DIR = BASE_PATH / "src" / "feature_engineering" 


# --- Utility Functions (Identical to preprocess.py, but targeting FE) ---

def discover_fe_scripts() -> list[Path]:
    """Find all usable feature engineering scripts in src/feature_engineering/."""
    if not FE_DIR.exists():
        raise FileNotFoundError(f"Feature Engineering folder not found: {FE_DIR}")

    scripts: list[Path] = []
    for path in sorted(FE_DIR.iterdir()):
        if path.is_file() and path.suffix == ".py" and not path.name.startswith("_"):
            scripts.append(path)

    if not scripts:
        raise FileNotFoundError(f"No feature engineering scripts found in {FE_DIR}")

    return scripts


def prompt_for_script_choice(scripts: list[Path]) -> Path:
    """Prompt the user to choose a feature engineering script."""
    if len(scripts) == 1:
        only = scripts[0]
        click.echo(f"Only one FE script found: {only.stem}. Using it.\n")
        return only

    click.echo(f"\nAvailable Feature Engineering scripts in {FE_DIR}:")
    for i, p in enumerate(scripts, start=1):
        click.echo(f"  {i}. {p.stem}")

    choice = click.prompt(
        f"\nSelect a script [1-{len(scripts)}]",
        type=click.IntRange(1, len(scripts)),
    )
    return scripts[choice - 1]

def load_engineer_function(script_path: Path) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Dynamically load an engineering module and return its 'engineer' callable."""
    module_stem = script_path.stem
    
    try:
        spec = importlib.util.spec_from_file_location(module_stem, script_path)
        if spec is None:
             raise ImportError(f"Could not create spec for module: {script_path}")
             
        module = importlib.util.module_from_spec(spec)
        import sys
        sys.modules[module_stem] = module
        spec.loader.exec_module(module)
    except Exception as e:
        raise click.ClickException(f"Failed to load module from {script_path}: {e}")
        
    func: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None
    
    if hasattr(module, "engineer") and callable(getattr(module, "engineer")):
        func = getattr(module, "engineer")

    if func is None:
        raise click.ClickException(
            f"Module '{module_stem}' must define a callable "
            "'engineer(df: DataFrame) -> DataFrame'."
        )

    return func

# --- Main Command Logic ---

@click.command(name="engineer")
def engineer_command() -> None:
    """
    Load preprocessed data, run a chosen feature engineering script,
    and save the engineered output.
    """
    input_dir = DATA_DIR / "initial" # Input directory for engineer command
    
    # 1. Non-mutating way to list files in a specific directory
    if not input_dir.exists():
        raise click.ClickException(f"Input data directory not found: {input_dir}")
        
    try:
        # We manually call discover_data_files but then filter/map the paths
        # to the input_dir. Note: discover_data_files uses the global BASE_PATH.
        # This requires an internal modification to data_loader, or a manual file search here.
        
        # --- Safest approach: Manual file discovery for a fixed directory ---
        # Find all supported files directly in the 'initial' folder
        supported_exts = set(ext for cls in data_loader.BaseLoader.__subclasses__() for ext in cls.extensions)
        
        input_files: list[Path] = []
        for path in sorted(input_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in supported_exts:
                input_files.append(path)
                
        if not input_files:
            raise click.ClickException(f"No supported data files found in {input_dir}. Did you run 'AI preprocess'?")
            
        # Prompt the user to choose a file from the list
        chosen_path = prompt_for_file_choice(input_files)
        
        # Now we load the file using its full path.
        loader = data_loader.get_loader_for(chosen_path)
        df = loader.load()
        
        file_stem = chosen_path.stem
        table_name = loader.table if isinstance(loader, data_loader.SQLiteLoader) else None
        
        click.echo(f"\nLoaded: {loader.path}")
        click.echo(f"Shape:  {df.shape}\n")
        click.echo(df.head(5).to_string(index=False))
        # -------------------------------------------------------------------
        
    except Exception as e:
        raise click.ClickException(f"Failed to load input data from {input_dir}: {e}")
        
    # 2. Select Engineering Script
    scripts = discover_fe_scripts()
    chosen_script = prompt_for_script_choice(scripts)
    module_stem = chosen_script.stem

    # 3. Run Engineering
    engineer_func = load_engineer_function(chosen_script)
    click.echo(f"\nRunning feature engineering script: {module_stem}")

    engineered_df = engineer_func(df)

    # 4. Save Output
    out_dir = DATA_DIR / "intermediate"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean up the stem and create new name (e.g., raw_cleaned -> raw)
    out_stem = file_stem
    if out_stem.endswith("_cleaned"):
        out_stem = out_stem.removesuffix("_cleaned")
        
    out_stem = f"{out_stem}_{module_stem}"

    out_path = out_dir / f"{out_stem}.feather"
    engineered_df.to_feather(out_path)

    # --- Print Output Columns & Head ---
    click.echo("\n--- Columns in Engineered DataFrame ---")
    click.echo("\n".join(f"* {col}" for col in engineered_df.columns))
    click.echo("---------------------------------------\n")
    
    click.echo("\n--- Head of Engineered DataFrame ---")
    click.echo(
        engineered_df.head().to_string(
            na_rep="NaN"
        )
    )
    click.echo("------------------------------------")
    
    click.echo(f"\nSaved engineered data to: {out_path}")
    click.echo(f"Shape: {engineered_df.shape}")