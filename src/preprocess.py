# /src/commands/preprocess.py
from pathlib import Path
import importlib
from typing import Callable, Optional

import click
import pandas as pd

# -----------------------------------------------------------------------------
# 1. Robust Import and Fallback for Core Utilities
# Uses absolute module imports which are correct when run via the CLI entry point
try:
    import data_loader
    from data_loader import load_from_prompt, find_project_root 
except ImportError as e:
    # Fallback/Stubs if core utilities fail to import (e.g., if data_loader.py is missing)
    class MockDataLoader:
        BASE_PATH = Path()
    data_loader = MockDataLoader()
    def find_project_root():
        return Path.cwd() # Fallback function
    def load_from_prompt(head=5):
        raise click.ClickException("Data loading utilities unavailable.")
# -----------------------------------------------------------------------------


# 2. Base Paths (Uses the robust find_project_root)
BASE_PATH = find_project_root()
DATA_DIR = BASE_PATH / "data"
PREPROCESS_DIR = BASE_PATH / "src" / "preprocess"


def discover_preprocess_scripts() -> list[Path]:
# ... (Functions are identical and correct) ...
    if not PREPROCESS_DIR.exists():
        raise FileNotFoundError(f"Preprocess folder not found: {PREPROCESS_DIR}")

    scripts: list[Path] = []
    for path in sorted(PREPROCESS_DIR.iterdir()):
        if path.is_file() and path.suffix == ".py" and not path.name.startswith("_"):
            scripts.append(path)

    if not scripts:
        raise FileNotFoundError(f"No preprocessing scripts found in {PREPROCESS_DIR}")

    return scripts

def prompt_for_script_choice(scripts: list[Path]) -> Path:
# ... (Identical and correct) ...
    if len(scripts) == 1:
        only = scripts[0]
        click.echo(f"Only one preprocessing script found: {only.stem}. Using it.\n")
        return only

    click.echo(f"\nAvailable Preprocessing scripts in {PREPROCESS_DIR}:")
    for i, p in enumerate(scripts, start=1):
        click.echo(f"  {i}. {p.stem}")

    choice = click.prompt(
        f"\nSelect a script [1-{len(scripts)}]",
        type=click.IntRange(1, len(scripts)),
    )
    return scripts[choice - 1]

def load_preprocess_function(script_path: Path) -> Callable[[pd.DataFrame], pd.DataFrame]:
# ... (Identical and correct) ...
    module_stem = script_path.stem
    
    try:
        # Load module spec from file location for robustness
        spec = importlib.util.spec_from_file_location(module_stem, script_path)
        if spec is None:
             raise ImportError(f"Could not create spec for module: {script_path}")
             
        # Create module object
        module = importlib.util.module_from_spec(spec)
        
        # This is critical: make the module available under its name
        import sys
        sys.modules[module_stem] = module
        
        # Execute the module code
        spec.loader.exec_module(module)
    except Exception as e:
        raise click.ClickException(f"Failed to load module from {script_path}: {e}")
        
    func: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None

    if hasattr(module, "preprocess") and callable(getattr(module, "preprocess")):
        func = getattr(module, "preprocess")

    if func is None:
        raise click.ClickException(
            f"Module '{module_stem}' must define a callable "
            "'preprocess(df: DataFrame) -> DataFrame'."
        )

    return func

# --- Main Command Logic ---

@click.command(name="preprocess")
def preprocess_command() -> Path:
    """
    Load data interactively, then run a chosen preprocessing script
    from src/preprocess/ and save the cleaned output.
    """
    
    # Interactively load data
    df, file_stem, table_name = load_from_prompt(head=5)

    # 1. Select Preprocessing Script
    scripts = discover_preprocess_scripts()
    chosen_script = prompt_for_script_choice(scripts)
    module_stem = chosen_script.stem

    # 2. Load the preprocessing function
    preprocess_func = load_preprocess_function(chosen_script)
    click.echo(f"\nRunning preprocessing script: {module_stem}")

    # 3. Run Preprocessing
    processed_df = preprocess_func(df)

    # --- Print Output Head & Columns ---
    # ... (Printing logic is identical and correct) ...
    click.echo("\n--- Columns in PROCESSED DataFrame (2 of 3) ---")
    click.echo("\n".join(f"* {col}" for col in processed_df.columns))
    click.echo("-----------------------------------------------\n")
    
    click.echo("\n--- Head of PROCESSED DataFrame (3 of 3) ---")
    click.echo(
        processed_df.head().to_string(
            na_rep="NaN"
        )
    )
    click.echo("------------------------------------------\n")
    # ----------------------------------------------------

    # Ensure output directory exists (Now points to AIAP22/data/initial)
    out_dir = DATA_DIR / "initial"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Decide on output name
    if table_name:
        out_stem = f"{table_name}_{module_stem}"
    else:
        out_stem = f"{file_stem}_{module_stem}"

    out_path = out_dir / f"{out_stem}.feather"
    processed_df.to_feather(out_path)

    click.echo(f"\nSaved preprocessed data to: {out_path}")
    click.echo(f"Shape: {processed_df.shape}")

    return out_path