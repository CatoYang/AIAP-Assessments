from pathlib import Path
import importlib.util
import json
import joblib
from typing import Callable, Optional, Tuple, Dict, Any, List

import click
import pandas as pd

# -----------------------------------------------------------------------------
# 1. Robust Import and Fallback for Core Utilities (FIXED)
# Ensure data_loader and essential utilities are imported using absolute module names
try:
    import data_loader
    from data_loader import (
        find_project_root, 
        BaseLoader, 
        SQLiteLoader, 
        get_loader_for, 
        prompt_for_file_choice
    )
except ImportError as e:
    # Minimal stubs if data_loader is not in the environment
    class MockDataLoader:
        BASE_PATH = Path()
        class MockBaseLoader:
            extensions = [".csv", ".json", ".feather", ".parquet", ".sqlite"]
        BaseLoader = MockBaseLoader
    data_loader = MockDataLoader()
    def find_project_root():
        return Path.cwd() 
    def get_loader_for(path):
        raise click.ClickException(f"Data loading utilities unavailable: {path}")
    def prompt_for_file_choice(files):
        raise click.ClickException("Data loading utilities unavailable.")

# -----------------------------------------------------------------------------

# 2. Base Paths (FIXED)
BASE_PATH = find_project_root()
DATA_DIR = BASE_PATH / "data"
MODELS_DIR = BASE_PATH / "src" / "models" # Source for model scripts

# --- Type Hinting for Model Execution ---
# The model script will return (metrics dictionary, optional trained model object)
ModelOutput = Tuple[Dict[str, Any], Optional[Any]]


# --- Utility Functions ---

def discover_model_scripts() -> List[Path]:
    """Find all usable model scripts in src/models/."""
    if not MODELS_DIR.exists():
        raise FileNotFoundError(f"Models folder not found: {MODELS_DIR}")

    scripts: List[Path] = []
    for path in sorted(MODELS_DIR.iterdir()):
        if path.is_file() and path.suffix == ".py" and not path.name.startswith("_"):
            scripts.append(path)

    if not scripts:
        raise FileNotFoundError(f"No model scripts found in {MODELS_DIR}")

    return scripts

def prompt_for_script_choice(scripts: List[Path]) -> Path:
    """Prompt the user to choose a model script."""
    if len(scripts) == 1:
        only = scripts[0]
        click.echo(f"Only one model script found: {only.stem}. Using it.\n")
        return only

    click.echo(f"\nAvailable Model scripts in {MODELS_DIR}:")
    for i, p in enumerate(scripts, start=1):
        click.echo(f"  {i}. {p.stem}")

    choice = click.prompt(
        f"\nSelect a script [1-{len(scripts)}]",
        type=click.IntRange(1, len(scripts)),
    )
    return scripts[choice - 1]

def load_model_function(script_path: Path) -> Callable[[pd.DataFrame], ModelOutput]:
    """
    Dynamically load a model module directly from its file path 
    and return its training callable.
    """
    
    module_stem = script_path.stem
    
    try:
        # Load module spec from file location
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

    func: Optional[Callable[[pd.DataFrame], ModelOutput]] = None
    
    # Check for 'train_evaluate' or 'run' functions
    if hasattr(module, "train_evaluate") and callable(getattr(module, "train_evaluate")):
        func = getattr(module, "train_evaluate")
    elif hasattr(module, "run") and callable(getattr(module, "run")):
        func = getattr(module, "run")

    if func is None:
        raise click.ClickException(
            f"Module '{module_stem}' must define a callable "
            "'train_evaluate(df: DataFrame) -> (metrics, model)' or "
            "'run(df: DataFrame) -> (metrics, model)'."
        )

    return func


# --- Main Command Logic ---

@click.command(name="model")
@click.option(
    '--save', 'save_model', 
    is_flag=True, 
    help='Save the trained model object to data/models/.'
)
def model_command(save_model: bool) -> None:
    """
    Load engineered data, train/evaluate a chosen model script, 
    and output performance metrics.
    """
    input_dir = DATA_DIR / "intermediate" # Input directory for model command
    
    # 1. Load Data from the fixed 'intermediate' directory (FIXED)
    if not input_dir.exists():
        raise click.ClickException(f"Input data directory not found: {input_dir}")
        
    try:
        # Find all supported files directly in the 'intermediate' folder
        # BaseLoader is used to correctly identify supported file extensions
        supported_exts = set(ext for cls in BaseLoader.__subclasses__() for ext in cls.extensions)
        
        input_files: list[Path] = []
        for path in sorted(input_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in supported_exts:
                input_files.append(path)
                
        if not input_files:
            raise click.ClickException(
                f"No supported data files found in {input_dir}. "
                "Did you run 'AI engineer'?"
            )
            
        click.echo(f"Searching for input data in: {input_dir}")
        chosen_path = prompt_for_file_choice(input_files)
        
        # Load the file using its full path.
        loader = get_loader_for(chosen_path)
        df = loader.load()
        
        file_stem = chosen_path.stem
        # SQLiteLoader table name extraction is handled by data_loader
        table_name = loader.table if isinstance(loader, SQLiteLoader) else None
        
        click.echo(f"\nLoaded: {loader.path}")
        click.echo(f"Shape:  {df.shape}\n")
        click.echo(df.head(5).to_string(index=False))
        
    except Exception as e:
        raise click.ClickException(f"Failed to load input data from {input_dir}: {e}")
        
    # Determine the base name for output files (e.g., 'phishing_data_df_drop')
    base_output_name = file_stem
    if table_name:
        base_output_name = table_name

    # 2. Select Model Script
    scripts = discover_model_scripts()
    chosen_script = prompt_for_script_choice(scripts)
    module_stem = chosen_script.stem

    # 3. Run Training and Evaluation
    model_func = load_model_function(chosen_script) 
    click.echo(f"Running model training and evaluation script: {module_stem}")

    # Pass the DataFrame to the model script
    metrics, trained_model = model_func(df)

    # 4. Output and Save Metrics
    metrics_dir = DATA_DIR / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = metrics_dir / f"{base_output_name}_{module_stem}_metrics.json"
    
    with open(metrics_file, 'w') as f:
        # Use simple dump to avoid issues with non-JSON serializable types
        # Note: If metrics contains non-serializable objects, the model script must handle conversion.
        json.dump(metrics, f, indent=4)
        
    click.echo("\n" + "="*50)
    click.echo(f"✅ METRICS for {module_stem}:")
    for k, v in metrics.items():
        # Clean up metric output for better readability
        if isinstance(v, (int, float)):
            click.echo(f"  - {k}: {v:.4f}")
        else:
            click.echo(f"  - {k}: {v}")
            
    click.echo("="*50)
    click.echo(f"Metrics saved to: {metrics_file}")

    # 5. Save Model (Optional, now with Y/N prompt)
    if trained_model is not None:
        save_confirmed = click.confirm(
            f"\nModel '{module_stem}' was trained. Do you want to save the object?",
            default=False
        )
        
        if save_confirmed:
            models_dir = DATA_DIR / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            model_file = models_dir / f"{base_output_name}_{module_stem}.joblib"
            
            joblib.dump(trained_model, model_file)
            click.echo(f"✅ Model saved to: {model_file}")
        else:
            click.echo("Model object discarded.")
    else:
        click.echo("⚠️ WARNING: No model object was returned by the script. Cannot save.")
        
    click.echo("Modeling command finished successfully.")