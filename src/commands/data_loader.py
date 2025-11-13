# src/commands/data_loader.py
from abc import ABC, abstractmethod
from pathlib import Path
import sqlite3
import importlib

import pandas as pd
import click

# --- Base Configuration -----------------------------------------------------------
# Define base path relative to the project root
BASE_PATH = Path(__file__).resolve().parent.parent.parent / "data"

# --- Base class --------------------------------------------------------------

class BaseLoader(ABC):
    """Abstract base class for file loaders."""
    extensions: tuple[str, ...] = ()

    def __init__(self, filename: str):
        # Convert to Path and resolve relative to BASE_PATH if needed
        path = Path(filename)
        if not path.is_absolute():
            path = BASE_PATH / path
        self.path = path.resolve()

        if not self.path.exists():
            raise FileNotFoundError(f"Data file not found: {self.path}")

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Load the file and return a pandas DataFrame."""
        pass


# --- Subclasses --------------------------------------------------------------

class CSVLoader(BaseLoader):
    extensions = (".csv",)

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self.path)


class ExcelLoader(BaseLoader):
    extensions = (".xls", ".xlsx")

    def load(self) -> pd.DataFrame:
        return pd.read_excel(self.path)


class JSONLoader(BaseLoader):
    extensions = (".json",)

    def load(self) -> pd.DataFrame:
        try:
            return pd.read_json(self.path)
        except ValueError:
            return pd.read_json(self.path, lines=True)


class ParquetLoader(BaseLoader):
    extensions = (".parquet",)

    def load(self) -> pd.DataFrame:
        return pd.read_parquet(self.path)


class FeatherLoader(BaseLoader):
    extensions = (".feather",)

    def load(self) -> pd.DataFrame:
        return pd.read_feather(self.path)


class SQLiteLoader(BaseLoader):
    extensions = (".db", ".sqlite", ".sqlite3")

    def __init__(self, filename: str, table: str | None = None):
        super().__init__(filename)
        self.table = table

    def load(self) -> pd.DataFrame:
        if self.table is None:
            raise ValueError("SQLiteLoader requires a table name. Pass it when instantiating.")

        conn = sqlite3.connect(self.path)
        try:
            df = pd.read_sql_query(f"SELECT * FROM {self.table}", conn)
        finally:
            conn.close()
        return df


# --- Factory to select the right subclass -----------------------------------

def get_loader_for(path: Path) -> BaseLoader:
    ext = path.suffix.lower()
    for cls in BaseLoader.__subclasses__():
        if ext in cls.extensions:
            # BaseLoader will resolve relative to BASE_PATH if needed
            return cls(str(path))
    raise ValueError(f"No loader available for extension {ext}")


# --- Click command: load & clean & save as feather --------------------------

@click.command(name="load-clean")
@click.argument("filename")
@click.argument("cleaning_module")
@click.option(
    "--table",
    "-t",
    help="Table name for SQLite databases (.db, .sqlite, .sqlite3).",
)
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=BASE_PATH / "intermediate",
    show_default=True,
    help="Directory to write the cleaned feather file into.",
)
def load_clean(filename: str, cleaning_module: str, table: str | None, out_dir: Path) -> None:
    """
    Load a raw data FILE, apply a cleaning script, and save as a Feather file.

    FILENAME: raw data file (relative to project 'data/' or an absolute path)
    CLEANING_MODULE: name of a module in 'src/cleaning/' that defines clean(df).
    """
    # --- Load raw data using your existing loaders ---
    path = Path(filename)

    try:
        loader = get_loader_for(path)
    except FileNotFoundError as e:
        raise click.ClickException(str(e))
    except ValueError as e:
        raise click.ClickException(str(e))

    if isinstance(loader, SQLiteLoader):
        if table is None:
            # List tables in the SQLite file
            try:
                conn = sqlite3.connect(loader.path)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = [row[0] for row in cursor.fetchall()]
            finally:
                conn.close()

            if not tables:
                table_list = "(no tables found)"
            else:
                table_list = ", ".join(tables)

            raise click.ClickException(
                f"SQLite database detected, but no --table provided.\n"
                f"Available tables: {table_list}\n"
                f"Use: --table <table_name>"
            )
        loader.table = table

    df_raw = loader.load()
    click.echo(f"[load-clean] Loaded {loader.path} with {loader.__class__.__name__}, shape={df_raw.shape}")

    # --- Import cleaning function dynamically ---
    try:
        # assumes a package 'cleaning' on your PYTHONPATH / installed package
        module = importlib.import_module(f"cleaning.{cleaning_module}")
    except ModuleNotFoundError as e:
        raise click.ClickException(
            f"Could not import module 'cleaning.{cleaning_module}'. "
            f"Make sure 'src/cleaning/{cleaning_module}.py' exists and is importable."
        ) from e

    if not hasattr(module, "clean"):
        raise click.ClickException(
            f"Cleaning module 'cleaning.{cleaning_module}' must define a function 'clean(df)'."
        )

    clean_fn = getattr(module, "clean")

    # --- Apply cleaning ---
    df_clean = clean_fn(df_raw)
    if df_clean is None:
        raise click.ClickException("Cleaning function returned None. It must return a pandas DataFrame.")

    # --- Prepare output path (Feather) ---
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"{Path(filename).stem}_{cleaning_module}.feather"
    out_path = out_dir / out_name

    # --- Save as Feather ---
    df_clean.to_feather(out_path)
    click.echo(f"[load-clean] Saved cleaned data to {out_path} with shape {df_clean.shape}")