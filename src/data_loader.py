# src/data_loader.py
from abc import ABC, abstractmethod
from pathlib import Path
import sqlite3
import pandas as pd
import os

# --- Path Configuration --------------------------------------------------------

def find_project_root(marker_file: str = "pyproject.toml") -> Path:
    """
    Traverse up from the current working directory to find the project root,
    marked by the presence of a specific file (pyproject.toml).
    """
    current_path = Path.cwd()
    
    # Traverse up max 5 levels (arbitrary limit to prevent endless search)
    for _ in range(5): 
        if (current_path / marker_file).exists():
            return current_path
        # Stop if we hit the file system root
        if current_path == current_path.parent:
            break
        current_path = current_path.parent
        
    # Fallback: If marker not found, assume CWD is the project root.
    return Path.cwd()

# Define project root and the data folder based on the discovered root
PROJECT_ROOT = find_project_root()
BASE_PATH = PROJECT_ROOT / "data" 
# Ensure BASE_PATH is accessible for the functions below, 
# but the loader will check for existence later.

# --- Base class --------------------------------------------------------------

class BaseLoader(ABC):
    """Abstract base class for file loaders."""
    extensions: tuple[str, ...] = ()

    def __init__(self, filename: str | Path):
        # Convert to Path and resolve relative to BASE_PATH if needed
        path = Path(filename)
        
        # If the path is relative, prefix it with the determined BASE_PATH
        if not path.is_absolute():
            path = BASE_PATH / path
            
        self.path = path.resolve() # Resolve the final, absolute path

        if not self.path.exists():
            # This FileNotFoundError will now correctly point to BASE_PATH/filename
            raise FileNotFoundError(f"Data file not found: {self.path}")

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Load the file and return a pandas DataFrame."""
        raise NotImplementedError

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
            # Fallback for JSON Lines
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

    def __init__(self, filename: str | Path, table: str | None = None):
        super().__init__(filename)
        self.table = table

    def load(self) -> pd.DataFrame:
        if self.table is None:
            raise ValueError("SQLiteLoader requires a table name. Set .table before calling load().")

        conn = sqlite3.connect(self.path)
        try:
            df = pd.read_sql_query(f"SELECT * FROM {self.table}", conn)
        finally:
            conn.close()
        return df

# --- Factory -----------------------------------------------------------------
def get_loader_for(path: Path) -> BaseLoader:
    ext = path.suffix.lower()
    for cls in BaseLoader.__subclasses__():
        if ext in cls.extensions:
            return cls(path)
    raise ValueError(f"No loader available for extension {ext}")

# --- Helpers for interactive selection ---------------------------------------
def discover_data_files() -> list[Path]:
    """
    Return a list of files under BASE_PATH that match any known loader extension.
    """
    if not BASE_PATH.exists():
        raise FileNotFoundError(f"Data folder not found: {BASE_PATH}")

    # Collect all supported extensions from subclasses
    supported_exts: set[str] = set()
    for cls in BaseLoader.__subclasses__():
        supported_exts.update(cls.extensions)

    files: list[Path] = []
    for path in sorted(BASE_PATH.iterdir()):
        if path.is_file() and path.suffix.lower() in supported_exts:
            files.append(path)

    if not files:
        raise FileNotFoundError(f"No supported data files found in {BASE_PATH}")

    return files

def prompt_for_file_choice(files: list[Path]) -> Path:
    """
    If multiple files: print a numbered list and prompt the user.
    If only one file: automatically choose it.
    """
    print(f"Data folder: {BASE_PATH}\n")

    if len(files) == 1:
        only = files[0]
        print(f"Found a single file: {only.name} – automatically selecting it.")
        return only

    print("Available data files:")
    for i, f in enumerate(files, start=1):
        print(f"  {i}. {f.name}")

    while True:
        choice = input(f"\nSelect a file [1-{len(files)}]: ").strip()
        if not choice.isdigit():
            print("Please enter a valid integer.")
            continue

        idx = int(choice)
        if 1 <= idx <= len(files):
            return files[idx - 1]
        else:
            print(f"Please enter a number between 1 and {len(files)}.")

def list_sqlite_tables(db_path: Path) -> list[str]:
    """
    List non-internal tables in a SQLite database.
    """
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
        )
        tables = [row[0] for row in cursor.fetchall()]
    finally:
        conn.close()
    return tables

def prompt_for_sqlite_table(db_path: Path) -> str:
    """
    Prompt user for a SQLite table to load.
    If only one table exists, select it automatically.
    """
    tables = list_sqlite_tables(db_path)

    if not tables:
        raise RuntimeError(f"No tables found in SQLite database: {db_path}")

    if len(tables) == 1:
        only = tables[0]
        print(f"Found a single table in {db_path.name}: {only} – automatically selecting it.")
        return only

    print(f"\nSQLite file selected: {db_path.name}")
    print("Available tables:")
    for i, t in enumerate(tables, start=1):
        print(f"  {i}. {t}")

    while True:
        choice = input(f"\nSelect a table [1-{len(tables)}]: ").strip()
        if not choice.isdigit():
            print("Please enter a valid integer.")
            continue

        idx = int(choice)
        if 1 <= idx <= len(tables):
            return tables[idx - 1]
        else:
            print(f"Please enter a number between 1 and {len(tables)}.")

# --- Public API --------------------------------------------------------------
def load_from_prompt(head: int = 5) -> tuple[pd.DataFrame, Path]:
    """
    Interactively:
    1. List files in the `data/` folder
    2. If >1 file: ask the user to choose one
       If 1 file: automatically select it
    3. For SQLite:
       - List tables, auto-select if only one
       - Otherwise prompt for which table to load
    4. Load the file into a DataFrame
    5. Print a small preview

    Returns
    -------
    (df, filename)
        df   : The loaded DataFrame for further processing
        file_stem = file name for naming subsequent outputs
        table_name = table name for naming subsequent outputs if loaded with SQL
    """
    files = discover_data_files()
    chosen_path = prompt_for_file_choice(files)

    loader = get_loader_for(chosen_path)

    # If it's SQLite, ask (or auto-select) the table name
    if isinstance(loader, SQLiteLoader):
        loader.table = prompt_for_sqlite_table(loader.path)

    df = loader.load()

    file_stem = chosen_path.stem            # removes extension
    table_name = loader.table if isinstance(loader, SQLiteLoader) else None

    print(f"\nLoaded: {loader.path}")
    print(f"Loader: {loader.__class__.__name__}")
    if table_name is not None:
        print(f"Table:  {table_name}")
    print(f"Shape:  {df.shape}\n")

    print(df.head(head).to_string(index=False))

    return df, file_stem, table_name

# --- Script entrypoint -------------------------------------------------------
# Use this for independent testing of this script (python -m src.data_loader)
if __name__ == "__main__":
    # When run as a script: do an interactive load
    load_from_prompt(head=5)