# src/data_loader.py
from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd
import sqlite3

# --- Base Configuration -----------------------------------------------------------
# Define base path relative to the project root
BASE_PATH = Path(__file__).resolve().parent.parent / "data"

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
# Factory to select the right subclass---------------------------
def get_loader_for(path: Path) -> BaseLoader:
    ext = path.suffix.lower()
    for cls in BaseLoader.__subclasses__():
        if ext in cls.extensions:
            return cls(path)
    raise ValueError(f"No loader available for extension {ext}")

import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Load data file into pandas DataFrame")
    parser.add_argument("--file", "-f", required=True, help="Path to data file")
    parser.add_argument("--head", type=int, default=5, help="Rows to preview")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        raise FileNotFoundError(path)

    loader = get_loader_for(path)
    df = loader.load()
    print(f"Loaded {path} with {loader.__class__.__name__}, shape={df.shape}")
    print(df.head(args.head).to_string(index=False))

if __name__ == "__main__":
    main()