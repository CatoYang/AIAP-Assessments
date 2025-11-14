# src/commands/data_validation.py

import sys
from pathlib import Path
import click
import pandas as pd
import pandas.testing as tm


SUPPORTED_EXTS = {".feather", ".fth", ".csv", ".parquet"}


def _resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.exists():
        return p

    data_path = Path("data") / path_str
    if data_path.exists():
        return data_path

    raise click.ClickException(
        f"File not found: {path_str} (also tried {data_path})"
    )


def _load_table(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext in {".feather", ".fth"}:
        return pd.read_feather(path)
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".parquet":
        return pd.read_parquet(path)

    raise click.ClickException(
        f"Unsupported file type '{ext}'. Supported: {SUPPORTED_EXTS}"
    )


@click.command(name="validate")
@click.argument("file1")
@click.argument("file2")
@click.option("--sample", "-s", default=10, show_default=True,
              help="Number of differing rows to display.")
def validate_command(file1: str, file2: str, sample: int):
    """
    Compare two data files and print differences.
    """

    p1 = _resolve_path(file1)
    p2 = _resolve_path(file2)

    click.echo(f"📂 File 1: {p1}")
    click.echo(f"📂 File 2: {p2}")

    df1 = _load_table(p1)
    df2 = _load_table(p2)

    # --- Shape differences ---
    if df1.shape != df2.shape:
        click.echo("❌ DataFrames differ in shape.")
        click.echo(f" - File 1 shape: {df1.shape}")
        click.echo(f" - File 2 shape: {df2.shape}")
    else:
        click.echo("✅ Shapes match.")

    # --- Column differences ---
    cols1 = list(df1.columns)
    cols2 = list(df2.columns)

    if cols1 != cols2:
        click.echo("\n❌ Column sets / order differ:")
        missing_1 = set(cols2) - set(cols1)
        missing_2 = set(cols1) - set(cols2)

        if missing_1:
            click.echo(f" - Columns only in file2: {sorted(missing_1)}")
        if missing_2:
            click.echo(f" - Columns only in file1: {sorted(missing_2)}")

        click.echo(f" - Column order file1: {cols1}")
        click.echo(f" - Column order file2: {cols2}")

    else:
        click.echo("✅ Columns match.")

    # Align common columns for deeper comparisons
    common_cols = sorted(set(df1.columns) & set(df2.columns))
    df1a = df1[common_cols]
    df2a = df2[common_cols]

    # --- Full DataFrame equality check ---
    try:
        tm.assert_frame_equal(
            df1a,
            df2a,
            atol=1e-8,
            rtol=1e-5,
            check_dtype=False,
            check_like=True,
        )
        click.echo("\n🎉 DataFrames are identical (within tolerance).")
        sys.exit(0)

    except AssertionError:
        click.echo("\n❌ DataFrames differ in content.")
    
    # --- Deep diff ---
    click.echo("\n🔍 Computing cell-level differences...")

    diff_mask = (df1a != df2a) & ~(df1a.isna() & df2a.isna())
    total_diff = diff_mask.to_numpy().sum()

    click.echo(f"Total differing cells: {total_diff}")

    # Per-column summary
    col_diff_counts = diff_mask.sum()
    click.echo("\n📊 Differences per column:")
    for col, count in col_diff_counts.items():
        if count > 0:
            click.echo(f" - {col}: {count} differing cells")

    # Extract rows with differences
    differing_rows = diff_mask.any(axis=1)
    diff_indices = df1a.index[differing_rows]

    click.echo(f"\n🔎 Showing up to {sample} differing rows:\n")

    # Display sample rows
    for i, idx in enumerate(diff_indices[:sample]):
        click.echo(f"--- Row {idx} ---")
        click.echo("File 1:")
        click.echo(df1a.loc[idx].to_string())
        click.echo("File 2:")
        click.echo(df2a.loc[idx].to_string())
        click.echo("")

    click.echo("❗ Files differ.")
    sys.exit(1)
