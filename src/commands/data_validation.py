# src/commands/data_validation.py

import sys
from pathlib import Path

import click
import pandas as pd
import pandas.testing as tm


SUPPORTED_EXTS = {".feather", ".fth", ".csv", ".parquet"}


def _resolve_path(path_str: str) -> Path:
    """
    Resolve a file path.
    - If it exists as given, use it.
    - Otherwise, try looking under the ./data directory.
    """
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
    if ext not in SUPPORTED_EXTS:
        raise click.ClickException(
            f"Unsupported file extension '{ext}' for {path}. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTS))}"
        )

    if ext in {".feather", ".fth"}:
        return pd.read_feather(path)
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".parquet":
        return pd.read_parquet(path)

    # Should never reach here because of SUPPORTED_EXTS, but just in case
    raise click.ClickException(f"Don’t know how to load file: {path}")


@click.command(name="validate")
@click.argument("file1")
@click.argument("file2")
def validate(file1: str, file2: str):
    """
    Compare two data files (e.g. feather exports) and report whether they are
    identical (shape, columns, values).
    
      aicli validate data/raw.feather data/clean.feather
      aicli validate raw.feather clean.feather   # looks under ./data as fallback
    """
    # ---- Resolve and load ----
    p1 = _resolve_path(file1)
    p2 = _resolve_path(file2)

    click.echo(f"📂 File 1: {p1}")
    click.echo(f"📂 File 2: {p2}")

    try:
        df1 = _load_table(p1)
        df2 = _load_table(p2)
    except Exception as e:
        # ClickException will already be nicely formatted; others we wrap
        if isinstance(e, click.ClickException):
            raise
        raise click.ClickException(str(e))

    click.echo(f"➡ File 1 shape: {df1.shape[0]} rows × {df1.shape[1]} cols")
    click.echo(f"➡ File 2 shape: {df2.shape[0]} rows × {df2.shape[1]} cols")

    # ---- Quick shape check ----
    if df1.shape != df2.shape:
        click.echo("❌ DataFrames have different shapes.")
        sys.exit(1)

    # ---- Columns check ----
    if list(df1.columns) != list(df2.columns):
        click.echo("⚠ Column order or names differ; comparing with aligned columns.")
    else:
        click.echo("✅ Column names and order match.")

    # Align by sorted column names for comparison
    common_cols = sorted(set(df1.columns) & set(df2.columns))
    missing_in_1 = set(df2.columns) - set(df1.columns)
    missing_in_2 = set(df1.columns) - set(df2.columns)

    if missing_in_1:
        click.echo(f"⚠ Columns only in file 2: {sorted(missing_in_1)}")
    if missing_in_2:
        click.echo(f"⚠ Columns only in file 1: {sorted(missing_in_2)}")

    df1_aligned = df1[common_cols]
    df2_aligned = df2[common_cols]

    # ---- Deep equality check ----
    try:
        tm.assert_frame_equal(
            df1_aligned,
            df2_aligned,
            check_like=True,    # ignore row order if index matches? (row order still matters)
            check_dtype=False,  # relax dtype differences (e.g. int vs float)
            atol=1e-8,
            rtol=1e-5,
        )
    except AssertionError as e:
        click.echo("❌ DataFrames differ in content.")

        # Optional: small summary of how many cells differ
        try:
            diff_mask = (df1_aligned != df2_aligned) & ~(
                df1_aligned.isna() & df2_aligned.isna()
            )
            n_diff = int(diff_mask.to_numpy().sum())
            if n_diff > 0:
                click.echo(f"   → Roughly {n_diff} differing cells in common columns.")
        except Exception:
            # Don’t let diff summary crash the command
            pass

        # Print only the first line of the assertion message (pandas can be very verbose)
        first_line = str(e).splitlines()[0]
        click.echo(f"   Details: {first_line}")
        sys.exit(1)

    click.echo("✅ DataFrames are identical (within tolerance).")
    sys.exit(0)
