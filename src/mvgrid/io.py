from __future__ import annotations

from pathlib import Path
import pandas as pd


def read_csv(path: str) -> pd.DataFrame:
    """Read a CSV using sensible defaults for operational datasets."""
    return pd.read_csv(path)


def write_parquet(df: pd.DataFrame, path: str) -> None:
    """Persist a dataframe as Parquet (fast, stable, ML-friendly)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def ensure_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Parse a column to datetime (timezone-naïve for reproducible training)."""
    out = df.copy()
    out[col] = pd.to_datetime(out[col], errors="coerce")
    return out

def read_parquet(path: str) -> pd.DataFrame:
    """Read a Parquet file (preferred format for ML training tables)."""
    return pd.read_parquet(path)
