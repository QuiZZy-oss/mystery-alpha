"""
Week 1, step 1: macro test pull (FRED Treasury yields + VIX).

Run from your project root:
    pip install pandas pandas_datareader yfinance
    python pull_macro.py

Creates the folder layout, saves raw CSVs, writes a manifest, and prints the
numbers that go into the source sheet: date range, row count, missing days.

No API key needed. pandas_datareader reads FRED's public CSV endpoint.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pandas_datareader.data as web
import yfinance as yf

START = "2010-01-01"
END = datetime.today().strftime("%Y-%m-%d")

FOLDERS = ["config", "data/raw", "data/interim", "data/processed",
           "src", "notebooks", "reports/manifests", "tests"]


def setup_folders():
    for f in FOLDERS:
        Path(f).mkdir(parents=True, exist_ok=True)
    gitignore = Path(".gitignore")
    if not gitignore.exists():
        gitignore.write_text(
            ".env\n*.key\ndata/\n*.parquet\n*.pkl\n__pycache__/\n"
            ".ipynb_checkpoints/\n.venv/\n"
        )
    print(f"folders ready, .gitignore {'created' if not gitignore.exists() else 'present'}")


def report(name: str, df: pd.DataFrame) -> dict:
    """Print and return the fields the source sheet asks for."""
    idx = pd.to_datetime(df.index)
    rec = {
        "source": name,
        "rows": int(len(df)),
        "date_min": str(idx.min().date()),
        "date_max": str(idx.max().date()),
        "columns": list(df.columns),
        "missing_cells": int(df.isna().sum().sum()),
        "pulled_at": datetime.now().isoformat(timespec="seconds"),
    }
    print(f"\n{name}")
    print(f"  rows        {rec['rows']:,}")
    print(f"  date range  {rec['date_min']} to {rec['date_max']}")
    print(f"  columns     {rec['columns']}")
    print(f"  missing     {rec['missing_cells']:,} cells")
    return rec


def save(df: pd.DataFrame, path: str, rec: dict):
    df.to_csv(path)
    rec["file"] = path
    rec["sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
    stem = Path(path).stem
    Path(f"reports/manifests/{stem}.json").write_text(json.dumps(rec, indent=2))
    print(f"  saved       {path}  (sha {rec['sha256']})")


def main():
    setup_folders()

    # --- FRED: 1-year and 10-year Treasury constant maturity yields ---
    # DGS1 is the r in any discounting you do; DGS10 gives the slope.
    fred = web.DataReader(["DGS1", "DGS10"], "fred", START, END)
    rec = report("FRED DGS1, DGS10", fred)
    save(fred, "data/raw/fred_rates.csv", rec)

    # --- VIX ---
    # auto_adjust=False keeps the raw Close column; yfinance changed this default.
    vix = yf.download("^VIX", start=START, end=END, auto_adjust=False, progress=False)
    if isinstance(vix.columns, pd.MultiIndex):       # yfinance returns MultiIndex columns
        vix.columns = vix.columns.get_level_values(0)
    vix = vix[["Close"]].rename(columns={"Close": "vix_close"})
    rec = report("Yahoo ^VIX", vix)
    save(vix, "data/raw/vix.csv", rec)

    # --- alignment check: do the two calendars agree? ---
    merged = fred.join(vix, how="outer")
    both = merged.dropna()
    print("\nalignment")
    print(f"  union of dates        {len(merged):,}")
    print(f"  dates with all fields {len(both):,}")
    print(f"  FRED-only dates       {len(merged) - len(vix.dropna()):,}")
    print("\nPaste the date range and row count into the source sheet, "
          "and today's date into 'tested'.")


if __name__ == "__main__":
    main()