"""
Week 1, step 2: inspect a news archive before committing to it.

    python inspect_news.py data/raw/news.csv
    python inspect_news.py data/raw/news.csv --start 2014-01-01 --keep

Reads the file in chunks, so a multi-GB CSV will not exhaust memory. Reports the
four things that decide whether this source is usable:

    1. date range          -> sets the project's sample period
    2. articles per year   -> catches a source that thins out at the edges
    3. ticker coverage     -> whether this is a news signal or a large-cap signal
    4. duplicate rate      -> how much wire-copy repetition Week 4 must strip

With --keep, writes a slim file holding only universe tickers inside the date
window, with body text dropped. That file is what the project actually uses.

    pip install pandas pyarrow lxml
"""

import argparse
import hashlib
import json
import requests, io
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

DATE_CANDIDATES = ["date", "datetime", "published", "publish_date", "published_at",
                   "timestamp", "time", "created_at", "Date"]
TICKER_CANDIDATES = ["ticker", "tickers", "symbol", "stock", "stock_symbol",
                     "related_tickers", "Ticker", "Stock_symbol"]
TEXT_CANDIDATES = ["headline", "title", "Article_title", "news_title", "text", "Headline"]


def pick(cols, candidates, label, override=None):
    if override:
        return override
    for c in candidates:
        if c in cols:
            return c
    raise SystemExit(
        f"could not find the {label} column. Columns present: {list(cols)}\n"
        f"pass it explicitly, e.g. --{label}-col <name>")


def sp500_universe() -> set:
    r = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                     headers={"User-Agent": "Mozilla/5.0 (research script)"}, timeout=30)
    tbl = pd.read_html(io.StringIO(r.text))[0]
    return {str(s).replace(".", "-").upper() for s in tbl["Symbol"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--date-col"), ap.add_argument("--ticker-col"), ap.add_argument("--headline-col")
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default=datetime.today().strftime("%Y-%m-%d"))
    ap.add_argument("--chunk", type=int, default=500_000)
    ap.add_argument("--keep", action="store_true", help="write the slim filtered file")
    a = ap.parse_args()

    path = Path(a.path)
    print(f"file: {path}  ({path.stat().st_size / 1e9:.2f} GB on disk)")

    head = (pd.read_parquet(path).head(5) if path.suffix == ".parquet"
            else pd.read_csv(path, nrows=5))
    dcol = pick(head.columns, DATE_CANDIDATES, "date", a.date_col)
    tcol = pick(head.columns, TICKER_CANDIDATES, "ticker", a.ticker_col)
    hcol = pick(head.columns, TEXT_CANDIDATES, "headline", a.headline_col)
    print(f"columns: date={dcol}  ticker={tcol}  headline={hcol}\n")

    universe = sp500_universe()
    rows = kept = 0
    dmin, dmax = None, None
    per_year, ticker_counts, ticker_months = Counter(), Counter(), set()
    seen_hashes, dupes = set(), 0
    slim_parts = []

    chunks = ([pd.read_parquet(path)] if path.suffix == ".parquet"
              else pd.read_csv(path, chunksize=a.chunk, low_memory=False))

    for ch in chunks:
        rows += len(ch)
        ch[dcol] = pd.to_datetime(ch[dcol], errors="coerce", utc=True).dt.tz_localize(None)
        ch = ch.dropna(subset=[dcol])
        if ch.empty:
            continue
        dmin = ch[dcol].min() if dmin is None else min(dmin, ch[dcol].min())
        dmax = ch[dcol].max() if dmax is None else max(dmax, ch[dcol].max())
        per_year.update(ch[dcol].dt.year.tolist())

        ch["_tkr"] = ch[tcol].astype(str).str.upper().str.strip()
        w = ch[ch[dcol].between(a.start, a.end)]
        ticker_counts.update(w["_tkr"].tolist())
        ticker_months.update(zip(w["_tkr"], w[dcol].dt.to_period("M").astype(str)))

        norm = ch[hcol].astype(str).str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True).str.strip()
        h = norm.map(lambda s: hashlib.md5(s.encode()).hexdigest()[:16])
        dupes += h.isin(seen_hashes).sum()
        seen_hashes.update(h.tolist())

        if a.keep:
            m = ch[dcol].between(a.start, a.end)
            if universe:
                m &= ch["_tkr"].isin(universe)
            sel = ch.loc[m, [dcol, "_tkr", hcol]].rename(
                columns={dcol: "ts", "_tkr": "ticker", hcol: "headline"})
            kept += len(sel)
            slim_parts.append(sel)
        print(f"  ...{rows:,} rows read", end="\r")

    print(f"\n\nrows                {rows:,}")
    print(f"date range          {dmin.date()} to {dmax.date()}")
    print(f"duplicate headlines {dupes:,}  ({dupes / max(rows,1):.1%})")
    print("\narticles per year")
    for y in sorted(per_year):
        bar = "#" * int(40 * per_year[y] / max(per_year.values()))
        print(f"  {y}  {per_year[y]:>10,}  {bar}")

    if universe:
        covered = {t for t in ticker_counts if t in universe}
        months = len({m for _, m in ticker_months})
        um = {(t, m) for t, m in ticker_months if t in universe}
        print(f"\nS&P 500 coverage (against today's list, {len(universe)} names)")
        print(f"  names appearing at all     {len(covered)} / {len(universe)}"
              f"  ({len(covered)/len(universe):.0%})")
        print(f"  ticker-month fill rate     {len(um) / max(len(universe)*months,1):.0%}")
        top = [f"{t}:{c:,}" for t, c in ticker_counts.most_common(5) if t in universe]
        print(f"  busiest names              {', '.join(top)}")
        thin = sum(1 for t in covered if ticker_counts[t] < 50)
        print(f"  names with under 50 total  {thin}")

    if a.keep and slim_parts:
        out = Path("data/interim/news_slim.parquet")
        out.parent.mkdir(parents=True, exist_ok=True)
        slim = pd.concat(slim_parts, ignore_index=True).drop_duplicates(["ts", "ticker", "headline"])
        slim.to_parquet(out, index=False)
        rec = {"file": str(out), "rows": len(slim), "source_file": str(path),
               "date_min": str(slim.ts.min()), "date_max": str(slim.ts.max()),
               "universe_filtered": bool(universe), "built_at": datetime.now().isoformat(timespec="seconds")}
        Path("reports/manifests").mkdir(parents=True, exist_ok=True)
        Path("reports/manifests/news_slim.json").write_text(json.dumps(rec, indent=2))
        print(f"\nslim file           {out}  ({len(slim):,} rows, "
              f"{out.stat().st_size / 1e6:.0f} MB)")

    print("\nPaste date range, row count, and coverage into the source sheet.")
    print("If the ticker-month fill rate is under ~30%, news is a large-cap feature here.")


if __name__ == "__main__":
    main()