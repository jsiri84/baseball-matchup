#!/usr/bin/env python3
"""One-off calibration: estimate league xwOBA / xBA / xSLG by bb_type.

Reads cached Statcast parquets under data/<year>/, dedupes pitches by
(game_pk, at_bat_number, pitch_number) so a BBE that appears in both a
batter and a pitcher cache file is counted once, then reports per-bb_type
means of estimated_woba_using_speedangle, estimated_ba_using_speedangle,
and estimated_slg_using_speedangle. Output is meant to be pasted into
matchup.py as LG_*_BY_BBTYPE constants.

Usage::

    python scripts/_estimate_league_bbtype_baselines.py            # use 2025
    python scripts/_estimate_league_bbtype_baselines.py --year 2024
    python scripts/_estimate_league_bbtype_baselines.py --year 2025 --year 2024
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

KEEP_COLS = [
    "game_pk", "at_bat_number", "pitch_number",
    "type", "bb_type",
    "estimated_woba_using_speedangle",
    "estimated_ba_using_speedangle",
    "estimated_slg_using_speedangle",
]

BB_TYPES = ["ground_ball", "line_drive", "fly_ball", "popup"]


def load_year(year: int) -> pd.DataFrame:
    folder = DATA / str(year)
    files = sorted(folder.glob("statcast_*.parquet"))
    if not files:
        print(f"[calibrate] no parquets under {folder}", file=sys.stderr)
        return pd.DataFrame(columns=KEEP_COLS)
    print(f"[calibrate] loading {len(files)} parquet(s) from {folder}")
    pieces: list[pd.DataFrame] = []
    for i, p in enumerate(files, 1):
        try:
            df = pd.read_parquet(p, columns=KEEP_COLS)
        except Exception as exc:                                      # noqa: BLE001
            print(f"  skip {p.name}: {exc.__class__.__name__}", file=sys.stderr)
            continue
        pieces.append(df)
        if i % 50 == 0:
            print(f"  {i}/{len(files)}")
    if not pieces:
        return pd.DataFrame(columns=KEEP_COLS)
    out = pd.concat(pieces, ignore_index=True)
    print(f"[calibrate] {year}: {len(out):,} raw rows pre-dedupe")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--year", action="append", type=int,
                    help="Season year(s) to read (repeatable). Default: 2025.")
    args = ap.parse_args()

    years = args.year or [2025]
    pieces = [load_year(y) for y in years]
    pieces = [p for p in pieces if not p.empty]
    if not pieces:
        sys.exit("[calibrate] no data loaded")
    df = pd.concat(pieces, ignore_index=True)

    before = len(df)
    df = df.drop_duplicates(subset=["game_pk", "at_bat_number", "pitch_number"])
    print(f"[calibrate] dedupe: {before:,} -> {len(df):,} rows")

    bbe = df[(df["type"] == "X") & df["bb_type"].notna()].copy()
    print(f"[calibrate] BBE with bb_type: {len(bbe):,}")

    rows = []
    for b in BB_TYPES:
        sub = bbe[bbe["bb_type"] == b]
        n = len(sub)
        n_xwoba = int(sub["estimated_woba_using_speedangle"].notna().sum())
        n_xba = int(sub["estimated_ba_using_speedangle"].notna().sum())
        n_xslg = int(sub["estimated_slg_using_speedangle"].notna().sum())
        rows.append({
            "bb_type": b,
            "n_bbe": n,
            "share": n / len(bbe) if len(bbe) else 0.0,
            "xwOBA": float(sub["estimated_woba_using_speedangle"].mean()),
            "n_xwoba": n_xwoba,
            "xBA": float(sub["estimated_ba_using_speedangle"].mean()),
            "n_xba": n_xba,
            "xSLG": float(sub["estimated_slg_using_speedangle"].mean()),
            "n_xslg": n_xslg,
        })

    out = pd.DataFrame(rows)
    print()
    print("Per-bb_type league baselines (years = "
          + ", ".join(str(y) for y in years) + "):")
    print(out.to_string(index=False, formatters={
        "share": lambda v: f"{v:.3f}",
        "xwOBA": lambda v: f"{v:.4f}",
        "xBA":   lambda v: f"{v:.4f}",
        "xSLG":  lambda v: f"{v:.4f}",
    }))

    print()
    print("# Paste into matchup.py:")
    print("LG_XWOBA_BY_BBTYPE = {")
    for r in rows:
        print(f"    {r['bb_type']!r}: {r['xwOBA']:.4f},")
    print("}")
    print("LG_XBA_BY_BBTYPE = {")
    for r in rows:
        print(f"    {r['bb_type']!r}: {r['xBA']:.4f},")
    print("}")
    print("LG_XSLG_BY_BBTYPE = {")
    for r in rows:
        print(f"    {r['bb_type']!r}: {r['xSLG']:.4f},")
    print("}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
