"""Snapshot pregame slates + matchups + flattened projections for backtesting.

The live ``reports/`` and ``matchups/`` directories are gitignored (they
churn every day; the matchups files even get rewritten as lineups firm
up).  But for backtesting -- "did my new projection model do better than
the production one on the May 2026 slate?" -- we need a frozen,
reproducible snapshot of the pregame inputs PLUS the projections the
production model emitted.

This script consolidates that snapshot into ``data/historical/``:

  data/historical/
    slates/slate_<date>.json     verbatim copy of reports/<d>/_data/slate.json
    matchups/matchups_<date>.csv verbatim copy of the latest matchups CSV
    projections.parquet          flat row-per-(date,mlbam) with every
                                 summary_rows field + a model_alpha tag
                                 for future sweeps

A "snapshot date" qualifies if it has BOTH a slate.json AND graded
actuals in ``data/accuracy/hitter_results.parquet`` -- otherwise it's
useless for backtesting (you can't score a projection without an actual
to grade against).

Idempotent: re-running just refreshes the snapshot.  Pass
``--model-alpha X`` to tag the projections.parquet rows with the alpha
that produced them (defaults to 0.0, matching current production).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
import unicodedata
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).parent.resolve()
REPORTS_DIR = ROOT / "reports"
MATCHUPS_DIR = ROOT / "matchups"
ACCURACY_DIR = ROOT / "data" / "accuracy"
ARCHIVE_DIR = ROOT / "data" / "historical"
ARCHIVE_SLATES = ARCHIVE_DIR / "slates"
ARCHIVE_MATCHUPS = ARCHIVE_DIR / "matchups"
PROJECTIONS_PARQUET = ARCHIVE_DIR / "projections.parquet"

OUTCOME_CLASSES = ("K", "BB", "HBP", "1B", "2B", "3B", "HR", "Out")
_MATCHUPS_RE = re.compile(r"^matchups_(\d{4}-\d{2}-\d{2})_")


def _norm_name(name: str) -> str:
    """Match calibrate_mix_shift's name normalization (ASCII-fold, lower, strip)."""
    if not name:
        return ""
    s = unicodedata.normalize("NFKD", name)
    s = s.encode("ascii", "ignore").decode("ascii")
    return s.lower().strip()


def latest_matchups_csv(date_iso: str) -> Path | None:
    """Newest matchups CSV for ``date_iso`` (live mode timestamped one wins
    over the backfill one if both exist; for backfill-only dates the
    backfill file is the only candidate)."""
    if not MATCHUPS_DIR.exists():
        return None
    # Prefer real timestamped files over backfill stubs when both exist.
    timestamped = sorted(MATCHUPS_DIR.glob(f"matchups_{date_iso}_[0-9]*.csv"))
    if timestamped:
        return timestamped[-1]
    fallback = sorted(MATCHUPS_DIR.glob(f"matchups_{date_iso}_*.csv"))
    return fallback[-1] if fallback else None


def discover_dates() -> list[str]:
    """Dates that have BOTH a production slate.json AND graded actuals."""
    if not REPORTS_DIR.exists():
        return []
    slate_dates: set[str] = set()
    for child in REPORTS_DIR.iterdir():
        if child.is_dir() and re.match(r"^\d{4}-\d{2}-\d{2}$", child.name):
            if (child / "_data" / "slate.json").exists():
                slate_dates.add(child.name)
    graded: set[str] = set()
    hr_path = ACCURACY_DIR / "hitter_results.parquet"
    if hr_path.exists():
        graded = set(pd.read_parquet(hr_path, columns=["date"])["date"]
                     .astype(str).unique())
    return sorted(slate_dates & graded)


def snapshot_slate(date_iso: str) -> Path | None:
    """Archive ``slate.json`` with the per-batter HTML blobs stripped out.

    Roughly 96% of each slate's bytes are ``per_batter_html`` -- chunks
    of pre-rendered HTML that roundup.py + build_site.py consume.  None
    of that is needed for backtesting; the structured projection data
    lives in ``summary_rows`` and is preserved verbatim.  Stripping the
    HTML knocks 7.5 MB/date down to ~30 KB and lets the whole archive
    fit in git without LFS.

    The HTML can be regenerated at any time by re-running
    ``matchup.py --batch data/historical/matchups/matchups_<d>.csv``.
    """
    src = REPORTS_DIR / date_iso / "_data" / "slate.json"
    if not src.exists():
        return None
    slate = json.loads(src.read_text(encoding="utf-8"))
    for game in slate.get("games", []) or []:
        # Preserve schema -- keep the key, just drop the payload.
        if "per_batter_html" in game:
            game["per_batter_html"] = ["" for _ in game["per_batter_html"]]
    ARCHIVE_SLATES.mkdir(parents=True, exist_ok=True)
    dst = ARCHIVE_SLATES / f"slate_{date_iso}.json"
    dst.write_text(json.dumps(slate, separators=(",", ":")), encoding="utf-8")
    return dst


def snapshot_matchups(date_iso: str) -> Path | None:
    src = latest_matchups_csv(date_iso)
    if src is None:
        return None
    ARCHIVE_MATCHUPS.mkdir(parents=True, exist_ok=True)
    dst = ARCHIVE_MATCHUPS / f"matchups_{date_iso}.csv"
    # Some legacy live-fetch CSVs were saved in cp1252; re-encode to
    # canonical utf-8 in the archive so backtesting tools see a single
    # stable encoding everywhere.
    with src.open("r", encoding="utf-8", errors="replace", newline="") as fin:
        rows = list(csv.reader(fin))
    with dst.open("w", encoding="utf-8", newline="") as fout:
        csv.writer(fout).writerows(rows)
    return dst


def build_projections_frame(dates: Iterable[str],
                            model_alpha: float = 0.0) -> pd.DataFrame:
    """Flatten slate.json summary_rows into a row-per-(date, mlbam) table.

    mlbam isn't stored directly on each summary_row, so we recover it via
    a (date, hitter_team, normalized_name) join against the canonical
    actuals table (``data/accuracy/hitter_results.parquet``).  Rows that
    fail to resolve an mlbam are dropped -- they couldn't be backtested
    anyway since we'd have no actual to grade against.
    """
    hr_path = ACCURACY_DIR / "hitter_results.parquet"
    if not hr_path.exists():
        raise FileNotFoundError(f"missing actuals table: {hr_path}")
    hr_df = pd.read_parquet(hr_path)
    name_to_mlbam: dict[tuple[str, str, str], int] = {}
    for _, r in hr_df.iterrows():
        try:
            mlbam = int(r["mlbam"])
        except (TypeError, ValueError):
            continue
        key = (str(r["date"]), str(r["team"]), _norm_name(str(r["name"])))
        name_to_mlbam[key] = mlbam

    out_rows: list[dict] = []
    for d in dates:
        slate_path = REPORTS_DIR / d / "_data" / "slate.json"
        if not slate_path.exists():
            continue
        slate = json.loads(slate_path.read_text(encoding="utf-8"))
        generated_at = slate.get("generated_at", "")
        for game in slate.get("games", []) or []:
            team = str(game.get("hitter_team", ""))
            pitcher_name = str(game.get("pitcher_name", ""))
            pitcher_meta = game.get("pitcher_meta") or {}
            p_throws = pitcher_meta.get("p_throws") or ""
            pitcher_id = pitcher_meta.get("id")
            try:
                pitcher_id_int: int | None = int(pitcher_id) if pitcher_id else None
            except (TypeError, ValueError):
                pitcher_id_int = None
            matchup_key = str(game.get("matchup_key", ""))
            projected = bool(game.get("projected", False))
            pa_per_batter = game.get("pa_per_batter")
            for sr in game.get("summary_rows", []) or []:
                name = sr.get("name", "")
                mlbam = name_to_mlbam.get((d, team, _norm_name(name)))
                if mlbam is None:
                    continue  # not backtestable; skip
                pd_dict = sr.get("proj_dist") or {}
                row: dict = {
                    "date": d,
                    "mlbam": int(mlbam),
                    "name": name,
                    "hitter_team": team,
                    "matchup_key": matchup_key,
                    "pitcher_name": pitcher_name,
                    "pitcher_id": pitcher_id_int,
                    "p_throws": p_throws,
                    "stand": sr.get("stand", ""),
                    "spot": sr.get("spot"),
                    "projected": projected,
                    "pa_per_batter": pa_per_batter,
                    "proj_xwoba": sr.get("proj_xwoba"),
                    "proj_xba": sr.get("proj_xba"),
                    "proj_xslg": sr.get("proj_xslg"),
                    "proj_xwoba_raw": sr.get("proj_xwoba_raw"),
                    "bbtype_adj_pts": sr.get("bbtype_adj_pts"),
                    "park_pf": sr.get("park_pf"),
                    "park_pts": sr.get("park_pts"),
                    "form_d14_xwoba": sr.get("form_d14_xwoba"),
                    "form_d14_xwoba_platoon": sr.get("form_d14_xwoba_platoon"),
                    "form_d14_platoon_label": sr.get("form_d14_platoon_label"),
                    "delta_pts": sr.get("delta_pts"),
                    "proj_k_pct": sr.get("k_pct"),
                    "proj_bb_pct": sr.get("bb_pct"),
                    "proj_hr_pct": sr.get("hr_pct"),
                    "proj_hit_pct": sr.get("hit_pct"),
                    "proj_ob_pct": sr.get("ob_pct"),
                    "proj_hardhit_pct": sr.get("proj_hardhit_pct"),
                    "proj_whiff_pct": sr.get("proj_whiff_pct"),
                    "proj_xwoba_on_contact": sr.get("proj_xwoba_on_contact"),
                    "best_pitch": sr.get("best_pitch"),
                    "worst_pitch": sr.get("worst_pitch"),
                    "verdict_label": sr.get("verdict_label"),
                    "model_alpha": float(model_alpha),
                    "generated_at": generated_at,
                }
                for c in OUTCOME_CLASSES:
                    row[f"proj_dist_{c}"] = pd_dict.get(c)
                out_rows.append(row)

    if not out_rows:
        return pd.DataFrame()
    df = pd.DataFrame(out_rows)
    # Stable sort makes commits / diffs reproducible.
    df = df.sort_values(["date", "matchup_key", "hitter_team", "spot", "name"])
    df = df.reset_index(drop=True)
    return df


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-alpha", type=float, default=0.0,
                    help="PITCHER_MIX_SHIFT_ALPHA value to tag rows with "
                         "(default 0.0 = current production)")
    ap.add_argument("--dates", default=None,
                    help="comma-separated subset of YYYY-MM-DD (default: "
                         "every date that has both a slate.json and "
                         "graded actuals)")
    args = ap.parse_args(argv)

    if args.dates:
        dates = sorted(d.strip() for d in args.dates.split(",") if d.strip())
    else:
        dates = discover_dates()
    if not dates:
        print("[archive] no qualifying dates found", file=sys.stderr)
        return 1
    print(f"[archive] {len(dates)} dates: {dates[0]} .. {dates[-1]}")

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    slate_n = 0
    matchup_n = 0
    for d in dates:
        if snapshot_slate(d) is not None:
            slate_n += 1
        else:
            print(f"[archive] {d}: no slate.json to snapshot")
        if snapshot_matchups(d) is not None:
            matchup_n += 1
        else:
            print(f"[archive] {d}: no matchups CSV to snapshot")
    print(f"[archive] copied {slate_n} slate snapshots, "
          f"{matchup_n} matchups CSVs")

    df = build_projections_frame(dates, model_alpha=args.model_alpha)
    if df.empty:
        print("[archive] WARN: projections frame is empty", file=sys.stderr)
        return 1
    df.to_parquet(PROJECTIONS_PARQUET, index=False)
    print(f"[archive] wrote {PROJECTIONS_PARQUET}: {len(df):,} rows, "
          f"{df['mlbam'].nunique():,} unique batters, "
          f"alpha={args.model_alpha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
