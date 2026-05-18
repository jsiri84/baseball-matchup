"""Backfill historical projections and graded actuals for the season.

For each date D in ``[--start .. --end]``:

  1. Ensure ``matchups/matchups_<D>_*.csv`` exists.  If not, build it from
     the MLB Stats API boxscores via :mod:`fetch_historical_lineups`.
  2. Run ``matchup.py --batch <csv> --date <D> --mix-shift-alpha 0 --force``
     to write the baseline ``reports/<D>/_data/slate.json`` (alpha=0 is
     bit-identical to current production behavior; calibrate_mix_shift.py
     re-runs each date with non-zero alphas against this baseline).
  3. Run ``postgame.py --date <D>`` to grade the slate and append rows
     to ``data/accuracy/hitter_results.parquet`` / ``pa_results.parquet``.

Defaults to ``[2026-03-27 .. yesterday]`` (season-start through last
completed game day).  Pass ``--skip-existing`` to no-op dates that
already have both a slate.json AND graded rows in hitter_results.parquet.

Writes a per-date manifest to ``data/accuracy/backfill_manifest.csv``
tracking which dates succeeded / failed at which step.

Usage::

    python backfill.py                           # season-to-date
    python backfill.py --start 2026-04-01 --end 2026-04-15
    python backfill.py --skip-existing           # resume after a crash
    python backfill.py --dry-run                 # just print what would run
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import date as dateclass, timedelta
from pathlib import Path

from fetch_historical_lineups import fetch_historical
from fetch_lineups import MATCHUPS_DIR, MATCHUPS_FILE_RE

ROOT = Path(__file__).parent.resolve()
REPORTS_DIR = ROOT / "reports"
ACCURACY_DIR = ROOT / "data" / "accuracy"
SEASON_START = dateclass(2026, 3, 27)

MANIFEST_PATH = ACCURACY_DIR / "backfill_manifest.csv"
MANIFEST_FIELDS = (
    "date", "csv_path", "n_games", "matchup_status",
    "postgame_status", "error",
)


def _existing_matchups_csv(d: dateclass) -> Path | None:
    """Return the latest matchups CSV already on disk for date ``d``, if any."""
    if not MATCHUPS_DIR.exists():
        return None
    candidates: list[tuple[str, Path]] = []
    for p in MATCHUPS_DIR.glob(f"matchups_{d.isoformat()}_*.csv"):
        m = MATCHUPS_FILE_RE.match(p.name)
        if m:
            candidates.append((m.group(2), p))
        else:
            # backfill-tagged file (matchups_<d>_backfill.csv)
            candidates.append(("zzz_backfill", p))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def _slate_exists(d: dateclass) -> bool:
    return (REPORTS_DIR / d.isoformat() / "_data" / "slate.json").exists()


def _graded_dates() -> set[str]:
    """Return the set of date strings already represented in hitter_results.parquet."""
    path = ACCURACY_DIR / "hitter_results.parquet"
    if not path.exists():
        return set()
    try:
        import pandas as pd
        df = pd.read_parquet(path, columns=["date"])
        return set(df["date"].astype(str).unique())
    except Exception as e:
        print(f"[backfill] WARN could not read graded dates from "
              f"{path}: {e}", file=sys.stderr)
        return set()


def _date_range(start: dateclass, end: dateclass):
    cur = start
    while cur <= end:
        yield cur
        cur = cur + timedelta(days=1)


def _run(cmd: list[str], log_label: str) -> tuple[bool, str]:
    """Run ``cmd`` as a subprocess; capture combined output for diagnostics."""
    print(f"  -> {log_label}: {' '.join(cmd)}")
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
        )
    except FileNotFoundError as e:
        return (False, f"{log_label} not found: {e}")
    if proc.returncode != 0:
        tail = (proc.stdout + proc.stderr).splitlines()
        snippet = "\n".join(tail[-12:])
        return (False, f"{log_label} rc={proc.returncode}\n{snippet}")
    return (True, "")


def _append_manifest(row: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_file = not MANIFEST_PATH.exists()
    with MANIFEST_PATH.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if new_file:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in MANIFEST_FIELDS})


def process_date(d: dateclass, *, skip_existing: bool,
                 skip_postgame: bool, dry_run: bool) -> dict:
    """Run the 3-step backfill for a single date.  Returns a manifest row."""
    iso = d.isoformat()
    row: dict = {
        "date": iso,
        "csv_path": "",
        "n_games": 0,
        "matchup_status": "pending",
        "postgame_status": "pending",
        "error": "",
    }

    # Step 1: ensure matchups CSV
    csv_path = _existing_matchups_csv(d)
    if csv_path is None:
        if dry_run:
            print(f"[backfill] {iso}: would fetch historical lineups")
            row["matchup_status"] = "skipped (dry-run)"
            row["postgame_status"] = "skipped (dry-run)"
            return row
        out_path, kept, _total = fetch_historical(d, MATCHUPS_DIR, verbose=False)
        if out_path is None:
            row["matchup_status"] = "no games"
            row["error"] = "fetch_historical returned no games"
            return row
        csv_path = out_path
        row["n_games"] = kept
    else:
        # Count games (unique matchup_keys) for the manifest.
        try:
            keys: set[str] = set()
            with csv_path.open("r", encoding="utf-8") as f:
                for r in csv.reader(f):
                    if r:
                        keys.add(r[0])
            row["n_games"] = len(keys)
        except Exception:
            pass
    row["csv_path"] = str(csv_path.relative_to(ROOT))

    # Step 2: matchup.py (skip if slate already exists AND we asked to)
    have_slate = _slate_exists(d)
    if skip_existing and have_slate:
        row["matchup_status"] = "skipped (existing)"
        print(f"[backfill] {iso}: slate.json already exists; skipping matchup")
    else:
        if dry_run:
            print(f"[backfill] {iso}: would run matchup.py --batch {csv_path}")
            row["matchup_status"] = "skipped (dry-run)"
        else:
            ok, err = _run(
                [sys.executable, "matchup.py", "--batch", str(csv_path),
                 "--date", iso, "--mix-shift-alpha", "0", "--force"],
                "matchup",
            )
            row["matchup_status"] = "ok" if ok else "failed"
            if not ok:
                row["error"] = err
                return row

    # Step 3: postgame.py
    if skip_postgame:
        row["postgame_status"] = "skipped (--skip-postgame)"
        return row

    if dry_run:
        print(f"[backfill] {iso}: would run postgame.py --date {iso}")
        row["postgame_status"] = "skipped (dry-run)"
        return row

    # If skip_existing and this date is already graded, skip postgame too.
    if skip_existing and iso in _graded_dates_cache:
        row["postgame_status"] = "skipped (existing)"
        return row

    ok, err = _run(
        [sys.executable, "postgame.py", "--date", iso],
        "postgame",
    )
    row["postgame_status"] = "ok" if ok else "failed"
    if not ok:
        row["error"] = err
    return row


_graded_dates_cache: set[str] = set()


def main(argv: list[str] | None = None) -> int:
    yesterday = dateclass.today() - timedelta(days=1)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default=SEASON_START.isoformat(),
                    help=f"start date (default {SEASON_START.isoformat()})")
    ap.add_argument("--end", default=yesterday.isoformat(),
                    help=f"end date inclusive (default yesterday: "
                         f"{yesterday.isoformat()})")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip dates that already have slate.json + graded "
                         "rows in hitter_results.parquet")
    ap.add_argument("--skip-postgame", action="store_true",
                    help="only build slate.json baselines; do not grade")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the planned actions and exit without running")
    args = ap.parse_args(argv)

    try:
        start = dateclass.fromisoformat(args.start)
        end = dateclass.fromisoformat(args.end)
    except ValueError as e:
        print(f"[backfill] invalid date: {e}", file=sys.stderr)
        return 2

    if start > end:
        print(f"[backfill] start ({start}) > end ({end}); nothing to do")
        return 0

    global _graded_dates_cache
    _graded_dates_cache = _graded_dates()
    if args.skip_existing:
        print(f"[backfill] {len(_graded_dates_cache)} dates already graded "
              f"(skip-existing will no-op those)")

    dates = list(_date_range(start, end))
    print(f"[backfill] processing {len(dates)} dates: {start} .. {end}")
    print(f"[backfill] manifest: {MANIFEST_PATH}")

    results: list[dict] = []
    for d in dates:
        print(f"\n[backfill] === {d.isoformat()} ===")
        row = process_date(
            d,
            skip_existing=args.skip_existing,
            skip_postgame=args.skip_postgame,
            dry_run=args.dry_run,
        )
        results.append(row)
        if not args.dry_run:
            _append_manifest(row)
        # Refresh graded-dates cache after each postgame run so subsequent
        # dates with --skip-existing see the update.
        if not args.dry_run and row.get("postgame_status") == "ok":
            _graded_dates_cache.add(d.isoformat())

    # Summary
    print("\n[backfill] === summary ===")
    ok_count = sum(
        1 for r in results
        if r["matchup_status"] in {"ok", "skipped (existing)"}
        and r["postgame_status"] in {"ok", "skipped (existing)",
                                     "skipped (--skip-postgame)"}
    )
    fail_count = sum(
        1 for r in results
        if r["matchup_status"] == "failed" or r["postgame_status"] == "failed"
    )
    no_games = sum(1 for r in results if r["matchup_status"] == "no games")
    print(f"  OK:        {ok_count}")
    print(f"  FAILED:    {fail_count}")
    print(f"  NO GAMES:  {no_games}")
    if fail_count:
        print("\nfailures:")
        for r in results:
            if r["matchup_status"] == "failed" or r["postgame_status"] == "failed":
                print(f"  {r['date']}: matchup={r['matchup_status']} "
                      f"postgame={r['postgame_status']}")
                if r["error"]:
                    print(f"    {r['error'].splitlines()[0]}")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
