#!/usr/bin/env python3
"""Daily run:

1. fetch today's lineups (with projected-lineup fallback for missing teams)
2. generate one matchup report per posted lineup
3. commit + push any new prior-season cache pulls (handled by matchup.py
   --commit-cache, which appends to data/<year>/ folders for 2024/2025)
4. build the day-level top-50 / bottom-50 hitter roundup reports from the
   sidecars matchup.py just dropped under reports/<date>/_data/
4b. build the navigation site (per-day reports/<date>/index.html plus the
    root index.html / archive.html that GitHub Pages serves)
5. commit + push today's reports/<date>/ folder (force-added because
   reports/ is .gitignored) plus the root index.html / archive.html

Usage:
    python daily.py                # full run
    python daily.py --no-push      # commit locally, do not push
    python daily.py --no-commit    # skip both git commits (just generate)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

from log_setup import setup_logging

ROOT = Path(__file__).parent
PY = sys.executable


def run(cmd: list[str], label: str, allow_fail: bool = False) -> int:
    """Run a subprocess in ROOT, streaming output.

    If allow_fail is True, log the failure but do not exit; returns the exit
    code so the caller can decide. Default is exit-on-failure.
    """
    print(f"\n=== {label} ===")
    print(f"$ {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        msg = f"[daily] step failed: {label} (exit {r.returncode})"
        if allow_fail:
            print(msg)
            print("[daily] continuing (step marked allow_fail).")
            return r.returncode
        sys.exit(msg)
    return 0


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    if check and r.returncode != 0:
        sys.exit(f"[daily] git {' '.join(args)} failed: {r.stderr.strip()}")
    return r


def commit_reports(today: str, yesterday: str, push: bool) -> None:
    today_dir = ROOT / "reports" / today
    if not today_dir.exists():
        print(f"\n[daily] no reports directory at reports/{today}; nothing to commit")
        return

    # Today's matchup outputs.
    today_rel = f"reports/{today}"
    for pattern in ("*.html", "*.md"):
        git("add", "-f", "--", f"{today_rel}/{pattern}", check=False)

    # Yesterday's postgame outputs (force-add since reports/ is gitignored).
    yesterday_rel = f"reports/{yesterday}"
    if (ROOT / yesterday_rel).exists():
        git("add", "-f", "--", f"{yesterday_rel}/postgame/", check=False)
        # Legacy: catch in-place postgame files from older runs.
        git("add", "-f", "--", f"{yesterday_rel}/postgame_*.html", check=False)
        # Updated per-day hub for yesterday (build_site.py was rerun).
        git("add", "-f", "--", f"{yesterday_rel}/index.html", check=False)

    # Rolling accuracy dashboard.
    accuracy_rel = "reports/accuracy"
    if (ROOT / accuracy_rel).exists():
        git("add", "-f", "--", f"{accuracy_rel}/index.html", check=False)
        git("add", "-f", "--", f"{accuracy_rel}/*.html", check=False)

    # Root navigation pages (not under reports/, so not gitignored).
    for root_page in ("index.html", "archive.html"):
        if (ROOT / root_page).exists():
            git("add", "--", root_page, check=False)

    pathspecs = [today_rel, yesterday_rel, accuracy_rel,
                 "index.html", "archive.html"]
    status = git("status", "--porcelain", "--", *pathspecs)
    if not status.stdout.strip():
        print(f"\n[daily] no new/changed files; nothing to commit")
        return

    n = sum(1 for _ in status.stdout.strip().splitlines())
    msg = f"reports: daily run {today} ({n} file{'s' if n != 1 else ''})"
    git("commit", "-m", msg)
    print(f"\n[daily] committed {n} file(s): {msg!r}")

    if push:
        pr = git("push", check=False)
        if pr.returncode == 0:
            print("[daily] pushed reports to origin")
        else:
            print(f"[daily] git push failed (commit is local): {pr.stderr.strip()}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-push", action="store_true",
                    help="commit locally but skip git push (for both cache and reports)")
    ap.add_argument("--no-commit", action="store_true",
                    help="generate reports but skip all git operations")
    ap.add_argument("--workers", type=int, default=None,
                    help="forwarded to matchup.py --workers; parallel report "
                         "generation thread count (default: matchup.py's own default)")
    ap.add_argument("--no-postgame", action="store_true",
                    help="skip yesterday's postgame + rolling accuracy steps "
                         "(useful for repro runs that don't have completed games)")
    ap.add_argument("--accuracy-window", default="30",
                    help="trailing window in days for accuracy.py (default 30, "
                         "or 'all')")
    args = ap.parse_args()

    log_path = setup_logging("daily")
    today_dt = datetime.now()
    today = today_dt.strftime("%Y-%m-%d")
    yesterday = (today_dt - timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"[daily] run for {today}")
    print(f"[daily] logging to {log_path.relative_to(ROOT)}")

    # 1. fetch lineups -> matchups_YYYY-MM-DD.csv
    fetch_cmd = [PY, "fetch_lineups.py"]
    opener_bulk_path = ROOT / "opener_bulk.csv"
    if opener_bulk_path.exists():
        fetch_cmd.extend(["--opener-bulk-file", str(opener_bulk_path)])
    run(fetch_cmd, "fetch lineups")

    csv_path = ROOT / f"matchups_{today}.csv"
    if not csv_path.exists():
        sys.exit(f"[daily] expected {csv_path.name} not found after fetch_lineups")

    # 2. + 3. run matchups; --commit-cache handles prior-season parquet commit/push
    matchup_cmd = [PY, "matchup.py", "--batch", csv_path.name]
    if args.workers is not None:
        matchup_cmd.extend(["--workers", str(args.workers)])
    if not args.no_commit:
        matchup_cmd.append("--commit-cache")
        if args.no_push:
            matchup_cmd.append("--no-push")
    run(matchup_cmd, "generate matchup reports")

    # 4. day-level roundup (top-50 / bottom-50 hitters by projected xwOBA)
    run([PY, "roundup.py", "--date", today], "build top/bottom-50 roundup")

    # 4b. postgame for yesterday + rolling accuracy dashboard. Both are
    # network-bound (postgame hits StatsAPI + pybaseball Statcast) and may
    # fail on transient errors -- mark allow_fail so the daily run still
    # produces today's outputs even if yesterday's postgame can't complete.
    if not args.no_postgame:
        pg_rc = run([PY, "postgame.py", "--date", yesterday],
                    f"postgame for {yesterday}", allow_fail=True)
        if pg_rc == 0:
            run([PY, "accuracy.py", "--window", args.accuracy_window],
                "rolling accuracy dashboard", allow_fail=True)
        else:
            print("[daily] postgame failed; skipping accuracy rebuild")

    # 4c. site navigation: rebuild yesterday's hub (to surface postgame links)
    # and today's hub + root index.html + archive.html.
    if (ROOT / "reports" / yesterday).exists() and not args.no_postgame:
        run([PY, "build_site.py", "--date", yesterday],
            f"rebuild yesterday hub ({yesterday}) for postgame links",
            allow_fail=True)
    run([PY, "build_site.py", "--date", today],
        "build navigation site (hub + root index)")

    # 5. commit today's reports
    if args.no_commit:
        print("\n[daily] --no-commit set; skipping reports commit")
    else:
        commit_reports(today, yesterday, push=not args.no_push)

    print("\n[daily] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
