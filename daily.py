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
from datetime import datetime
from pathlib import Path

from log_setup import setup_logging

ROOT = Path(__file__).parent
PY = sys.executable


def run(cmd: list[str], label: str) -> None:
    """Run a subprocess in ROOT, streaming output. Exit on failure."""
    print(f"\n=== {label} ===")
    print(f"$ {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        sys.exit(f"\n[daily] step failed: {label} (exit {r.returncode})")


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    if check and r.returncode != 0:
        sys.exit(f"[daily] git {' '.join(args)} failed: {r.stderr.strip()}")
    return r


def commit_reports(today: str, push: bool) -> None:
    reports_dir = ROOT / "reports" / today
    if not reports_dir.exists():
        print(f"\n[daily] no reports directory at reports/{today}; nothing to commit")
        return

    rel = f"reports/{today}"
    # reports/ is gitignored, so force-add. Once added, future modifications
    # are tracked normally even though the ignore pattern remains. Force-add
    # only the .html / .md outputs (skip the _data/ sidecar JSONs that the
    # roundup consumes -- they're regenerable and noisy). The per-day
    # index.html written by build_site.py is included by the *.html glob.
    for pattern in ("*.html", "*.md"):
        git("add", "-f", "--", f"{rel}/{pattern}", check=False)

    # Root navigation pages (not under reports/, so not gitignored).
    for root_page in ("index.html", "archive.html"):
        if (ROOT / root_page).exists():
            git("add", "--", root_page, check=False)

    status = git("status", "--porcelain", "--", rel, "index.html", "archive.html")
    if not status.stdout.strip():
        print(f"\n[daily] no new/changed files under {rel} or root index pages; nothing to commit")
        return

    n = sum(1 for _ in status.stdout.strip().splitlines())
    msg = f"reports: daily run {today} ({n} file{'s' if n != 1 else ''})"
    git("commit", "-m", msg)
    print(f"\n[daily] committed {n} report file(s): {msg!r}")

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
    args = ap.parse_args()

    log_path = setup_logging("daily")
    today = datetime.now().strftime("%Y-%m-%d")
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

    # 4b. site navigation: today's hub + root index.html + archive.html
    run([PY, "build_site.py", "--date", today], "build navigation site (hub + root index)")

    # 5. commit today's reports
    if args.no_commit:
        print("\n[daily] --no-commit set; skipping reports commit")
    else:
        commit_reports(today, push=not args.no_push)

    print("\n[daily] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
