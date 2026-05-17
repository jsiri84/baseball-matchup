# AGENTS.md

## Cursor Cloud specific instructions

This is a pure-Python CLI toolkit for MLB batter-vs-pitcher matchup analysis. No databases, Docker, or background services are required. All scripts talk to public MLB web APIs over HTTPS.

### Running the application

See `README.md` "Daily workflow" and "Quick start" sections for full CLI usage.

Quick smoke-test (generates one HTML report under `reports/<date>/`):

```bash
python3 matchup.py --batter "Yordan Alvarez" --pitcher "Paul Skenes"
```

The first invocation for a player in the current season fetches live Statcast data from Baseball Savant (requires internet). Subsequent runs use the Parquet cache under `data/<year>/`.

### Key scripts

| Script | Purpose |
|--------|---------|
| `matchup.py` | Core matchup report generator (single, lineup, or batch mode) |
| `fetch_lineups.py` | Pulls today's MLB.com lineups → writes `matchups/matchups_<date>_<HHMMSS>.csv`. Merges with the most-recent prior same-day file: preserves confirmed lineups, upgrades projected → confirmed, reuses the prior file (no new write) when nothing changed. |
| `roundup.py` | Builds top-50/bottom-50 hitter roundup from sidecar JSON |
| `build_site.py` | Generates static HTML navigation (per-day hub + root index/archive) |
| `daily.py` | Orchestrates the full pipeline (fetch → matchup → roundup → site → git commit) |
| `batter.py` / `pitcher.py` | Statcast data pull + standalone player summary helpers |

### Linting

No formal linter config exists in the repo. Use `flake8 --select=E9,F63,F7,F82 *.py` for critical error checks, or `python3 -m py_compile <file>` for syntax validation.

### Testing

No automated test suite exists. Validate changes by running `matchup.py` in single-matchup mode and inspecting the generated HTML report under `reports/<date>/`.

### Smoke tests — ALWAYS use --out-dir sandbox

**Critical**: `matchup.py --batch` writes to `reports/<date>/` and overwrites the canonical `_data/slate.json` (a working-tree-only file not stored in git). Running a smoke test with a real date against a real `reports/<date>/` will silently destroy that day's slate sidecar and break postgame grading for that date.

For ANY smoke test that uses `--batch` with a real date, always pass `--out-dir sandbox` (or set `BASEBALL_BOT_REPORT_ROOT=sandbox`). Output goes to `sandbox/<date>/` which is `.gitignored` and never touched by the real pipeline.

```bash
# RIGHT -- smoke testing AZ@COL against the real 2026-05-16 date
python matchup.py --batch _smoke_az_col.csv --date 2026-05-16 --out-dir sandbox

# WRONG -- this will clobber reports/2026-05-16/_data/slate.json
python matchup.py --batch _smoke_az_col.csv --date 2026-05-16
```

A safety guard in `matchup.py` refuses to overwrite a real `reports/<date>/` slate.json when the batch input has significantly fewer games than the existing slate, and prints the fix. `--force` bypasses it for legitimate small-slate replays.

### Dependencies

`pip install -r requirements.txt` plus `tabulate` (needed by pandas `to_markdown()`). The `tabulate` package is an unlisted transitive dependency.

### Gotchas

- The `pybaseball` player lookup table (`playerid_lookup`) prints "No identically matched names found!" warnings to stdout on first use per session — this is normal, not an error.
- Prior-season Parquet caches (`data/2024/`, `data/2025/`) are committed to the repo. Current-season data (`data/2026/`) is git-ignored and regenerated on demand.
- Reports output to `reports/` which is also git-ignored.
- Logs go to `logs/<timestamp>/` (git-ignored).
- `~/.local/bin` must be on PATH for user-installed CLI tools (flake8, tabulate, etc.).
