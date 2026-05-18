# Historical projections archive

Frozen snapshots of pregame model inputs/outputs for backtesting.  The
live `reports/` and `matchups/` directories are gitignored (they churn
daily as lineups firm up), so without this archive a fresh clone has no
way to reproduce or backtest historical projections.

Build / refresh with:

```bash
python archive_historical.py
```

Idempotent -- re-runs just overwrite each per-date file with whatever is
currently in `reports/<date>/_data/slate.json` and the latest
`matchups/matchups_<date>_*.csv`.

## Layout

```
data/historical/
├── README.md                       this file
├── slates/
│   └── slate_<YYYY-MM-DD>.json     verbatim copy of reports/<d>/_data/slate.json
├── matchups/
│   └── matchups_<YYYY-MM-DD>.csv   verbatim copy of the latest matchups
│                                   CSV (canonicalized to UTF-8)
└── projections.parquet             flat row-per-(date, mlbam) table with
                                    every summary_rows field expanded
```

A snapshot date qualifies if it has BOTH a `slate.json` AND graded
actuals in `data/accuracy/hitter_results.parquet` -- otherwise it can't
be scored.

## projections.parquet schema

One row per (`date`, `mlbam`).  Schema is stable across runs so backtest
notebooks can pin the columns they expect.

| Column | Type | Source |
|---|---|---|
| `date` | str (YYYY-MM-DD) | slate.json `date` |
| `mlbam` | int | recovered via (date, hitter_team, normalized_name) join against `hitter_results.parquet` |
| `name`, `hitter_team`, `matchup_key`, `pitcher_name`, `pitcher_id`, `p_throws`, `stand`, `spot`, `projected`, `pa_per_batter` | meta | slate `summary_rows` + `pitcher_meta` |
| `proj_xwoba`, `proj_xba`, `proj_xslg`, `proj_xwoba_raw`, `bbtype_adj_pts`, `park_pf`, `park_pts`, `form_d14_*`, `delta_pts` | float | scalar projections |
| `proj_k_pct`, `proj_bb_pct`, `proj_hr_pct`, `proj_hit_pct`, `proj_ob_pct`, `proj_hardhit_pct`, `proj_whiff_pct`, `proj_xwoba_on_contact` | float | rate projections |
| `proj_dist_K`, `proj_dist_BB`, `proj_dist_HBP`, `proj_dist_1B`, `proj_dist_2B`, `proj_dist_3B`, `proj_dist_HR`, `proj_dist_Out` | float | per-PA 8-bucket outcome distribution (sums to 1) |
| `best_pitch`, `worst_pitch`, `verdict_label` | str | qualitative tags |
| `model_alpha` | float | `PITCHER_MIX_SHIFT_ALPHA` value the snapshot was built under (0.0 for current production) |
| `generated_at` | str (ISO) | when `matchup.py` ran |

## Backtesting workflow

The archive separates the IMMUTABLE pregame ground truth (matchups +
projections) from the GRADED actuals.  A backtest takes a new model,
re-projects, and re-scores:

```python
import pandas as pd

proj = pd.read_parquet("data/historical/projections.parquet")
acts = pd.read_parquet("data/accuracy/hitter_results.parquet")
pas  = pd.read_parquet("data/accuracy/pa_results.parquet")

# A) headline xwOBA RMSE per date
joined = proj.merge(
    acts[["date", "mlbam", "actual_xwoba", "pa", "actual_k_pct",
          "actual_bb_pct", "actual_hr_pct"]],
    on=["date", "mlbam"], how="inner",
)
joined["delta_pts"] = (joined["proj_xwoba"] - joined["actual_xwoba"]) * 1000
print(joined.groupby("date")["delta_pts"].apply(lambda s: (s**2).mean()**0.5))

# B) PA-level log-loss replay
# pas has actual_class + already-graded model_logloss for the baseline;
# replace proj_dist with a new model's distribution and recompute.
```

### Reproducing a slate from scratch

For a deeper backtest where you want to change the model itself (not
just re-score), drive `matchup.py` off the archived matchups CSV:

```bash
python matchup.py \
    --batch data/historical/matchups/matchups_2026-05-15.csv \
    --date 2026-05-15 \
    --out-dir sandbox/backtest_<your_change> \
    --slate-only \
    --force \
    --mix-shift-alpha <whatever>
```

Sandbox output keeps the experiment from clobbering the archive.  The
new `sandbox/backtest_*/2026-05-15/_data/slate.json` can be loaded the
same way as `data/historical/slates/slate_2026-05-15.json` for
comparison.

### Alpha-sweep precedent

`calibrate_mix_shift.py` is a fully-worked example of this workflow: it
takes the same matchups CSVs, runs `matchup.py --slate-only` with
varying `--mix-shift-alpha`, and re-grades against `pa_results.parquet`.
Results live in `data/accuracy/calibration/`.

## Notes

* `model_alpha=0.0` is current production.  Tag rows from sweeps with a
  non-zero alpha by passing `--model-alpha X` when archiving so you can
  keep multiple snapshots side-by-side.
* Rows whose `mlbam` couldn't be resolved (typically: a name that didn't
  appear in `hitter_results.parquet` for that date because the player
  didn't actually start) are dropped from `projections.parquet`.  The
  raw `slates/slate_<date>.json` still has them.
* If you regenerate `reports/<date>/_data/slate.json` (e.g. mid-day
  lineup refresh), re-run `archive_historical.py` to refresh the
  snapshot to match.
