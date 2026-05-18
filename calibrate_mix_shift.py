"""Empirical alpha sweep for the scouting-adaptive pitch mix shift.

For each alpha in the grid, this script:

  1. Re-runs ``matchup.py`` with ``--mix-shift-alpha <alpha>`` against the
     backfilled matchups CSVs, writing slate.json into a per-alpha sandbox
     directory (``sandbox/alpha_sweep/alpha_<alpha>/<date>/_data/slate.json``).
     Sandbox output guarantees we never touch real ``reports/<date>/``.

  2. Re-grades each per-PA actual in ``data/accuracy/pa_results.parquet``
     against the new ``proj_dist`` (multinomial log-loss + Brier).

  3. Re-scores the per-hitter projections in
     ``data/accuracy/hitter_results.parquet`` (RMSE on proj_xwoba, pass-rate
     at the 30-pts threshold, extreme-decile pass-rate, K%/BB%/HR% RMSE).

Outputs ``sandbox/alpha_sweep/results.csv`` and ``index.html`` with a
metric-vs-alpha plot per row.

Skip ``matchup.py`` re-runs with ``--skip-matchup`` to reuse existing
sandbox slates (useful when iterating only on the metrics).

The historical actuals (pa_results, hitter_results) were graded under
alpha=0 and are independent of alpha; that's why we only need to re-run
the FORWARD projection per alpha.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
import time
import unicodedata
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).parent.resolve()
SANDBOX_ROOT = ROOT / "sandbox" / "alpha_sweep"
MATCHUPS_DIR = ROOT / "matchups"
ACCURACY_DIR = ROOT / "data" / "accuracy"

# Wider grid than the plan's [0..1] because the 0.4 smoke test only
# shifted xwOBA by ~1pt -- the response curve looks gentle until alpha
# gets bigger.  Includes 2.0 (heavier than plan) so the metrics can
# overshoot and we see the peak.  alpha=0.0 reuses the cached production
# slate (see read_reports_slate_idx) so the actual matchup.py re-runs
# are only the 4 non-baseline points.
DEFAULT_ALPHAS = (0.0, 0.25, 0.5, 1.0, 2.0)
REPORTS_DIR = ROOT / "reports"
OUTCOME_CLASSES = ("K", "BB", "HBP", "1B", "2B", "3B", "HR", "Out")
PASS_THRESHOLD_PTS = 30.0  # postgame "pass" threshold on outcome-axis xwOBA delta

_MATCHUPS_RE = re.compile(r"^matchups_(\d{4}-\d{2}-\d{2})_")


# ----------------------------- utility helpers -----------------------------

def _norm_name(name: str) -> str:
    """Lowercase / strip / ASCII-fold a player name for cross-source joins."""
    if not name:
        return ""
    s = unicodedata.normalize("NFKD", name)
    s = s.encode("ascii", "ignore").decode("ascii")
    return s.lower().strip()


def discover_backfilled_dates(start: str | None,
                              end: str | None) -> list[str]:
    """Return sorted ISO-date strings that have BOTH a matchups CSV AND
    cached actuals in pa_results.parquet (without actuals there is nothing
    to grade against)."""
    if not MATCHUPS_DIR.exists():
        return []
    dates: set[str] = set()
    for p in MATCHUPS_DIR.glob("matchups_*.csv"):
        m = _MATCHUPS_RE.match(p.name)
        if m:
            dates.add(m.group(1))

    # Filter to dates that have graded PAs (otherwise the sweep can't score them).
    pa_path = ACCURACY_DIR / "pa_results.parquet"
    if pa_path.exists():
        graded = set(
            pd.read_parquet(pa_path, columns=["date"])["date"].astype(str).unique()
        )
        dates &= graded

    if start:
        dates = {d for d in dates if d >= start}
    if end:
        dates = {d for d in dates if d <= end}
    return sorted(dates)


def latest_matchups_csv(date_iso: str) -> Path | None:
    """Return the most recent matchups CSV for ``date_iso``."""
    candidates = sorted(MATCHUPS_DIR.glob(f"matchups_{date_iso}_*.csv"))
    return candidates[-1] if candidates else None


# ----------------------------- matchup runner -----------------------------

def alpha_tag(alpha: float) -> str:
    """Filesystem-safe tag for an alpha value, e.g. 0.4 -> 'alpha_0p40'."""
    return f"alpha_{alpha:.2f}".replace(".", "p")


def run_matchup_for_alpha(alpha: float, dates: Iterable[str],
                          skip_existing: bool = True) -> tuple[int, int, list[str]]:
    """Run matchup.py for each date under the alpha's sandbox subdir.

    For ``alpha == 0`` we short-circuit and reuse the cached production
    slate at ``reports/<date>/_data/slate.json`` -- it's bit-identical
    (verified in the impl_regression_check todo) and saves ~30 minutes
    of redundant compute.  The slate path will resolve to the reports
    tree via ``slate_path_for_alpha`` rather than the sandbox.

    Returns ``(ok_count, fail_count, failed_dates)``.
    """
    if alpha == 0.0:
        ok = sum(1 for d in dates
                 if (REPORTS_DIR / d / "_data" / "slate.json").exists())
        missing = [d for d in dates
                   if not (REPORTS_DIR / d / "_data" / "slate.json").exists()]
        for d in missing:
            print(f"    {d}: MISSING production slate (alpha=0 baseline)")
        return (ok, len(missing), missing)

    sandbox_dir = SANDBOX_ROOT / alpha_tag(alpha)
    sandbox_dir.mkdir(parents=True, exist_ok=True)
    ok = 0
    failed: list[str] = []
    for d in dates:
        slate_path = sandbox_dir / d / "_data" / "slate.json"
        if skip_existing and slate_path.exists():
            ok += 1
            continue
        csv_path = latest_matchups_csv(d)
        if csv_path is None:
            failed.append(f"{d} (no matchups CSV)")
            continue
        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, "matchup.py",
             "--batch", str(csv_path),
             "--date", d,
             "--mix-shift-alpha", str(alpha),
             "--out-dir", str(sandbox_dir),
             "--slate-only",
             "--force"],
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",
        )
        dt = time.time() - t0
        if proc.returncode == 0:
            ok += 1
            print(f"    {d}: ok ({dt:.0f}s)")
        else:
            failed.append(d)
            tail = (proc.stdout + proc.stderr).splitlines()[-5:]
            print(f"    {d}: FAIL rc={proc.returncode}: {' | '.join(tail)}")
    return (ok, len(failed), failed)


# ----------------------------- index building -----------------------------

def build_proj_index(alpha: float, dates: Iterable[str],
                     hr_df: pd.DataFrame) -> tuple[
                         dict[tuple[str, int], list[float]],
                         dict[tuple[str, int], dict[str, float]]]:
    """Build ``(date, mlbam) -> proj_dist`` and ``-> {proj_xwoba, k_pct, ...}``
    indices by reading the alpha's sandbox slate.json files.

    mlbam isn't stored in slate.json summary_rows directly, so we look up
    by ``(date, team, norm_name)`` against the baseline hitter_results
    that postgame.py already resolved.
    """
    # Build name -> mlbam lookup from already-graded hitter_results.
    name_to_mlbam: dict[tuple[str, str, str], int] = {}
    for _, r in hr_df.iterrows():
        date_s = str(r["date"])
        team = str(r["team"])
        name_norm = _norm_name(str(r["name"]))
        key = (date_s, team, name_norm)
        try:
            mlbam = int(r["mlbam"])
        except (TypeError, ValueError):
            continue
        name_to_mlbam[key] = mlbam

    # alpha=0 reads the cached production slate (see run_matchup_for_alpha
    # short-circuit); all other alphas read from the per-alpha sandbox.
    if alpha == 0.0:
        slate_root = REPORTS_DIR
    else:
        slate_root = SANDBOX_ROOT / alpha_tag(alpha)
    dist_idx: dict[tuple[str, int], list[float]] = {}
    scalar_idx: dict[tuple[str, int], dict[str, float]] = {}
    for d in dates:
        slate_path = slate_root / d / "_data" / "slate.json"
        if not slate_path.exists():
            continue
        try:
            slate = json.loads(slate_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for game in slate.get("games", []):
            team = str(game.get("hitter_team", ""))
            for sr in game.get("summary_rows", []) or []:
                name = sr.get("name", "")
                mlbam = name_to_mlbam.get((d, team, _norm_name(name)))
                if mlbam is None:
                    continue
                pd_dict = sr.get("proj_dist") or {}
                dist_idx[(d, mlbam)] = [float(pd_dict.get(c, 0.0))
                                        for c in OUTCOME_CLASSES]
                scalar_idx[(d, mlbam)] = {
                    "proj_xwoba": float(sr.get("proj_xwoba", 0.0)),
                    "k_pct": float(sr.get("k_pct", 0.0)),
                    "bb_pct": float(sr.get("bb_pct", 0.0)),
                    "hr_pct": float(sr.get("hr_pct", 0.0)),
                    "hit_pct": float(sr.get("hit_pct", 0.0)),
                }
    return dist_idx, scalar_idx


# ----------------------------- metric scoring -----------------------------

def score_pa_metrics(pa_df: pd.DataFrame,
                     dist_idx: dict[tuple[str, int], list[float]]) -> dict:
    """Multinomial log-loss + Brier under the new proj_dist."""
    eps = 1e-12
    ll_sum = 0.0
    br_sum = 0.0
    n = 0
    missing = 0
    for _, r in pa_df.iterrows():
        try:
            mlbam = int(r["mlbam"])
        except (TypeError, ValueError):
            continue
        key = (str(r["date"]), mlbam)
        dist = dist_idx.get(key)
        if not dist:
            missing += 1
            continue
        ac = r["actual_class"]
        if ac not in OUTCOME_CLASSES:
            continue
        idx = OUTCOME_CLASSES.index(ac)
        p = max(eps, min(1.0 - eps, dist[idx]))
        ll_sum -= math.log(p)
        br_sum += sum(
            (dist[i] - (1.0 if c == ac else 0.0)) ** 2
            for i, c in enumerate(OUTCOME_CLASSES)
        )
        n += 1
    return {
        "n_pa": n,
        "missing_pa": missing,
        "logloss": ll_sum / n if n else float("nan"),
        "brier": br_sum / n if n else float("nan"),
    }


def score_hitter_metrics(hr_df: pd.DataFrame,
                         scalar_idx: dict[tuple[str, int], dict[str, float]],
                         ) -> dict:
    """RMSE + pass-rates on proj_xwoba, plus K%/BB%/HR% RMSE."""
    hr_df = hr_df.dropna(subset=["actual_xwoba"]).copy()
    if hr_df.empty:
        return {"n_hitters": 0}

    new_xwoba: list[float] = []
    new_k: list[float] = []
    new_bb: list[float] = []
    new_hr: list[float] = []
    for _, r in hr_df.iterrows():
        try:
            mlbam = int(r["mlbam"])
        except (TypeError, ValueError):
            new_xwoba.append(float("nan"))
            new_k.append(float("nan"))
            new_bb.append(float("nan"))
            new_hr.append(float("nan"))
            continue
        scalars = scalar_idx.get((str(r["date"]), mlbam))
        if scalars:
            new_xwoba.append(scalars["proj_xwoba"])
            new_k.append(scalars["k_pct"])
            new_bb.append(scalars["bb_pct"])
            new_hr.append(scalars["hr_pct"])
        else:
            new_xwoba.append(float("nan"))
            new_k.append(float("nan"))
            new_bb.append(float("nan"))
            new_hr.append(float("nan"))

    hr_df["proj_xwoba_new"] = new_xwoba
    hr_df["proj_k_pct_new"] = new_k
    hr_df["proj_bb_pct_new"] = new_bb
    hr_df["proj_hr_pct_new"] = new_hr

    sub = hr_df.dropna(subset=["proj_xwoba_new"])
    if sub.empty:
        return {"n_hitters": 0}

    delta_pts = (sub["proj_xwoba_new"] - sub["actual_xwoba"]) * 1000.0
    rmse = math.sqrt(float((delta_pts ** 2).mean()))
    pass_rate = float((delta_pts.abs() < PASS_THRESHOLD_PTS).mean())

    ext_mask = (sub["proj_xwoba_new"] >= 0.380) | (sub["proj_xwoba_new"] <= 0.260)
    ext = sub[ext_mask]
    ext_rmse = math.nan
    ext_pass = math.nan
    if not ext.empty:
        ext_delta = (ext["proj_xwoba_new"] - ext["actual_xwoba"]) * 1000.0
        ext_rmse = math.sqrt(float((ext_delta ** 2).mean()))
        ext_pass = float((ext_delta.abs() < PASS_THRESHOLD_PTS).mean())

    k_rmse = math.sqrt(float(((sub["proj_k_pct_new"] - sub["actual_k_pct"]) ** 2).mean()))
    bb_rmse = math.sqrt(float(((sub["proj_bb_pct_new"] - sub["actual_bb_pct"]) ** 2).mean()))
    hr_rmse = math.sqrt(float(((sub["proj_hr_pct_new"] - sub["actual_hr_pct"]) ** 2).mean()))

    return {
        "n_hitters": int(len(sub)),
        "rmse_xwoba_pts": rmse,
        "pass_rate_30pts": pass_rate,
        "n_extreme": int(len(ext)),
        "rmse_xwoba_extreme_pts": ext_rmse,
        "pass_rate_extreme": ext_pass,
        "rmse_k_pct": k_rmse,
        "rmse_bb_pct": bb_rmse,
        "rmse_hr_pct": hr_rmse,
    }


# ----------------------------- HTML chart -----------------------------

def _svg_line_chart(xs: list[float], ys: list[float], title: str,
                    width: int = 320, height: int = 180) -> str:
    """Tiny SVG line chart with axis labels.  Skips NaN points."""
    pts = [(x, y) for x, y in zip(xs, ys)
           if isinstance(y, (int, float)) and not math.isnan(y)]
    if not pts:
        return f"<div class='chart'><h4>{title}</h4>(no data)</div>"
    xs_, ys_ = zip(*pts)
    pad = 30
    x_lo, x_hi = min(xs_), max(xs_)
    y_lo, y_hi = min(ys_), max(ys_)
    if x_hi == x_lo:
        x_hi = x_lo + 1
    if y_hi == y_lo:
        y_hi = y_lo + 1e-6
    def sx(x): return pad + (x - x_lo) / (x_hi - x_lo) * (width - 2 * pad)
    def sy(y): return height - pad - (y - y_lo) / (y_hi - y_lo) * (height - 2 * pad)
    poly = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in pts)
    circles = "".join(
        f"<circle cx='{sx(x):.1f}' cy='{sy(y):.1f}' r='3' fill='#0a4'/>"
        for x, y in pts
    )
    # baseline = alpha=0 value (first point with x==0 if present)
    base_line = ""
    base = next((y for x, y in pts if x == 0.0), None)
    if base is not None:
        base_y = sy(base)
        base_line = (f"<line x1='{pad}' y1='{base_y:.1f}' x2='{width-pad}' "
                     f"y2='{base_y:.1f}' stroke='#aaa' stroke-dasharray='3 3'/>")
    return (
        f"<div class='chart'><h4>{title}</h4>"
        f"<svg width='{width}' height='{height}' xmlns='http://www.w3.org/2000/svg'>"
        f"<rect x='{pad}' y='{pad/2}' width='{width-2*pad}' "
        f"height='{height-pad*1.5}' fill='#fafafa' stroke='#ddd'/>"
        f"{base_line}"
        f"<polyline fill='none' stroke='#0a4' stroke-width='2' points='{poly}'/>"
        f"{circles}"
        f"<text x='{pad}' y='{height-8}' font-size='10'>α={x_lo:.2f}</text>"
        f"<text x='{width-pad-30}' y='{height-8}' font-size='10'>α={x_hi:.2f}</text>"
        f"<text x='4' y='{pad+10}' font-size='10'>{y_hi:.3g}</text>"
        f"<text x='4' y='{height-pad}' font-size='10'>{y_lo:.3g}</text>"
        f"</svg></div>"
    )


def write_results(results: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"
    fieldnames = list(results[0].keys()) if results else []
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)

    metric_specs = [
        ("logloss", "PA log-loss (lower=better)"),
        ("brier", "PA Brier (lower=better)"),
        ("rmse_xwoba_pts", "proj_xwoba RMSE [pts] (lower=better)"),
        ("pass_rate_30pts", "pass rate <30pts (higher=better)"),
        ("rmse_xwoba_extreme_pts", "extreme proj_xwoba RMSE [pts]"),
        ("pass_rate_extreme", "extreme pass rate (higher=better)"),
        ("rmse_k_pct", "K% RMSE (lower=better)"),
        ("rmse_bb_pct", "BB% RMSE"),
        ("rmse_hr_pct", "HR% RMSE"),
    ]
    alphas = [r["alpha"] for r in results]
    charts_html = "".join(
        _svg_line_chart(alphas, [r.get(metric, math.nan) for r in results], title)
        for metric, title in metric_specs
    )

    # Headline table
    rows_html = ""
    for r in results:
        rows_html += "<tr>"
        rows_html += f"<td>{r['alpha']:.2f}</td>"
        for k in ("n_pa", "logloss", "brier", "n_hitters", "rmse_xwoba_pts",
                  "pass_rate_30pts", "pass_rate_extreme",
                  "rmse_k_pct", "rmse_bb_pct", "rmse_hr_pct"):
            val = r.get(k, "")
            if isinstance(val, float):
                rows_html += f"<td>{val:.4f}</td>"
            else:
                rows_html += f"<td>{val}</td>"
        rows_html += "</tr>"

    html = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>α sweep</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif;
        margin: 24px; color: #222; }}
table {{ border-collapse: collapse; margin: 12px 0; }}
th, td {{ padding: 6px 10px; border: 1px solid #ddd; font-size: 13px; }}
th {{ background: #f4f4f4; }}
.chart {{ display: inline-block; margin: 8px 12px 8px 0;
          vertical-align: top; }}
.chart h4 {{ margin: 0 0 4px; font-size: 12px; color: #555; }}
small {{ color: #888; }}
</style></head><body>
<h2>Scouting-adaptive pitch mix &mdash; α sweep</h2>
<small>generated {pd.Timestamp.utcnow():%Y-%m-%d %H:%M UTC}</small>
<p>Dashed line = α=0 baseline (current production model).
Tracks all eight metrics across the alpha grid; pick the alpha that
improves the headline metrics (log-loss, RMSE, pass-rate) without
materially harming K%/BB%/HR% RMSE.</p>
<table><tr>
<th>α</th><th>n_pa</th><th>logloss</th><th>brier</th>
<th>n_hitters</th><th>RMSE xwOBA</th>
<th>pass &lt;30pts</th><th>extreme pass</th>
<th>K% RMSE</th><th>BB% RMSE</th><th>HR% RMSE</th>
</tr>
{rows_html}
</table>
<div>{charts_html}</div>
</body></html>
"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")
    print(f"\n[calibrate] wrote {csv_path}")
    print(f"[calibrate] wrote {out_dir / 'index.html'}")


# ----------------------------- driver -----------------------------

def main(argv: list[str] | None = None) -> int:
    global SANDBOX_ROOT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--alphas", default=",".join(f"{a}" for a in DEFAULT_ALPHAS),
                    help="comma-separated alpha grid "
                         f"(default {','.join(f'{a}' for a in DEFAULT_ALPHAS)})")
    ap.add_argument("--start", default=None, help="earliest date (YYYY-MM-DD)")
    ap.add_argument("--end", default=None, help="latest date (YYYY-MM-DD)")
    ap.add_argument("--skip-matchup", action="store_true",
                    help="reuse existing sandbox slates; don't re-run matchup.py")
    ap.add_argument("--out-dir", default=str(SANDBOX_ROOT),
                    help=f"sandbox root (default {SANDBOX_ROOT})")
    args = ap.parse_args(argv)

    SANDBOX_ROOT = Path(args.out_dir).resolve()
    SANDBOX_ROOT.mkdir(parents=True, exist_ok=True)

    try:
        alphas = sorted({float(a.strip()) for a in args.alphas.split(",") if a.strip()})
    except ValueError as e:
        print(f"[calibrate] invalid --alphas: {e}", file=sys.stderr)
        return 2

    dates = discover_backfilled_dates(args.start, args.end)
    if not dates:
        print("[calibrate] no backfilled dates found "
              "(need both matchups CSV and graded pa_results)", file=sys.stderr)
        return 1
    print(f"[calibrate] sweep alphas={alphas} on {len(dates)} dates "
          f"({dates[0]} .. {dates[-1]})")

    # Load actuals once.
    hr_df = pd.read_parquet(ACCURACY_DIR / "hitter_results.parquet")
    pa_df = pd.read_parquet(ACCURACY_DIR / "pa_results.parquet")
    hr_df = hr_df[hr_df["date"].astype(str).isin(dates)].copy()
    pa_df = pa_df[pa_df["date"].astype(str).isin(dates)].copy()
    print(f"[calibrate] actuals scope: {len(hr_df)} hitter-rows, {len(pa_df)} PAs")

    results: list[dict] = []
    for alpha in alphas:
        print(f"\n[calibrate] === alpha={alpha:.2f} ===")
        if not args.skip_matchup:
            t0 = time.time()
            ok, failc, _ = run_matchup_for_alpha(alpha, dates, skip_existing=True)
            print(f"[calibrate]   matchup runs: {ok} ok / {failc} fail "
                  f"({time.time() - t0:.0f}s)")
            if ok == 0:
                print(f"[calibrate]   no successful runs at alpha={alpha}; skipping")
                continue
        else:
            print(f"[calibrate]   reusing existing sandbox at {SANDBOX_ROOT / alpha_tag(alpha)}")

        dist_idx, scalar_idx = build_proj_index(alpha, dates, hr_df)
        print(f"[calibrate]   indexed {len(dist_idx)} (date,mlbam) projections")
        pa_metrics = score_pa_metrics(pa_df, dist_idx)
        hit_metrics = score_hitter_metrics(hr_df, scalar_idx)
        row = {"alpha": float(alpha), **pa_metrics, **hit_metrics}
        results.append(row)
        print(f"[calibrate]   alpha={alpha:.2f}: "
              f"logloss={pa_metrics.get('logloss'):.4f} "
              f"brier={pa_metrics.get('brier'):.4f} "
              f"xwOBA RMSE={hit_metrics.get('rmse_xwoba_pts', math.nan):.2f}pts "
              f"pass={hit_metrics.get('pass_rate_30pts', math.nan):.3f}")

    if not results:
        print("[calibrate] no successful alpha runs", file=sys.stderr)
        return 1
    write_results(results, SANDBOX_ROOT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
