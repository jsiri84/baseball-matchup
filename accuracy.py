#!/usr/bin/env python3
"""Rolling multi-day projection accuracy dashboard.

Reads the append-only stores written by ``postgame.py``:

  - ``data/accuracy/hitter_results.parquet`` (one row per hitter-game)
  - ``data/accuracy/pa_results.parquet``     (one row per scored PA)

Computes calibration tables, per-PA log-loss vs naive baselines,
hitter-grain xwOBA RMSE/MAE, and per-slate Spearman rank correlation.

Single-game proj-vs-actual is dominated by outcome variance; this report
aggregates across a trailing window of days to surface real model skill.

Usage::

    python accuracy.py                       # default 30-day window
    python accuracy.py --window 60
    python accuracy.py --since 2026-04-01
    python accuracy.py --window all          # full history
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from datetime import date as date_cls, timedelta
from pathlib import Path

import pandas as pd

from matchup import (
    LG_XWOBA, LG_XBA, LG_K_PCT, LG_BB_PCT, LG_HARD_HIT,
    _HTML_CSS, _h, _td, edge_class, fmt3,
)
from log_setup import setup_logging
from sortable import sortable_html

ROOT = Path(__file__).parent
ACCURACY_DIR = ROOT / "data" / "accuracy"
HITTER_PARQUET = ACCURACY_DIR / "hitter_results.parquet"
PA_PARQUET = ACCURACY_DIR / "pa_results.parquet"

OUT_DIR = ROOT / "reports" / "accuracy"


# ---------- data loading --------------------------------------------------

def _load_window(window: str | int, since: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the two parquets, filtered by trailing window (days) or --since cutoff."""
    if not HITTER_PARQUET.exists():
        sys.exit(f"[accuracy] missing {HITTER_PARQUET} - run postgame.py first.")
    hitter = pd.read_parquet(HITTER_PARQUET)
    if PA_PARQUET.exists():
        pa = pd.read_parquet(PA_PARQUET)
    else:
        pa = pd.DataFrame()

    hitter["date"] = pd.to_datetime(hitter["date"]).dt.strftime("%Y-%m-%d")
    if not pa.empty:
        pa["date"] = pd.to_datetime(pa["date"]).dt.strftime("%Y-%m-%d")

    if since:
        hitter = hitter[hitter["date"] >= since]
        if not pa.empty:
            pa = pa[pa["date"] >= since]
    elif isinstance(window, int) and window > 0:
        cutoff = (date_cls.today() - timedelta(days=window)).isoformat()
        hitter = hitter[hitter["date"] >= cutoff]
        if not pa.empty:
            pa = pa[pa["date"] >= cutoff]
    # "all" => no filter
    return hitter.reset_index(drop=True), pa.reset_index(drop=True)


# ---------- metric helpers -------------------------------------------------

def _wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% CI for a proportion."""
    if n <= 0 or math.isnan(p):
        return float("nan"), float("nan")
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _sem(x: pd.Series) -> float:
    """Standard error of the mean (NaN-safe)."""
    x = x.dropna()
    if len(x) < 2:
        return float("nan")
    return float(x.std(ddof=1) / math.sqrt(len(x)))


@dataclass
class CalibrationRow:
    label: str
    n: int
    mean_proj: float
    mean_actual: float
    delta_pts: float  # (actual - proj) * 1000 for rates/xwOBA-like quantities
    ci_low: float
    ci_high: float


def _bin_xwoba(v: float) -> str | None:
    if v is None or math.isnan(v):
        return None
    if v < 0.250: return "<.250"
    if v < 0.275: return ".250-.274"
    if v < 0.300: return ".275-.299"
    if v < 0.325: return ".300-.324"
    if v < 0.350: return ".325-.349"
    if v < 0.375: return ".350-.374"
    if v < 0.400: return ".375-.399"
    return ">=.400"


def _bin_pct(v: float, edges: tuple[float, ...]) -> str | None:
    if v is None or math.isnan(v):
        return None
    prev = 0.0
    for e in edges:
        if v < e:
            return f"{prev*100:.0f}-{e*100:.0f}%"
        prev = e
    return f">={prev*100:.0f}%"


def _calibration_rows(df: pd.DataFrame, proj_col: str, actual_col: str,
                      bin_fn, *, is_rate: bool, min_n_pa: int = 1) -> list[CalibrationRow]:
    """One CalibrationRow per occupied bin.

    is_rate=True implies the value is a proportion in [0,1] — CI is Wilson on
    the actual mean weighted by count. is_rate=False implies xwOBA-like — CI
    is mean +/- 1.96 SEM.
    """
    sub = df.dropna(subset=[proj_col, actual_col]).copy()
    if "pa" in sub.columns:
        sub = sub[sub["pa"].fillna(0) >= min_n_pa]
    if sub.empty:
        return []
    sub["__bin"] = sub[proj_col].map(bin_fn)
    rows: list[CalibrationRow] = []
    for label, group in sub.groupby("__bin", dropna=True):
        if group.empty:
            continue
        n = int(len(group))
        mean_proj = float(group[proj_col].mean())
        mean_actual = float(group[actual_col].mean())
        delta_pts = (mean_actual - mean_proj) * 1000.0
        if is_rate:
            lo, hi = _wilson_ci(mean_actual, n)
        else:
            sem = _sem(group[actual_col])
            if math.isnan(sem):
                lo = hi = float("nan")
            else:
                lo = mean_actual - 1.96 * sem
                hi = mean_actual + 1.96 * sem
        rows.append(CalibrationRow(label=label, n=n, mean_proj=mean_proj,
                                    mean_actual=mean_actual, delta_pts=delta_pts,
                                    ci_low=lo, ci_high=hi))
    # Sort by bin label - use the underlying numeric (mean_proj) for deterministic order.
    rows.sort(key=lambda r: r.mean_proj)
    return rows


def _logloss_table(pa: pd.DataFrame) -> list[dict]:
    """Per-PA log-loss + Brier for model vs league prior, with skill score."""
    if pa.empty:
        return []
    n = int(len(pa))
    m_ll = float(pa["model_logloss"].mean())
    m_br = float(pa["model_brier"].mean())
    lg_ll = float(pa["lg_logloss"].mean())
    lg_br = float(pa["lg_brier"].mean())
    skill_ll = 1.0 - (m_ll / lg_ll) if lg_ll else float("nan")
    skill_br = 1.0 - (m_br / lg_br) if lg_br else float("nan")
    return [
        {"name": "Model", "n_pa": n,
         "logloss": m_ll, "brier": m_br,
         "skill_ll": skill_ll, "skill_br": skill_br},
        {"name": "League prior", "n_pa": n,
         "logloss": lg_ll, "brier": lg_br,
         "skill_ll": 0.0, "skill_br": 0.0},
    ]


def _hitter_accuracy(df: pd.DataFrame, proj_col: str, actual_col: str,
                     min_pa: int = 1) -> dict:
    """RMSE / MAE / bias on (proj, actual) where both are non-null."""
    sub = df.dropna(subset=[proj_col, actual_col])
    if "pa" in sub.columns:
        sub = sub[sub["pa"].fillna(0) >= min_pa]
    if sub.empty:
        return {"n": 0, "rmse": float("nan"), "mae": float("nan"),
                "bias": float("nan")}
    err = sub[actual_col] - sub[proj_col]
    return {
        "n": int(len(sub)),
        "rmse": float((err ** 2).mean() ** 0.5),
        "mae": float(err.abs().mean()),
        "bias": float(err.mean()),
    }


def _pa_bucket_rmse(df: pd.DataFrame, proj_col: str, actual_col: str) -> list[dict]:
    """RMSE/MAE/bias of headline xwOBA broken out by PA count buckets.

    Single-PA games are 80% of single-game variance; bucketing exposes how much
    of the headline RMSE is genuine model miss vs sample-size noise. Buckets
    drop from `>=4` once a hitter has had a full game's worth of PAs.
    """
    sub = df.dropna(subset=[proj_col, actual_col, "pa"]).copy()
    if sub.empty:
        return []
    def _bucket(pa: float) -> str:
        if pa <= 1: return "1 PA"
        if pa == 2: return "2 PA"
        if pa == 3: return "3 PA"
        return ">=4 PA"
    sub["__bucket"] = sub["pa"].map(_bucket)
    out: list[dict] = []
    for label in ("1 PA", "2 PA", "3 PA", ">=4 PA"):
        grp = sub[sub["__bucket"] == label]
        if grp.empty:
            continue
        err = grp[actual_col] - grp[proj_col]
        out.append({
            "label": label,
            "n": int(len(grp)),
            "rmse": float((err ** 2).mean() ** 0.5),
            "mae": float(err.abs().mean()),
            "bias": float(err.mean()),
            "actual_std": float(grp[actual_col].std()),
        })
    return out


# Per-event outcome variance for each metric, used to bias-correct the
# observed actual variance for sample-size noise.  Without this correction the
# ratio diagnostic is unreliable on short windows where per-game sampling noise
# dwarfs the true between-hitter variance.  Each entry is (per-event variance,
# sample-size column).  For rates that condition on BIP (hard-hit, on-contact
# xwOBA) we divide by n_bip rather than n_pa.  Per-event variances:
#   xwOBA per PA: outcomes 0 (K/Out) to ~2.0 (HR), Var ~ 0.25
#   xwOBA per BIP: no zeros from K/BB, slightly wider, Var ~ 0.30
#   binomial rates: p*(1-p) at the league baseline
_METRIC_PER_EVENT_VAR = {
    "proj_xwoba":            (0.25,        "pa"),
    "proj_xwoba_on_contact": (0.30,        "n_bip"),
    "proj_hardhit_pct":      (0.40 * 0.60, "n_bip"),
    "proj_k_pct":            (0.225 * 0.775, "pa"),
    "proj_bb_pct":           (0.085 * 0.915, "pa"),
}


def _spread_table(df: pd.DataFrame, metrics: list[tuple[str, str, str]],
                  min_pa: int = 3) -> list[dict]:
    """Bias-corrected variance comparison: proj_std vs implied-true actual_std.

    A single-game actual rate of a hitter has TWO sources of variance:
        Var(actual) = Var(true rate across hitters) + per-PA variance / n
    The first term is what the model is trying to estimate; the second is just
    sample-size noise from a finite-PA game.  Subtracting an estimate of the
    second term gives the implied "true" between-hitter variance, which is
    directly comparable to the model's projection variance.

    Ratio = proj_std / sqrt(max(observed_var - noise_var, eps))
        <  0.85  : projections too narrow -- model is hedging toward league mean
        >= 0.85 and <= 1.15 : calibrated spread
        >  1.15  : projections too wide -- model overconfident at the tails

    The raw (uncorrected) actual std is also reported so the reader can see how
    much of the observed variance is per-PA noise vs real between-hitter
    variance.  When the corrected estimate would be negative (very short
    windows, very small samples), the row reports "n/a" rather than a number.
    """
    sub = df[df["pa"].fillna(0) >= min_pa]
    out: list[dict] = []
    for label, proj_col, actual_col in metrics:
        if proj_col not in sub.columns or actual_col not in sub.columns:
            continue
        per_event = _METRIC_PER_EVENT_VAR.get(proj_col)
        if per_event is None:
            continue
        per_event_var, n_col = per_event
        if n_col not in sub.columns:
            continue
        g = sub.dropna(subset=[proj_col, actual_col, n_col])
        # For BIP-conditioned rates, drop rows with no BIP at all.
        g = g[g[n_col] >= 1]
        if len(g) < 5:
            continue
        proj_std = float(g[proj_col].std())
        act_std = float(g[actual_col].std())
        mean_n = float(g[n_col].mean())
        # Bias correction: average per-hitter sampling variance is
        # E[per_event_var / n_i], where n_i is that hitter's sample size.
        noise_var = float((per_event_var / g[n_col]).mean())
        corrected_var = act_std ** 2 - noise_var
        if corrected_var <= 0:
            corrected_std = float("nan")
            ratio = float("nan")
        else:
            corrected_std = corrected_var ** 0.5
            ratio = proj_std / corrected_std if corrected_std > 0 else float("nan")
        out.append({
            "label": label,
            "n": int(len(g)),
            "mean_n": mean_n,
            "n_col": n_col,
            "proj_mean": float(g[proj_col].mean()),
            "actual_mean": float(g[actual_col].mean()),
            "proj_std": proj_std,
            "actual_std_raw": act_std,
            "actual_std_corrected": corrected_std,
            "ratio": ratio,
        })
    return out


def _spearman_per_slate(df: pd.DataFrame, proj_col: str, actual_col: str,
                        min_hitters: int = 30) -> dict:
    """Spearman rho between proj and actual for each date with enough hitters."""
    sub = df.dropna(subset=[proj_col, actual_col])
    sub = sub[sub["pa"].fillna(0) >= 1]
    rhos: list[float] = []
    by_date: list[tuple[str, int, float]] = []
    for d, group in sub.groupby("date"):
        if len(group) < min_hitters:
            continue
        rho = group[proj_col].corr(group[actual_col], method="spearman")
        if rho is None or (isinstance(rho, float) and math.isnan(rho)):
            continue
        rhos.append(float(rho))
        by_date.append((str(d), int(len(group)), float(rho)))
    if not rhos:
        return {"n_days": 0, "mean": float("nan"),
                "p25": float("nan"), "median": float("nan"),
                "p75": float("nan"), "by_date": []}
    s = pd.Series(rhos)
    return {
        "n_days": len(rhos),
        "mean": float(s.mean()),
        "p25": float(s.quantile(0.25)),
        "median": float(s.median()),
        "p75": float(s.quantile(0.75)),
        "by_date": sorted(by_date, key=lambda t: t[0]),
    }


# ---------- HTML rendering -------------------------------------------------

_ACCURACY_CSS_EXTRA = """
.acc-section { margin-bottom: 28px; }
.acc-section h2 { margin-bottom: 6px; }
.acc-table { border-collapse: collapse; font-size: 0.92em;
             font-variant-numeric: tabular-nums; margin-bottom: 8px; }
.acc-table th, .acc-table td { padding: 4px 12px; }
.acc-table th { font-weight: 500; color: var(--muted);
                border-bottom: 1px solid var(--border, #ccc); text-align: right; }
.acc-table th.label, .acc-table td.label { text-align: left; color: var(--muted); }
.acc-table td { text-align: right; }
.acc-note { color: var(--muted); font-size: 0.9em; margin: 4px 0 14px 0; max-width: 70ch; }
.skill-pos { color: #1c8c4e; font-weight: 600; }
.skill-neg { color: #b34141; font-weight: 600; }
.delta-pos { color: #1c8c4e; font-weight: 600; }
.delta-neg { color: #b34141; font-weight: 600; }
.delta-zero { color: var(--muted); }
"""


def _fmt_pct(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v*100:.1f}%"


def _fmt_pts(v, decimals: int = 0) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:+.{decimals}f}"


def _fmt_float(v, decimals: int = 3) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:.{decimals}f}"


def _delta_cls(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "delta-zero"
    return "delta-pos" if v > 0 else ("delta-neg" if v < 0 else "delta-zero")


def _render_calibration_table(title: str, rows: list[CalibrationRow],
                              *, is_rate: bool, baseline: float | None = None,
                              note: str = "") -> str:
    parts = [f'<div class="acc-section"><h2>{_h(title)}</h2>']
    if note:
        # Notes may embed HTML entities (e.g. &ge;) or <span> formatting -- do
        # not escape them again here; callers control the markup.
        parts.append(f'<div class="acc-note">{note}</div>')
    if not rows:
        parts.append('<div class="acc-note">No data in window.</div></div>')
        return "\n".join(parts)
    parts.append('<table class="acc-table"><thead><tr>'
                 '<th class="label">Proj bin</th><th>n</th>'
                 '<th>mean proj</th><th>mean actual</th>'
                 '<th>&Delta; pts</th><th>95% CI</th></tr></thead><tbody>')
    fmt_v = (lambda v: _fmt_pct(v)) if is_rate else (lambda v: fmt3(v))
    for r in rows:
        delta_cls = _delta_cls(r.delta_pts) if abs(r.delta_pts) >= 5 else "delta-zero"
        ci_str = (f"[{fmt_v(r.ci_low)}, {fmt_v(r.ci_high)}]"
                  if not math.isnan(r.ci_low) else "—")
        parts.append(
            "<tr>"
            f'<td class="label">{_h(r.label)}</td>'
            f"<td>{r.n}</td>"
            f"<td>{fmt_v(r.mean_proj)}</td>"
            f"<td>{fmt_v(r.mean_actual)}</td>"
            f'<td class="{delta_cls}">{_fmt_pts(r.delta_pts)}</td>'
            f"<td>{_h(ci_str)}</td>"
            "</tr>"
        )
    parts.append("</tbody></table>")
    if baseline is not None:
        parts.append(f'<div class="acc-note">League baseline: '
                     f'{fmt_v(baseline) if is_rate else fmt3(baseline)}. '
                     "A well-calibrated model has &Delta; pts near zero in every bin.</div>")
    parts.append("</div>")
    return "\n".join(parts)


def _render_logloss_table(rows: list[dict]) -> str:
    parts = ['<div class="acc-section"><h2>Per-PA log-loss vs baselines</h2>',
             '<div class="acc-note">Proper score on the 8-class outcome '
             'distribution. Lower is better. Skill = 1 − model / baseline. '
             'Positive skill means the model is doing real work beyond the '
             'league prior. Sample shrinks fast in early-season windows.</div>']
    if not rows:
        parts.append('<div class="acc-note">No scored PAs in window.</div></div>')
        return "\n".join(parts)
    parts.append('<table class="acc-table"><thead><tr>'
                 '<th class="label">Source</th><th>n PA</th>'
                 '<th>log-loss</th><th>Brier</th>'
                 '<th>skill (LL)</th><th>skill (Brier)</th>'
                 '</tr></thead><tbody>')
    for r in rows:
        cls_ll = "skill-pos" if r["skill_ll"] > 0 else ("skill-neg" if r["skill_ll"] < 0 else "")
        cls_br = "skill-pos" if r["skill_br"] > 0 else ("skill-neg" if r["skill_br"] < 0 else "")
        parts.append(
            "<tr>"
            f'<td class="label">{_h(r["name"])}</td>'
            f'<td>{r["n_pa"]}</td>'
            f'<td>{_fmt_float(r["logloss"])}</td>'
            f'<td>{_fmt_float(r["brier"])}</td>'
            f'<td class="{cls_ll}">{_fmt_pct(r["skill_ll"])}</td>'
            f'<td class="{cls_br}">{_fmt_pct(r["skill_br"])}</td>'
            "</tr>"
        )
    parts.append("</tbody></table></div>")
    return "\n".join(parts)


def _render_rmse_table(blocks: list[tuple[str, dict]]) -> str:
    parts = ['<div class="acc-section"><h2>Hitter-grain accuracy (RMSE / MAE / bias)</h2>',
             '<div class="acc-note">RMSE / MAE on the proj vs actual headline '
             'numbers, computed only on hitters with PA ≥ 1. Lower is better. '
             'Bias = mean(actual − proj); persistent non-zero bias signals '
             'systematic under/over-projection.</div>']
    parts.append('<table class="acc-table"><thead><tr>'
                 '<th class="label">Metric</th><th>n</th>'
                 '<th>RMSE</th><th>MAE</th><th>bias</th>'
                 '</tr></thead><tbody>')
    for label, d in blocks:
        bias_cls = _delta_cls(d["bias"]) if abs(d["bias"] or 0) > 0.005 else "delta-zero"
        parts.append(
            "<tr>"
            f'<td class="label">{_h(label)}</td>'
            f'<td>{d["n"]}</td>'
            f'<td>{_fmt_float(d["rmse"])}</td>'
            f'<td>{_fmt_float(d["mae"])}</td>'
            f'<td class="{bias_cls}">{_fmt_float(d["bias"])}</td>'
            "</tr>"
        )
    parts.append("</tbody></table></div>")
    return "\n".join(parts)


def _render_pa_bucket_rmse(rows: list[dict]) -> str:
    parts = ['<div class="acc-section">'
             '<h2>Headline xwOBA RMSE by PA count</h2>',
             '<div class="acc-note">Single-PA games carry ~80% of the per-hitter '
             'xwOBA variance (a HR vs a K is a 2.0 wOBA point gap on one PA). '
             'Splitting RMSE by PA count separates real model miss from sample-size '
             'noise. A model with skill should show RMSE shrinking with PA count.</div>']
    if not rows:
        parts.append('<div class="acc-note">No data in window.</div></div>')
        return "\n".join(parts)
    parts.append('<table class="acc-table"><thead><tr>'
                 '<th class="label">PA bucket</th><th>n</th>'
                 '<th>actual std</th><th>RMSE</th><th>MAE</th><th>bias</th>'
                 '</tr></thead><tbody>')
    for r in rows:
        bias_cls = _delta_cls(r["bias"]) if abs(r["bias"] or 0) > 0.01 else "delta-zero"
        parts.append(
            "<tr>"
            f'<td class="label">{_h(r["label"])}</td>'
            f'<td>{r["n"]}</td>'
            f'<td>{_fmt_float(r["actual_std"])}</td>'
            f'<td>{_fmt_float(r["rmse"])}</td>'
            f'<td>{_fmt_float(r["mae"])}</td>'
            f'<td class="{bias_cls}">{_fmt_float(r["bias"])}</td>'
            "</tr>"
        )
    parts.append("</tbody></table></div>")
    return "\n".join(parts)


def _render_spread(rows: list[dict], min_pa: int) -> str:
    parts = ['<div class="acc-section">'
             '<h2>Distribution spread: projection variance vs implied true variance</h2>',
             '<div class="acc-note">For each metric, compares projected std '
             f'(across hitter-games with PA &ge; {min_pa}) to the implied true '
             'between-hitter std after subtracting per-PA sampling noise. Raw '
             'actual std also shown so you can see how much of it is genuine '
             'between-hitter spread vs single-game noise. '
             '<b>Ratio = proj std / corrected actual std.</b> '
             '<span class="skill-neg">&lt; 0.85</span> = projections too '
             'narrow (hedging toward league); '
             '<span class="skill-pos">&gt; 1.15</span> = projections too wide '
             '(overconfident at tails); 0.85-1.15 = calibrated spread. '
             'On short windows (few days) the bias correction can fail '
             '(corrected std non-finite) — gather more data.</div>']
    if not rows:
        parts.append('<div class="acc-note">No data in window.</div></div>')
        return "\n".join(parts)
    parts.append('<table class="acc-table"><thead><tr>'
                 '<th class="label">Metric</th><th>n</th>'
                 '<th title="Mean sample-size denominator: PA for headline / K / BB; '
                 'n_bip for hard-hit and on-contact xwOBA">mean n</th>'
                 '<th>mean proj</th><th>mean actual</th>'
                 '<th>proj std</th><th>actual std (raw)</th>'
                 '<th>actual std (corrected)</th>'
                 '<th>ratio</th><th>read</th>'
                 '</tr></thead><tbody>')
    for r in rows:
        ratio = r["ratio"]
        if not isinstance(ratio, float) or math.isnan(ratio):
            ratio_cls = ""
            verdict = "n/a (noisy)"
            ratio_str = "—"
            corr_str = "—"
        else:
            if ratio < 0.85:
                ratio_cls = "skill-neg"
                verdict = "hedging"
            elif ratio > 1.15:
                ratio_cls = "skill-pos"
                verdict = "overconfident"
            else:
                ratio_cls = ""
                verdict = "calibrated"
            ratio_str = _fmt_float(ratio, 2)
            corr_str = _fmt_float(r["actual_std_corrected"])
        parts.append(
            "<tr>"
            f'<td class="label">{_h(r["label"])}</td>'
            f'<td>{r["n"]}</td>'
            f'<td>{_fmt_float(r["mean_n"], 1)}</td>'
            f'<td>{_fmt_float(r["proj_mean"])}</td>'
            f'<td>{_fmt_float(r["actual_mean"])}</td>'
            f'<td>{_fmt_float(r["proj_std"])}</td>'
            f'<td>{_fmt_float(r["actual_std_raw"])}</td>'
            f'<td>{_h(corr_str)}</td>'
            f'<td class="{ratio_cls}">{_h(ratio_str)}</td>'
            f'<td class="{ratio_cls}">{verdict}</td>'
            "</tr>"
        )
    parts.append("</tbody></table></div>")
    return "\n".join(parts)


def _render_spearman(blocks: list[tuple[str, dict]]) -> str:
    parts = ['<div class="acc-section"><h2>Discrimination: Spearman rho per slate</h2>',
             '<div class="acc-note">For each date with at least 30 played '
             'hitters, Spearman rank correlation between projected and actual. '
             'Sustained positive rho means the model is ordering hitters '
             'correctly even when individual deltas are noisy. Spearman is '
             'less sensitive to BABIP outliers than RMSE.</div>']
    parts.append('<table class="acc-table"><thead><tr>'
                 '<th class="label">Metric</th><th>n days</th>'
                 '<th>mean &rho;</th><th>p25</th><th>median</th><th>p75</th>'
                 '</tr></thead><tbody>')
    for label, d in blocks:
        parts.append(
            "<tr>"
            f'<td class="label">{_h(label)}</td>'
            f'<td>{d["n_days"]}</td>'
            f'<td>{_fmt_float(d["mean"])}</td>'
            f'<td>{_fmt_float(d["p25"])}</td>'
            f'<td>{_fmt_float(d["median"])}</td>'
            f'<td>{_fmt_float(d["p75"])}</td>'
            "</tr>"
        )
    parts.append("</tbody></table>")
    # Per-date breakdown for the first (xwOBA) block.
    if blocks and blocks[0][1]["by_date"]:
        parts.append('<details><summary>Per-day xwOBA rank correlation</summary>')
        parts.append('<table class="acc-table"><thead><tr>'
                     '<th class="label">Date</th><th>hitters</th><th>&rho;</th>'
                     '</tr></thead><tbody>')
        for d_str, n_h, rho in blocks[0][1]["by_date"]:
            parts.append(
                "<tr>"
                f'<td class="label">{_h(d_str)}</td>'
                f'<td>{n_h}</td>'
                f'<td>{_fmt_float(rho)}</td>'
                "</tr>"
            )
        parts.append("</tbody></table></details>")
    parts.append("</div>")
    return "\n".join(parts)


def _render_html(hitter: pd.DataFrame, pa: pd.DataFrame,
                 window_label: str, min_pa: int = 3) -> str:
    title = f"Projection accuracy — {window_label}"
    parts = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        f"<title>{_h(title)}</title>",
        f"<style>{_HTML_CSS}{_ACCURACY_CSS_EXTRA}</style>",
        "</head><body>",
        '<main class="container">',
        '<header class="page-head">',
        f"<h1>{_h(title)}</h1>",
        f'<div class="meta">{len(hitter)} hitter-games across '
        f'{hitter["date"].nunique() if not hitter.empty else 0} slates, '
        f'{len(pa)} scored PAs.</div>',
        "</header>",
    ]

    parts.append('<section class="card">')

    # Phase 5: calibration filters out short games (1-2 PA) so single-PA noise
    # doesn't dominate the bin means.  Per-block min_pa overrides for the rare
    # HR% case (already starved for data).
    pa_default = min_pa
    calib_blocks = [
        ("proj_xwoba", "actual_xwoba", _bin_xwoba, False, LG_XWOBA, pa_default,
         f"Rolled-up xwOBA, hitters with PA &ge; {pa_default}. "
         "Highest variance per hitter-game; converges slowly.",
         "Calibration: proj xwOBA → actual xwOBA"),
        ("proj_xwoba_on_contact", "actual_xwoba_on_contact", _bin_xwoba, False, None, pa_default,
         f"On-contact xwOBA strips BABIP noise; converges 2-3x faster than "
         f"rolled-up xwOBA. PA &ge; {pa_default}.",
         "Calibration: proj on-contact xwOBA → actual on-contact xwOBA"),
        ("proj_hardhit_pct", "actual_hardhit_pct",
         lambda v: _bin_pct(v, (0.30, 0.35, 0.40, 0.45, 0.50, 0.55)), True, LG_HARD_HIT,
         pa_default,
         f"Hard-hit % converges fastest of any contact-quality metric. PA &ge; {pa_default}.",
         "Calibration: proj hard-hit% → actual hard-hit%"),
        ("proj_k_pct", "actual_k_pct",
         lambda v: _bin_pct(v, (0.15, 0.20, 0.25, 0.30, 0.35)), True, LG_K_PCT,
         pa_default,
         f"Strikeout rate. Discrete event, low-variance baseline. PA &ge; {pa_default}.",
         "Calibration: proj K% → actual K%"),
        ("proj_hr_pct", "actual_hr_pct",
         lambda v: _bin_pct(v, (0.02, 0.04, 0.06, 0.08, 0.12)), True, None,
         1,  # HR is rare; don't starve the bins further
         "HR rate — rare event, needs hundreds of hitter-games per bin to converge.",
         "Calibration: proj HR% → actual HR%"),
    ]
    for (proj_col, actual_col, bin_fn, is_rate,
         baseline, blk_min_pa, note, title_c) in calib_blocks:
        if proj_col not in hitter.columns or actual_col not in hitter.columns:
            continue
        rows = _calibration_rows(hitter, proj_col, actual_col, bin_fn,
                                 is_rate=is_rate, min_n_pa=blk_min_pa)
        parts.append(_render_calibration_table(title_c, rows, is_rate=is_rate,
                                                baseline=baseline, note=note))

    # A2. Distribution spread diagnostic (hedging-detector).
    spread_rows = _spread_table(hitter, [
        ("xwOBA",             "proj_xwoba",             "actual_xwoba"),
        ("on-contact xwOBA",  "proj_xwoba_on_contact",  "actual_xwoba_on_contact"),
        ("hard-hit %",        "proj_hardhit_pct",       "actual_hardhit_pct"),
        ("K %",               "proj_k_pct",             "actual_k_pct"),
        ("BB %",              "proj_bb_pct",            "actual_bb_pct"),
    ], min_pa=min_pa)
    parts.append(_render_spread(spread_rows, min_pa))

    # A3. Headline xwOBA RMSE split by PA count.
    parts.append(_render_pa_bucket_rmse(_pa_bucket_rmse(
        hitter, "proj_xwoba", "actual_xwoba")))

    # B. Per-PA log-loss vs baselines.
    parts.append(_render_logloss_table(_logloss_table(pa)))

    # C. RMSE / MAE / bias.
    rmse_blocks = [
        ("xwOBA",            _hitter_accuracy(hitter, "proj_xwoba", "actual_xwoba")),
        ("on-contact xwOBA", _hitter_accuracy(hitter, "proj_xwoba_on_contact",
                                              "actual_xwoba_on_contact")),
        ("hard-hit %",       _hitter_accuracy(hitter, "proj_hardhit_pct",
                                              "actual_hardhit_pct")),
        ("K %",              _hitter_accuracy(hitter, "proj_k_pct", "actual_k_pct")),
        ("BB %",             _hitter_accuracy(hitter, "proj_bb_pct", "actual_bb_pct")),
    ]
    parts.append(_render_rmse_table(rmse_blocks))

    # D. Spearman rho per slate.
    spearman_blocks = [
        ("xwOBA",            _spearman_per_slate(hitter, "proj_xwoba", "actual_xwoba")),
        ("on-contact xwOBA", _spearman_per_slate(hitter, "proj_xwoba_on_contact",
                                                 "actual_xwoba_on_contact")),
        ("hard-hit %",       _spearman_per_slate(hitter, "proj_hardhit_pct",
                                                 "actual_hardhit_pct")),
    ]
    parts.append(_render_spearman(spearman_blocks))

    parts.append("</section>")
    parts.append("</main>")
    parts.append(sortable_html())
    parts.append("</body></html>")
    return "\n".join(parts)


# ---------- main -----------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--window", default="30",
                    help="Trailing window in days, or 'all' (default: 30)")
    ap.add_argument("--since", default=None,
                    help="Override --window with an explicit YYYY-MM-DD cutoff.")
    ap.add_argument("--out", default=str(OUT_DIR),
                    help=f"Output directory (default: {OUT_DIR})")
    ap.add_argument("--min-pa", type=int, default=3,
                    help="Filter calibration + spread tables to hitters with "
                         "at least N PA in their game (default: 3). Drops "
                         "1-2 PA games whose single-PA variance otherwise "
                         "drowns out real signal in the bin means.")
    args = ap.parse_args()

    log_path = setup_logging("accuracy")
    print(f"[accuracy] logging to {log_path}")

    if args.window == "all":
        window = "all"
    else:
        try:
            window = int(args.window)
        except ValueError:
            sys.exit(f"[accuracy] invalid --window: {args.window!r}")

    hitter, pa = _load_window(window, args.since)
    if hitter.empty:
        sys.exit("[accuracy] no hitter rows in window - run postgame.py first.")

    if args.since:
        window_label = f"since {args.since} ({hitter['date'].min()} to {hitter['date'].max()})"
    elif window == "all":
        window_label = (f"all dates ({hitter['date'].min()} to "
                        f"{hitter['date'].max()})")
    else:
        window_label = (f"trailing {window} days "
                        f"({hitter['date'].min()} to {hitter['date'].max()})")

    html = _render_html(hitter, pa, window_label, min_pa=args.min_pa)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.html"
    snapshot_path = out_dir / f"{date_cls.today().isoformat()}.html"
    index_path.write_text(html, encoding="utf-8")
    snapshot_path.write_text(html, encoding="utf-8")

    def _rel(p: Path) -> str:
        try:
            return str(p.relative_to(ROOT))
        except ValueError:
            return str(p)

    print(f"[accuracy] window: {window_label}")
    print(f"[accuracy] hitter-games: {len(hitter)}  scored PAs: {len(pa)}")
    print(f"[accuracy] wrote {_rel(index_path)}")
    print(f"[accuracy] wrote {_rel(snapshot_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
