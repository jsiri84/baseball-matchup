"""
matchup.py - batter vs pitcher matchup analysis.

Pulls Statcast data for a batter and a pitcher (current + prior season,
recency-weighted), runs a multi-layer projection (count-conditional pitch mix,
shape-aware comps, zone overlay, TTO curve, bat-tracking interaction,
sub-profiles, deception, defensive alignment), and writes a markdown report
with a verdict, narrative, and per-PA outcome probability table.

Usage:
    python matchup.py
    python matchup.py --batter "Yordan Alvarez" --pitcher "Paul Skenes"
    python matchup.py --batter-id 670541 --pitcher-id 694973 --season 2026
    python matchup.py --batch matchups.csv

CSV formats (no header):
    - Legacy format (2 columns): batter,pitcher  (names or numeric MLBAM ids)
    - Lineup format (4 columns): away@home,hitter_name,pitcher_name,lineup_position
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import html
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from batter import (
    PITCH_GROUPS,
    SWING_DESCRIPTIONS,
    WHIFF_DESCRIPTIONS,
    WOBA_1B,
    WOBA_2B,
    WOBA_3B,
    WOBA_BB,
    WOBA_HBP,
    WOBA_HR,
    _pitch_group,
    get_player_id,
    pull as _pull_batter_raw,
)
from pitcher import pull as _pull_pitcher_raw
from log_setup import setup_logging
from sortable import sortable_html

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)

ROOT = Path(__file__).parent

# ---------- constants ------------------------------------------------------

# 2025 league-average baselines (FanGraphs / Savant); 2026 finals aren't
# published until the offseason. Tune `SEASON_WEIGHTS` to control how the
# rolling window of seasons is combined: index 0 = current season, index 1 =
# prior season, etc. Defaults to a BPP-style 3-season decay [1.0, 0.5, 0.25].
# Set to [1.0] for current-only, [1.0, 0.5] for the original 2-season blend,
# or [1.0, 0.5, 0.25, 0.1] to extend further back.
SEASON_WEIGHTS = [1.0, 0.5, 0.25]

# Within the CURRENT season, multiply each row's weight by an exponential
# recency decay so hot/cold streaks actually move the headline. Half-life is
# the number of days for the multiplier to drop to 0.5x; an April pitch in
# late September never falls below RECENCY_FLOOR. Prior-season rows are not
# touched - the SEASON_WEIGHTS cascade already handles cross-year staleness,
# and any meaningful half-life would zero out 2024/2025 rows.
RECENCY_HALF_LIFE_DAYS = 30.0
RECENCY_FLOOR = 0.20
# Override the "today" reference used by the decay (None -> date.today()).
# Set in tests to make the math deterministic.
RECENCY_REFERENCE_DATE: date | None = None
# Hot/cold streak threshold: if the batter's (or pitcher's) 14-day rolling
# xwOBA differs from their season blend by more than this many wOBA points,
# the verdict narrative appends a "heater" / "slump" tail.
RECENT_FORM_STREAK_THRESHOLD = 0.040
# Min effective PA in a window before we trust the rolling snapshot (panel
# renders "-" below this gate, narrative skips the hot/cold note).
RECENT_FORM_MIN_PA = 25.0
# Rolling windows surfaced in the Recent form panel.
RECENT_FORM_WINDOWS = (14, 30)

LG_XWOBA = 0.315
LG_XBA = 0.245
LG_XSLG = 0.405
LG_K_PCT = 0.225
LG_BB_PCT = 0.085
LG_HBP_PCT = 0.012
LG_WHIFF = 0.245
LG_HARD_HIT = 0.40

# Per-PA outcome distribution baseline (sums to 1.00; sourced from 2025 MLB).
LG_OUTCOMES = {
    "K": 0.225,
    "BB": 0.085,
    "HBP": 0.012,
    "1B": 0.140,
    "2B": 0.045,
    "3B": 0.005,
    "HR": 0.030,
    "BIP_out": 0.458,
}

HIT_EVENTS = {"single", "double", "triple", "home_run"}
NON_AB_EVENTS = {"walk", "hit_by_pitch", "sac_fly", "sac_bunt", "catcher_interf"}
K_EVENTS = {"strikeout", "strikeout_double_play"}

# League BBE-only contact baselines by Statcast bb_type, calibrated from
# data/2025/ via scripts/_estimate_league_bbtype_baselines.py. Used by the
# bb_type-aware per-pitch projection (Layer 1) to re-weight a batter's per-pitch
# xwOBA/xBA/xSLG using the pitcher's induced bb_type distribution rather than
# the batter's career launch-angle mix. Re-run the calibration script when the
# league environment shifts (typically once per offseason).
BB_TYPES = ("ground_ball", "line_drive", "fly_ball", "popup")
LG_XWOBA_BY_BBTYPE = {
    "ground_ball": 0.2312,
    "line_drive":  0.6462,
    "fly_ball":    0.4435,
    "popup":       0.0323,
}
LG_XBA_BY_BBTYPE = {
    "ground_ball": 0.2504,
    "line_drive":  0.6204,
    "fly_ball":    0.2585,
    "popup":       0.0340,
}
LG_XSLG_BY_BBTYPE = {
    "ground_ball": 0.2756,
    "line_drive":  0.8899,
    "fly_ball":    0.8421,
    "popup":       0.0387,
}
# Sample-size thresholds for the per-(pitch, bb_type) -> per-bb_type -> league
# fallback chain when computing bb_type-stratified xwOBA/xBA/xSLG.
BBTYPE_MIN_BBE_PER_PITCH = 10.0
BBTYPE_MIN_BBE_OVERALL = 25.0

# Statcast zone grid: 1-9 in-zone, 11-14 out-of-zone quadrants.
IN_ZONE = list(range(1, 10))
OOZ_ZONE = [11, 12, 13, 14]
ALL_ZONES = IN_ZONE + OOZ_ZONE

# Layer-2 shape-comp tolerances (and 1.5x widening on fallback).
SHAPE_TOL_VELO = 2.0   # mph (effective_speed)
SHAPE_TOL_HB = 3.0     # inches (api_break_x_batter_in)
SHAPE_TOL_VB = 3.0     # inches (api_break_z_with_gravity)
SHAPE_FALLBACK = 1.5

# Defaults for single-matchup mode.
DEFAULT_BATTER_NAME = "Yordan Alvarez"
DEFAULT_BATTER_ID = 670541
DEFAULT_PITCHER_NAME = "Paul Skenes"
DEFAULT_PITCHER_ID = 694973
DEFAULT_SEASON = 2026


# ---------- caching wrappers ----------------------------------------------

# Round end date to yesterday so all matchups run on a given day share parquet
# files and the in-process cache.
def _season_window(season: int, offset: int) -> tuple[str, str]:
    """Date range for a season offset by `offset` years from the current season.

    `offset == 0` returns the current season's window (March 27 -> yesterday).
    `offset >= 1` returns the regular-season window of a prior year.
    """
    yr = season - offset
    if offset == 0:
        end = (date.today() - timedelta(days=1)).isoformat()
        return f"{yr}-03-27", end
    return f"{yr}-03-27", f"{yr}-11-01"


_BATTER_CACHE: dict[int, pd.DataFrame] = {}
_PITCHER_CACHE: dict[int, pd.DataFrame] = {}
_BATTER_CACHE_LOCK = threading.Lock()
_PITCHER_CACHE_LOCK = threading.Lock()


def _tag(df: pd.DataFrame, weight: float) -> pd.DataFrame:
    out = df.copy()
    out["weight"] = float(weight)
    return out


def _per_season_counts(df: pd.DataFrame) -> dict[float, int]:
    """Map season-weight -> row count for a blended frame. Missing weights -> 0.

    Bucketed by `game_date.year` (not by row weight), so the count survives the
    recency decay that mutates current-season row weights down from 1.0.
    """
    out: dict[float, int] = {float(w): 0 for w in SEASON_WEIGHTS}
    if df.empty or "game_date" not in df.columns:
        return out
    gd = df["game_date"]
    if not pd.api.types.is_datetime64_any_dtype(gd):
        gd = pd.to_datetime(gd, errors="coerce")
    years = gd.dt.year
    base_year = DEFAULT_SEASON
    for offset, w in enumerate(SEASON_WEIGHTS):
        out[float(w)] = int(years.eq(base_year - offset).sum())
    return out


def _load_blended(
    player_id: int,
    season: int,
    pull_fn,
    label: str,
) -> pd.DataFrame:
    """Walk SEASON_WEIGHTS and pull each year, tag with weight, concat."""
    pieces: list[pd.DataFrame] = []
    template: pd.DataFrame | None = None
    for offset, weight in enumerate(SEASON_WEIGHTS):
        start, end = _season_window(season, offset)
        try:
            df = pull_fn(player_id, start, end)
        except Exception as exc:                                  # noqa: BLE001
            yr = season - offset
            print(f"  {label} {yr} pull failed ({exc.__class__.__name__}); skipping")
            continue
        if template is None and len(df) > 0:
            template = df.iloc[0:0]
        if len(df) > 0:
            pieces.append(_tag(df, weight))

    if not pieces:
        empty = template if template is not None else pd.DataFrame()
        empty = empty.copy()
        empty["weight"] = pd.Series(dtype=float)
        return empty

    return pd.concat(pieces, ignore_index=True)


def load_blended_batter(player_id: int, season: int = DEFAULT_SEASON) -> pd.DataFrame:
    """Return Statcast for a batter blended across SEASON_WEIGHTS years."""
    with _BATTER_CACHE_LOCK:
        if player_id in _BATTER_CACHE:
            return _BATTER_CACHE[player_id]
    blended = _load_blended(player_id, season, _pull_batter_raw, "batter")
    with _BATTER_CACHE_LOCK:
        _BATTER_CACHE[player_id] = blended
    return blended


def load_blended_pitcher(player_id: int, season: int = DEFAULT_SEASON) -> pd.DataFrame:
    with _PITCHER_CACHE_LOCK:
        if player_id in _PITCHER_CACHE:
            return _PITCHER_CACHE[player_id]
    blended = _load_blended(player_id, season, _pull_pitcher_raw, "pitcher")
    with _PITCHER_CACHE_LOCK:
        _PITCHER_CACHE[player_id] = blended
    return blended


def preload_player_data(
    batter_ids: set[int], pitcher_ids: set[int], season: int = DEFAULT_SEASON,
    max_workers: int | None = None,
) -> None:
    """Pull all batter and pitcher data into the in-process cache."""
    if not batter_ids and not pitcher_ids:
        return

    if max_workers is None:
        max_workers = min(32, (os.cpu_count() or 1) * 5)

    print(f"Preloading {len(batter_ids)} batters + {len(pitcher_ids)} pitchers into cache...")
    futures: dict[concurrent.futures.Future[None], str] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for bid in sorted(batter_ids):
            futures[executor.submit(load_blended_batter, bid, season)] = f"batter {bid}"
        for pid in sorted(pitcher_ids):
            futures[executor.submit(load_blended_pitcher, pid, season)] = f"pitcher {pid}"

        for future in concurrent.futures.as_completed(futures):
            label = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                print(f"  {label} preload failed: {exc.__class__.__name__}: {exc}", file=sys.stderr)

    print("Preload complete.")


# ---------- weighted helpers ----------------------------------------------

def w_sum(values: pd.Series, weights: pd.Series) -> float:
    s = float(weights.sum())
    if not s:
        return 0.0
    return float((values * weights).sum())


def w_mean(values: pd.Series, weights: pd.Series) -> float:
    s = float(weights.sum())
    if not s:
        return 0.0
    return float((values * weights).sum() / s)


def w_rate(mask: pd.Series, weights: pd.Series) -> float:
    s = float(weights.sum())
    if not s:
        return 0.0
    return float((mask.astype(float) * weights).sum() / s)


def w_xwoba(group: pd.DataFrame) -> float:
    """Statcast-style xwOBA: per-PA weighted mean of estimated_woba_using_speedangle.

    Uses BIP estimates plus walk/HBP fixed weights, matching the convention in
    batter.py compute_stats.

    Returns NaN (not 0.0) when there are no resolved PAs in the subset, so
    downstream callers can distinguish "no sample" from "everyone made an out".
    """
    pa_rows = group[group["events"].notna()]
    if pa_rows.empty:
        return float("nan")
    n_pa = float(pa_rows["weight"].sum())
    if not n_pa:
        return float("nan")

    bb_w = float(pa_rows.loc[pa_rows["events"] == "walk", "weight"].sum())
    hbp_w = float(pa_rows.loc[pa_rows["events"] == "hit_by_pitch", "weight"].sum())

    bbe_x = group[(group["type"] == "X") & group["estimated_woba_using_speedangle"].notna()]
    contact = float((bbe_x["estimated_woba_using_speedangle"] * bbe_x["weight"]).sum())

    return (contact + WOBA_BB * bb_w + WOBA_HBP * hbp_w) / n_pa


def w_xba_xslg(group: pd.DataFrame) -> tuple[float, float]:
    """Return (xBA, xSLG) computed Savant-style: per-AB, K's count as 0.

    Returns (NaN, NaN) when there are no AB-eligible events in the subset.
    """
    pa_rows = group[group["events"].notna()]
    if pa_rows.empty:
        return float("nan"), float("nan")

    so_w = float(pa_rows.loc[pa_rows["events"].isin(K_EVENTS), "weight"].sum())
    bbe_x = group[(group["type"] == "X") & group["estimated_ba_using_speedangle"].notna()]
    bbe_w = float(bbe_x["weight"].sum())
    denom = bbe_w + so_w
    if not denom:
        return float("nan"), float("nan")
    xba = float((bbe_x["estimated_ba_using_speedangle"] * bbe_x["weight"]).sum()) / denom
    xslg = float((bbe_x["estimated_slg_using_speedangle"] * bbe_x["weight"]).sum()) / denom
    return xba, xslg


def log5(b: float, p: float, lg: float) -> float:
    """Odds-ratio combination of batter and pitcher rates against league avg."""
    if lg <= 0 or lg >= 1:
        return (b + p) / 2
    b = min(max(b, 1e-6), 1 - 1e-6)
    p = min(max(p, 1e-6), 1 - 1e-6)
    num = (b * p / lg)
    den = num + (1 - b) * (1 - p) / (1 - lg)
    return num / den if den else 0.0


def additive(b: float, p: float, lg: float) -> float:
    """Additive combination for continuous stats: clipped to [0, 1]."""
    return float(np.clip(b + p - lg, 0.0, 1.0))


def american_odds(p: float) -> str:
    """Convert a probability to American odds (e.g., 0.032 -> '+3025')."""
    if p <= 0:
        return "—"
    if p >= 1:
        return "—"
    if p >= 0.5:
        return f"{int(round(-100 * p / (1 - p)))}"
    return f"+{int(round(100 * (1 - p) / p))}"


# ---------- player metadata ------------------------------------------------

def _player_meta(df: pd.DataFrame, role: str) -> dict:
    """Read handedness + display name from a Statcast frame.

    Statcast stores `player_name` as 'Last, First'; we normalize to 'First Last'
    for the report and capture `last_name` separately for the output filename
    and possessive forms.
    """
    if df.empty:
        return {"name": "Unknown", "last": "unknown", "possessive": "Unknown's",
                "stand": None, "p_throws": None}
    raw = df["player_name"].dropna().iloc[0] if "player_name" in df.columns else "Unknown"

    if "," in raw:
        last, first = [p.strip() for p in raw.split(",", 1)]
        display = f"{first} {last}"
    else:
        parts = raw.split()
        last = parts[-1] if parts else "Unknown"
        display = raw

    possessive = f"{last}'s" if not last.endswith("s") else f"{last}'"

    if role == "batter":
        stand_series = df["stand"].dropna()
        stand = stand_series.iloc[0] if not stand_series.empty else None
        return {"name": display, "last": last, "possessive": possessive,
                "stand": stand, "p_throws": None}
    else:
        pt_series = df["p_throws"].dropna()
        p_throws = pt_series.iloc[0] if not pt_series.empty else None
        return {"name": display, "last": last, "possessive": possessive,
                "stand": None, "p_throws": p_throws}


def _resolve_player(arg: str, fallback_id: int | None = None) -> int:
    """Resolve a 'First Last' string or numeric id to an MLBAM id."""
    arg = arg.strip()
    if arg.isdigit():
        return int(arg)
    parts = arg.split()
    if len(parts) < 2:
        raise SystemExit(f"Cannot parse '{arg}' — use 'First Last' or an MLBAM id")
    first, last = parts[0], " ".join(parts[1:])
    return get_player_id(last, first, fallback=fallback_id)


# ---------- Layer 0: prep -------------------------------------------------

def _recency_reference_date() -> date:
    """Reference 'today' for recency decay. Honors RECENCY_REFERENCE_DATE override."""
    return RECENCY_REFERENCE_DATE if RECENCY_REFERENCE_DATE is not None else date.today()


def _apply_recency_decay(df: pd.DataFrame, ref_date: date | None = None) -> pd.DataFrame:
    """Multiply current-season row weights by an exponential recency decay.

    Prior-season rows are untouched. `df` is mutated in place and returned.
    Idempotent guard: writes a `_recency_applied` attribute on the frame so
    repeat calls (e.g. when `_prepare` is invoked twice in tests) don't
    compound the decay.
    """
    if df.empty or "weight" not in df.columns or "game_date" not in df.columns:
        return df
    if getattr(df, "attrs", {}).get("_recency_applied"):
        return df
    if ref_date is None:
        ref_date = _recency_reference_date()
    gd = df["game_date"]
    if not pd.api.types.is_datetime64_any_dtype(gd):
        gd = pd.to_datetime(gd, errors="coerce")
    cur_year_mask = gd.dt.year.eq(ref_date.year).fillna(False).to_numpy()
    if not cur_year_mask.any():
        df.attrs["_recency_applied"] = True
        return df
    days_old = (pd.Timestamp(ref_date) - gd).dt.days.to_numpy(dtype=float)
    decay = np.power(2.0, -np.maximum(days_old, 0.0) / RECENCY_HALF_LIFE_DAYS)
    decay = np.clip(decay, RECENCY_FLOOR, 1.0)
    multiplier = np.where(cur_year_mask, decay, 1.0)
    df["weight"] = df["weight"].astype(float).to_numpy() * multiplier
    df.attrs["_recency_applied"] = True
    return df


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Add count_state, in_zone, tto_bucket columns and apply the current-season
    recency decay. Idempotent on the cached frame.
    """
    if df.empty:
        return df
    df = df.copy()
    df["count_state"] = df["balls"].astype("Int64").astype(str) + "-" + df["strikes"].astype("Int64").astype(str)
    df["in_zone"] = df["zone"].between(1, 9)
    df["tto_bucket"] = df["n_thruorder_pitcher"].clip(upper=3).fillna(1).astype(int)
    _apply_recency_decay(df)
    return df


# ---------- per-pitch-type aggregation ------------------------------------

def _bbtype_breakdown(group: pd.DataFrame) -> dict:
    """For a (pitch_name) group, return per-bb_type weighted BBE counts and
    BBE-only contact means used by the bb_type-aware projection.

    Output keys per bb_type b in BB_TYPES: bbe_<b>_w, share_<b>, xwoba_<b>,
    xba_<b>, xslg_<b>. NaN means insufficient sample for that cell.
    """
    out: dict = {}
    bbe_typed = group[(group["type"] == "X") & group["bb_type"].notna()]
    n_total = float(bbe_typed["weight"].sum())
    for b in BB_TYPES:
        sub = bbe_typed[bbe_typed["bb_type"] == b]
        n_w = float(sub["weight"].sum())
        out[f"bbe_{b}_w"] = n_w
        out[f"share_{b}"] = (n_w / n_total) if n_total > 0 else float("nan")
        if n_w <= 0:
            out[f"xwoba_{b}"] = float("nan")
            out[f"xba_{b}"] = float("nan")
            out[f"xslg_{b}"] = float("nan")
            continue
        out[f"xwoba_{b}"] = w_mean(sub["estimated_woba_using_speedangle"], sub["weight"])
        out[f"xba_{b}"]   = w_mean(sub["estimated_ba_using_speedangle"],   sub["weight"])
        out[f"xslg_{b}"]  = w_mean(sub["estimated_slg_using_speedangle"],  sub["weight"])
    return out


def per_pitch_type_table(df: pd.DataFrame) -> pd.DataFrame:
    """Weighted per-pitch-name aggregates used by Layers 1, 8, 10."""
    if df.empty or "pitch_name" not in df.columns:
        return pd.DataFrame()

    rows = []
    for pitch_name, group in df.dropna(subset=["pitch_name"]).groupby("pitch_name"):
        n_pitches_w = float(group["weight"].sum())
        if n_pitches_w == 0:
            continue
        pa = group[group["events"].notna()]
        n_pa_w = float(pa["weight"].sum())
        # Drop pitch types so thin we have essentially no PA-ending evidence.
        # The row would otherwise carry xwOBA == NaN through the projection
        # ladder; we'd rather fall back to the player's overall numbers in
        # `project()` for that pitch.
        if n_pa_w < 1.0:
            continue

        swings = group["description"].isin(SWING_DESCRIPTIONS)
        whiffs = group["description"].isin(WHIFF_DESCRIPTIONS)
        n_swings = float((swings.astype(float) * group["weight"]).sum())
        whiff_pct = float((whiffs.astype(float) * group["weight"]).sum()) / n_swings if n_swings else 0.0

        bbe = group[(group["type"] == "X") & group["launch_speed"].notna()]
        n_bbe_w = float(bbe["weight"].sum())
        hh_pct = (float(((bbe["launch_speed"] >= 95).astype(float) * bbe["weight"]).sum()) / n_bbe_w
                  if n_bbe_w else 0.0)

        # PA outcome counts (weighted)
        def _w_count(events_subset):
            mask = pa["events"].isin(events_subset)
            return float((mask.astype(float) * pa["weight"]).sum())

        n_k_w = _w_count(K_EVENTS)
        n_bb_w = _w_count({"walk"})
        n_hbp_w = _w_count({"hit_by_pitch"})
        n_1b_w = _w_count({"single"})
        n_2b_w = _w_count({"double"})
        n_3b_w = _w_count({"triple"})
        n_hr_w = _w_count({"home_run"})

        k_pct = n_k_w / n_pa_w if n_pa_w else 0.0
        xwoba = w_xwoba(group)
        xba, xslg = w_xba_xslg(group)

        velo_w = w_mean(group["effective_speed"].fillna(group["release_speed"]), group["weight"])
        ivb_w = w_mean(group["api_break_z_with_gravity"], group["weight"])
        hb_in_w = w_mean(group["api_break_x_batter_in"], group["weight"])
        spin_w = w_mean(group["release_spin_rate"], group["weight"])
        ext_w = w_mean(group["release_extension"], group["weight"])
        spin_axis_w = w_mean(group["spin_axis"], group["weight"])
        rel_x_w = w_mean(group["release_pos_x"], group["weight"])
        rel_z_w = w_mean(group["release_pos_z"], group["weight"])

        bb_cells = _bbtype_breakdown(group)

        rows.append({
            "pitch_name": pitch_name,
            "pitch_group": _pitch_group(pitch_name) or "Other",
            "pitches_w": n_pitches_w,
            "pa_w": n_pa_w,
            "bbe_w": n_bbe_w,
            "K_w": n_k_w, "BB_w": n_bb_w, "HBP_w": n_hbp_w,
            "1B_w": n_1b_w, "2B_w": n_2b_w, "3B_w": n_3b_w, "HR_w": n_hr_w,
            "K_pct": k_pct,
            "Whiff_pct": whiff_pct,
            "HardHit_pct": hh_pct,
            "xwOBA": xwoba, "xBA": xba, "xSLG": xslg,
            "velo": velo_w, "ivb": ivb_w, "hb_in": hb_in_w,
            "spin": spin_w, "ext": ext_w, "spin_axis": spin_axis_w,
            "rel_x": rel_x_w, "rel_z": rel_z_w,
            **bb_cells,
        })

    return pd.DataFrame(rows).sort_values("pitches_w", ascending=False).reset_index(drop=True)


def overall_rates(df: pd.DataFrame) -> dict:
    """Weighted overall K%/BB%/HBP%/Whiff%/HardHit%/Chase%/xwOBA/xBA/xSLG."""
    pa = df[df["events"].notna()]
    n_pa_w = float(pa["weight"].sum())

    def _w_count(events_subset):
        mask = pa["events"].isin(events_subset)
        return float((mask.astype(float) * pa["weight"]).sum())

    k = _w_count(K_EVENTS) / n_pa_w if n_pa_w else 0.0
    bb = _w_count({"walk"}) / n_pa_w if n_pa_w else 0.0
    hbp = _w_count({"hit_by_pitch"}) / n_pa_w if n_pa_w else 0.0

    swings = df["description"].isin(SWING_DESCRIPTIONS)
    whiffs = df["description"].isin(WHIFF_DESCRIPTIONS)
    n_sw = float((swings.astype(float) * df["weight"]).sum())
    whiff = float((whiffs.astype(float) * df["weight"]).sum()) / n_sw if n_sw else 0.0

    ooz = df["zone"].isin(OOZ_ZONE)
    n_ooz = float((ooz.astype(float) * df["weight"]).sum())
    chase = float(((swings & ooz).astype(float) * df["weight"]).sum()) / n_ooz if n_ooz else 0.0

    bbe = df[(df["type"] == "X") & df["launch_speed"].notna()]
    n_bbe_w = float(bbe["weight"].sum())
    hh = (float(((bbe["launch_speed"] >= 95).astype(float) * bbe["weight"]).sum()) / n_bbe_w
          if n_bbe_w else 0.0)
    barrel = (float(((bbe["launch_speed_angle"] == 6).astype(float) * bbe["weight"]).sum()) / n_bbe_w
              if n_bbe_w else 0.0)

    bbe_typed = df[(df["type"] == "X") & df["bb_type"].notna()]
    n_bbet_w = float(bbe_typed["weight"].sum())
    gb = (float(((bbe_typed["bb_type"] == "ground_ball").astype(float) * bbe_typed["weight"]).sum())
          / n_bbet_w if n_bbet_w else 0.0)
    air_mask = bbe_typed["bb_type"].isin(["fly_ball", "line_drive", "popup"])
    air = (float((air_mask.astype(float) * bbe_typed["weight"]).sum()) / n_bbet_w
           if n_bbet_w else 0.0)

    overall_bb = _bbtype_breakdown(df)

    return {
        "n_pa_w": n_pa_w, "n_pitches_w": float(df["weight"].sum()),
        "K_pct": k, "BB_pct": bb, "HBP_pct": hbp,
        "Whiff_pct": whiff, "Chase_pct": chase,
        "HardHit_pct": hh, "Barrel_pct": barrel,
        "GB_pct": gb, "Air_pct": air,
        "xwOBA": w_xwoba(df), "xBA": w_xba_xslg(df)[0], "xSLG": w_xba_xslg(df)[1],
        # Per-bb_type fallbacks (BBE-only contact means across all pitch types):
        # bbe_<b>_w, share_<b>, xwoba_<b>, xba_<b>, xslg_<b>.
        **overall_bb,
    }


# ---------- bb_type-aware projection helpers ------------------------------

# League BBE bb_type share mix, calibrated alongside LG_*_BY_BBTYPE. Used as
# the final fallback when both per-pitch and overall samples are too thin to
# infer a side's bb_type distribution, and as the reference "own" mix when
# computing the league's contact xwOBA in the bb_type-aware shift.
LG_BBTYPE_SHARES = {
    "ground_ball": 0.420,
    "line_drive":  0.243,
    "fly_ball":    0.269,
    "popup":       0.067,
}
LG_BBTYPE_DISPLAY = {
    "ground_ball": "GB",
    "line_drive":  "LD",
    "fly_ball":    "FB",
    "popup":       "PU",
}


def _bbtype_share_dict(row_or_overall: dict) -> tuple[dict, float]:
    """Extract per-bb_type BBE share dict and the BBE total weight.

    Empty / missing cells yield 0, total may be 0 if the row has no BBE data.
    """
    sh: dict[str, float] = {}
    total = 0.0
    for b in BB_TYPES:
        w = float(row_or_overall.get(f"bbe_{b}_w", 0.0) or 0.0)
        sh[b] = w
        total += w
    if total > 0:
        return {b: sh[b] / total for b in BB_TYPES}, total
    return {b: float("nan") for b in BB_TYPES}, 0.0


def _resolve_bbtype_shares(pitch_row: dict | None, side_overall: dict) -> dict:
    """Per-pitch -> overall -> league fallback chain for a side's bb_type mix."""
    if pitch_row is not None:
        sh, total = _bbtype_share_dict(pitch_row)
        if total >= BBTYPE_MIN_BBE_PER_PITCH:
            return sh
    sh, total = _bbtype_share_dict(side_overall)
    if total >= BBTYPE_MIN_BBE_OVERALL:
        return sh
    return dict(LG_BBTYPE_SHARES)


def _resolve_batter_bbtype_means(pitch_row: dict | None, batter_overall: dict,
                                  prefix: str, lg_by_bb: dict) -> dict:
    """Per-pitch -> overall -> league fallback chain for batter per-bb_type means.

    `prefix` is one of 'xwoba', 'xba', 'xslg'. Returns one value per bb_type.
    """
    out: dict[str, float] = {}
    for b in BB_TYPES:
        if pitch_row is not None:
            n = float(pitch_row.get(f"bbe_{b}_w", 0.0) or 0.0)
            v = pitch_row.get(f"{prefix}_{b}", float("nan"))
            v = float("nan") if v is None else float(v)
            if n >= BBTYPE_MIN_BBE_PER_PITCH and not math.isnan(v):
                out[b] = v
                continue
        n_ov = float(batter_overall.get(f"bbe_{b}_w", 0.0) or 0.0)
        v_ov = batter_overall.get(f"{prefix}_{b}", float("nan"))
        v_ov = float("nan") if v_ov is None else float(v_ov)
        if n_ov >= BBTYPE_MIN_BBE_OVERALL and not math.isnan(v_ov):
            out[b] = v_ov
            continue
        out[b] = lg_by_bb[b]
    return out


def _bbtype_weighted(means: dict, shares: dict) -> float:
    """sum_b [shares_b * means_b], skipping NaN cells, renormalizing weights."""
    num = 0.0
    tot = 0.0
    for b in BB_TYPES:
        m = means.get(b, float("nan"))
        s = shares.get(b, float("nan"))
        if math.isnan(m) or math.isnan(s):
            continue
        num += s * m
        tot += s
    return num / tot if tot > 0 else float("nan")


def _bbtype_perpa_shift(pitch_row: dict | None,
                         batter_overall: dict,
                         pit_shares: dict,
                         bat_shares: dict,
                         bbe_per_pa: float,
                         prefix: str,
                         lg_by_bb: dict) -> float:
    """Per-PA shift for one metric (xwoba/xba/xslg) on one pitch.

    Difference between the batter's BBE-only contact mean weighted by the
    pitcher's induced bb_type mix vs the batter's own mix, scaled by the
    BBE-per-PA share. Zero when the two mixes match or sample is too thin.
    """
    means = _resolve_batter_bbtype_means(pitch_row, batter_overall, prefix, lg_by_bb)
    own = _bbtype_weighted(means, bat_shares)
    adj = _bbtype_weighted(means, pit_shares)
    if math.isnan(own) or math.isnan(adj):
        return 0.0
    return (adj - own) * bbe_per_pa


# ---------- Layer 1: count-conditional pitch mix + projection ------------

def count_conditional_marginal(pit_vs_bat: pd.DataFrame, bat_vs_pit: pd.DataFrame) -> pd.Series:
    """Marginal pitch usage = sum_c P(c | batter) * P(pitch | c, pitcher).

    Falls back to flat pitcher usage if either side has no count-state data.
    """
    pit = pit_vs_bat.dropna(subset=["pitch_name", "count_state"])
    bat = bat_vs_pit.dropna(subset=["count_state"])

    if pit.empty:
        return pd.Series(dtype=float)

    if bat.empty:
        flat = pit.groupby("pitch_name")["weight"].sum()
        return flat / flat.sum()

    # P(c) from batter: weighted share of UNIQUE PAs that reached count-state c.
    # Previously summed `weight` over every pitch in each count, which
    # over-weighted late counts (e.g. 2-strike fouls multiply pitches in the
    # same state). Dedupe by (game_pk, at_bat_number, count_state) so each
    # state visit counts once per PA.
    if {"game_pk", "at_bat_number"}.issubset(bat.columns):
        bat_pa = bat.drop_duplicates(["game_pk", "at_bat_number", "count_state"])
    else:
        bat_pa = bat  # fallback if those columns aren't present in the cache
    bat_count_w = bat_pa.groupby("count_state")["weight"].sum()
    bat_count_p = bat_count_w / bat_count_w.sum()

    # P(pitch | c) from pitcher.
    pit_pivot = (pit.groupby(["count_state", "pitch_name"])["weight"]
                    .sum()
                    .unstack(fill_value=0.0))
    row_sums = pit_pivot.sum(axis=1)
    pit_pivot = pit_pivot.div(row_sums.where(row_sums > 0, 1.0), axis=0)

    # Restrict batter count distribution to counts the pitcher has thrown in.
    common = pit_pivot.index.intersection(bat_count_p.index)
    if common.empty:
        flat = pit.groupby("pitch_name")["weight"].sum()
        return flat / flat.sum()

    pit_pivot = pit_pivot.loc[common]
    weights = bat_count_p.loc[common]
    weights = weights / weights.sum()

    marginal = pit_pivot.T @ weights
    s = float(marginal.sum())
    return marginal / s if s else marginal


# ---------- Layer 1b: batter count-state blend (B1) -----------------------

# Min effective PA per count_state before we trust the batter's xwOBA there.
COUNT_XWOBA_MIN_PA_W = 5.0
# Damping factor for the count-state shift. The per-pitch metrics already
# implicitly average the batter's performance across the counts they faced
# each pitch in, so applying the full count-state delta would partly
# double-count. 0.5 lets the signal show up in the headline without dominating
# the per-pitch additive ladder.
COUNT_XWOBA_BLEND_ALPHA = 0.5


def batter_xwoba_by_count(bat_vs_pit: pd.DataFrame,
                          min_pa_w: float = COUNT_XWOBA_MIN_PA_W) -> dict[str, float]:
    """Per-count_state batter xwOBA, gated by min PA weight.

    Returns {} when there's no `count_state` column or no count meets the gate.
    """
    if bat_vs_pit.empty or "count_state" not in bat_vs_pit.columns:
        return {}
    out: dict[str, float] = {}
    for c, grp in bat_vs_pit.dropna(subset=["count_state"]).groupby("count_state"):
        pa = grp[grp["events"].notna()]
        n_pa_w = float(pa["weight"].sum())
        if n_pa_w < min_pa_w:
            continue
        x = w_xwoba(grp)
        if x is None or (isinstance(x, float) and math.isnan(x)):
            continue
        out[str(c)] = float(x)
    return out


def pitcher_count_state_distribution(pit_vs_bat: pd.DataFrame) -> dict[str, float]:
    """P(count_state) from the pitcher frame, PA-anchored to avoid the same
    over-weighting bug that A4 fixed in `count_conditional_marginal`.
    """
    if pit_vs_bat.empty or "count_state" not in pit_vs_bat.columns:
        return {}
    g = pit_vs_bat.dropna(subset=["count_state"])
    if {"game_pk", "at_bat_number"}.issubset(g.columns):
        g = g.drop_duplicates(["game_pk", "at_bat_number", "count_state"])
    w = g.groupby("count_state")["weight"].sum()
    s = float(w.sum())
    if s <= 0:
        return {}
    return {str(k): float(v) / s for k, v in w.items()}


def project(batter_pt: pd.DataFrame, pitcher_pt: pd.DataFrame,
            batter_overall: dict, pitcher_overall: dict,
            marginal: pd.Series,
            batter_count_xwoba: dict[str, float] | None = None,
            pitcher_count_dist: dict[str, float] | None = None) -> dict:
    """Headline projection: log5 for rates, additive for xwOBA/xBA/xSLG.

    The per-pitch xwOBA/xBA/xSLG additive is computed twice: a `_raw` version
    using the batter's career per-pitch metrics as-is, and an `_adj` version
    that shifts the batter's BBE-only contact contribution to use the
    pitcher's induced bb_type mix on that pitch (sample-size fallback chain:
    per-pitch -> overall-by-bbtype -> league baseline). The adjusted value is
    what feeds downstream layers; the raw value + delta are surfaced so the
    GB-mix discount is auditable in the report.
    """
    # Per-PA rates: log5 on overall numbers, not per-pitch.
    proj_k = log5(batter_overall["K_pct"], pitcher_overall["K_pct"], LG_K_PCT)
    proj_bb = log5(batter_overall["BB_pct"], pitcher_overall["BB_pct"], LG_BB_PCT)
    proj_hbp = log5(batter_overall["HBP_pct"], pitcher_overall["HBP_pct"], LG_HBP_PCT)

    # Guard against an empty / all-zero marginal: previously the per-pitch loop
    # silently contributed nothing while K/BB/HBP kept firing, leaving the
    # report with a 0.000 xwOBA next to plausible-looking rate stats. Fall
    # back to a flat pitcher pitch usage from `pitcher_pt`; if even that is
    # missing, synthesize a single sentinel-name row so the loop body's
    # existing "no per-pitch data -> use overall" branch produces an honest
    # one-pitch additive projection from the overall numbers.
    marginal_fallback: str | None = None
    if marginal is None or len(marginal) == 0 or float(np.nansum(marginal.values)) <= 0:
        if not pitcher_pt.empty and "pitches_w" in pitcher_pt.columns:
            w = pitcher_pt.set_index("pitch_name")["pitches_w"].astype(float)
            s = float(w.sum())
            if s > 0:
                marginal = (w / s)
                marginal_fallback = "flat_pitcher_pt"
        if marginal_fallback is None:
            # Sentinel name will miss both bat_idx and pit_idx, triggering the
            # overall-stats fallback inside the per-pitch loop.
            marginal = pd.Series({"(no per-pitch data)": 1.0}, dtype=float)
            marginal_fallback = "overall_only"

    # Per-pitch-type contributions, weighted by marginal pitcher usage.
    bat_idx = batter_pt.set_index("pitch_name") if not batter_pt.empty else pd.DataFrame()
    pit_idx = pitcher_pt.set_index("pitch_name") if not pitcher_pt.empty else pd.DataFrame()

    def _row_or_overall(row: dict | None, key: str, overall: dict) -> float:
        """Pull `key` from a per-pitch row, falling back to `overall[key]` when
        the per-pitch row is missing OR when its value is NaN/None (e.g. a
        pitch type with too thin a PA sample for that metric).
        """
        if row is not None and key in row:
            v = row[key]
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                return float(v)
        return float(overall[key])

    proj_xwoba = 0.0
    proj_xba = 0.0
    proj_xslg = 0.0
    proj_xwoba_raw = 0.0
    proj_xba_raw = 0.0
    proj_xslg_raw = 0.0
    proj_whiff = 0.0
    proj_hh = 0.0
    rows = []
    for pitch_name, p in marginal.items():
        bat_row = bat_idx.loc[pitch_name].to_dict() if pitch_name in bat_idx.index else None
        pit_row = pit_idx.loc[pitch_name].to_dict() if pitch_name in pit_idx.index else None

        b_x = _row_or_overall(bat_row, "xwOBA", batter_overall)
        p_x = _row_or_overall(pit_row, "xwOBA", pitcher_overall)
        b_a = _row_or_overall(bat_row, "xBA",   batter_overall)
        p_a = _row_or_overall(pit_row, "xBA",   pitcher_overall)
        b_s = _row_or_overall(bat_row, "xSLG",  batter_overall)
        p_s = _row_or_overall(pit_row, "xSLG",  pitcher_overall)
        b_w = _row_or_overall(bat_row, "Whiff_pct",   batter_overall)
        p_w = _row_or_overall(pit_row, "Whiff_pct",   pitcher_overall)
        b_h = _row_or_overall(bat_row, "HardHit_pct", batter_overall)
        p_h = _row_or_overall(pit_row, "HardHit_pct", pitcher_overall)

        # bb_type mix adjustment: shift batter's BBE contribution by the
        # difference between the pitcher's induced bb_type mix and the
        # batter's own career mix on this pitch.
        if bat_row:
            b_bbe_w = float(bat_row.get("bbe_w", 0.0) or 0.0)
            b_pa_w = float(bat_row.get("pa_w", 0.0) or 0.0)
        else:
            b_pa_w = float(batter_overall.get("n_pa_w", 0.0) or 0.0)
            b_bbe_w = sum(
                float(batter_overall.get(f"bbe_{b}_w", 0.0) or 0.0) for b in BB_TYPES
            )
        bbe_per_pa = (b_bbe_w / b_pa_w) if b_pa_w > 0 else 0.0
        pit_shares = _resolve_bbtype_shares(pit_row, pitcher_overall)
        bat_shares = _resolve_bbtype_shares(bat_row, batter_overall)

        sh_x = _bbtype_perpa_shift(bat_row, batter_overall, pit_shares, bat_shares,
                                    bbe_per_pa, "xwoba", LG_XWOBA_BY_BBTYPE)
        sh_a = _bbtype_perpa_shift(bat_row, batter_overall, pit_shares, bat_shares,
                                    bbe_per_pa, "xba",   LG_XBA_BY_BBTYPE)
        sh_s = _bbtype_perpa_shift(bat_row, batter_overall, pit_shares, bat_shares,
                                    bbe_per_pa, "xslg",  LG_XSLG_BY_BBTYPE)
        b_x_adj = b_x + sh_x
        b_a_adj = b_a + sh_a
        b_s_adj = b_s + sh_s

        m_xwoba_raw = additive(b_x, p_x, LG_XWOBA)
        m_xba_raw   = additive(b_a, p_a, LG_XBA)
        m_xslg_raw  = additive(b_s, p_s, LG_XSLG)
        m_xwoba = additive(b_x_adj, p_x, LG_XWOBA)
        m_xba   = additive(b_a_adj, p_a, LG_XBA)
        m_xslg  = additive(b_s_adj, p_s, LG_XSLG)
        m_whiff = log5(b_w, p_w, LG_WHIFF)
        m_hh = log5(b_h, p_h, LG_HARD_HIT)

        proj_xwoba += p * m_xwoba
        proj_xba   += p * m_xba
        proj_xslg  += p * m_xslg
        proj_xwoba_raw += p * m_xwoba_raw
        proj_xba_raw   += p * m_xba_raw
        proj_xslg_raw  += p * m_xslg_raw
        proj_whiff += p * m_whiff
        proj_hh    += p * m_hh

        bb_mix_str = " / ".join(
            f"{LG_BBTYPE_DISPLAY[b]} {pit_shares[b]*100:.0f}%" for b in BB_TYPES
        )

        rows.append({
            "Pitch": pitch_name,
            "Marginal Usage %": p * 100,
            "Batter xwOBA": b_x,
            "Pitcher xwOBA allowed": p_x,
            "Pitcher BB-mix": bb_mix_str,
            "Adj batter xwOBA": b_x_adj,
            "Adj delta (pts)": (m_xwoba - m_xwoba_raw) * 1000,
            "Projected xwOBA": m_xwoba,
            "Projected xBA": m_xba,
            "Projected xSLG": m_xslg,
            "Projected Whiff %": m_whiff * 100,
        })

    pitch_table = pd.DataFrame(rows).sort_values("Marginal Usage %", ascending=False).reset_index(drop=True)

    # B1: count-state batter blend. Compute the weighted shift in batter
    # xwOBA induced by the pitcher's count distribution (relative to the
    # batter's overall xwOBA), damped by COUNT_XWOBA_BLEND_ALPHA so we don't
    # double-count what the per-pitch numbers already capture. The shift is
    # added on top of the bb_type-adjusted projection.
    proj_xwoba_bbtype = proj_xwoba
    count_shift = 0.0
    count_breakdown: list[dict] = []
    if batter_count_xwoba and pitcher_count_dist:
        bat_x_overall = float(batter_overall.get("xwOBA", 0.0) or 0.0)
        if not (isinstance(bat_x_overall, float) and math.isnan(bat_x_overall)):
            common = sorted(set(batter_count_xwoba) & set(pitcher_count_dist))
            p_renorm = sum(pitcher_count_dist[c] for c in common)
            if common and p_renorm > 0:
                for c in common:
                    p_c = pitcher_count_dist[c] / p_renorm
                    delta_c = batter_count_xwoba[c] - bat_x_overall
                    count_breakdown.append({
                        "count_state": c,
                        "P(count) [pitcher]": p_c,
                        "Batter xwOBA": batter_count_xwoba[c],
                        "Delta vs overall": delta_c,
                    })
                count_shift = sum(
                    (pitcher_count_dist[c] / p_renorm)
                    * (batter_count_xwoba[c] - bat_x_overall)
                    for c in common
                ) * COUNT_XWOBA_BLEND_ALPHA
    proj_xwoba_final = proj_xwoba_bbtype + count_shift

    return {
        "K_pct": proj_k, "BB_pct": proj_bb, "HBP_pct": proj_hbp,
        "xwOBA": proj_xwoba_final, "xBA": proj_xba, "xSLG": proj_xslg,
        "xwOBA_raw": proj_xwoba_raw, "xBA_raw": proj_xba_raw, "xSLG_raw": proj_xslg_raw,
        "xwOBA_bbtype": proj_xwoba_bbtype,
        "xwOBA_adj_pts": (proj_xwoba_bbtype - proj_xwoba_raw) * 1000,
        "xwOBA_count_pts": count_shift * 1000,
        "count_blend": count_breakdown,
        "Whiff_pct": proj_whiff, "HardHit_pct": proj_hh,
        "pitch_table": pitch_table,
        "marginal": marginal,
        "marginal_fallback": marginal_fallback,
    }


# ---------- Layer 2: shape-aware comps ------------------------------------

def shape_comps(bat_vs_pit: pd.DataFrame, arsenal: pd.DataFrame) -> pd.DataFrame:
    """For each arsenal pitch, find batter's pitches with matching shape."""
    if bat_vs_pit.empty or arsenal.empty:
        return pd.DataFrame()

    rows = []
    bat = bat_vs_pit.dropna(subset=["effective_speed", "api_break_x_batter_in", "api_break_z_with_gravity"])

    for _, ar in arsenal.iterrows():
        if math.isnan(ar["velo"]) or math.isnan(ar["hb_in"]) or math.isnan(ar["ivb"]):
            continue
        group = ar["pitch_group"]
        sub = bat[bat["pitch_name"].map(_pitch_group) == group]

        def _filter(tol_mult: float) -> pd.DataFrame:
            v = SHAPE_TOL_VELO * tol_mult
            h = SHAPE_TOL_HB * tol_mult
            z = SHAPE_TOL_VB * tol_mult
            return sub[
                (sub["effective_speed"].sub(ar["velo"]).abs() <= v) &
                (sub["api_break_x_batter_in"].sub(ar["hb_in"]).abs() <= h) &
                (sub["api_break_z_with_gravity"].sub(ar["ivb"]).abs() <= z)
            ]

        comps = _filter(1.0)
        confidence = "—"
        eff_n = float(comps["weight"].sum())
        if eff_n >= 30:
            confidence = "high"
        elif eff_n >= 15:
            confidence = "medium"
        else:
            comps = _filter(SHAPE_FALLBACK)
            eff_n = float(comps["weight"].sum())
            confidence = "low" if eff_n > 0 else "no comps"

        if comps.empty:
            rows.append({
                "Pitch": ar["pitch_name"],
                "Shape (eff velo / IVB / HB-in)": f"{ar['velo']:.1f} / {ar['ivb']:+.1f} / {ar['hb_in']:+.1f}",
                "n comps (eff)": 0.0,
                "Whiff %": float("nan"),
                "xwOBA": float("nan"),
                "xBA": float("nan"),
                "xSLG": float("nan"),
                "Hard Hit %": float("nan"),
                "Confidence": confidence,
            })
            continue

        whiffs = comps["description"].isin(WHIFF_DESCRIPTIONS)
        swings = comps["description"].isin(SWING_DESCRIPTIONS)
        n_sw = float((swings.astype(float) * comps["weight"]).sum())
        whiff_pct = float((whiffs.astype(float) * comps["weight"]).sum()) / n_sw if n_sw else 0.0

        bbe = comps[(comps["type"] == "X") & comps["launch_speed"].notna()]
        n_bbe = float(bbe["weight"].sum())
        hh = (float(((bbe["launch_speed"] >= 95).astype(float) * bbe["weight"]).sum()) / n_bbe
              if n_bbe else 0.0)
        xba_c, xslg_c = w_xba_xslg(comps)

        rows.append({
            "Pitch": ar["pitch_name"],
            "Shape (eff velo / IVB / HB-in)": f"{ar['velo']:.1f} / {ar['ivb']:+.1f} / {ar['hb_in']:+.1f}",
            "n comps (eff)": eff_n,
            "Whiff %": whiff_pct * 100,
            "xwOBA": w_xwoba(comps),
            "xBA": xba_c,
            "xSLG": xslg_c,
            "Hard Hit %": hh * 100,
            "Confidence": confidence,
        })

    return pd.DataFrame(rows)


# ---------- Layer 3: zone overlay -----------------------------------------

def zone_overlay(pit_vs_bat: pd.DataFrame, bat_vs_pit: pd.DataFrame,
                 arsenal: pd.DataFrame) -> pd.DataFrame:
    """Per arsenal pitch: weighted intersection of attack-share x batter xwOBA per zone.

    For zones where the batter has no data, impute the league baseline xwOBA
    instead of dropping the zone (which used to silently renormalize the
    intersection over only-covered zones, making two batters with very
    different coverage maps non-comparable). The new "Coverage %" column
    reports the share of the pitcher's attack on this pitch that landed in
    zones where we actually have batter data.
    """
    if pit_vs_bat.empty or arsenal.empty:
        return pd.DataFrame()

    bat_zone_xwoba: dict = {}
    bat_zone_xba: dict = {}
    bat_zone_xslg: dict = {}
    for z in ALL_ZONES:
        cell = bat_vs_pit[bat_vs_pit["zone"] == z]
        if cell.empty or not cell["events"].notna().any():
            bat_zone_xwoba[z] = float("nan")
            bat_zone_xba[z] = float("nan")
            bat_zone_xslg[z] = float("nan")
            continue
        bat_zone_xwoba[z] = w_xwoba(cell)
        ba_z, slg_z = w_xba_xslg(cell)
        bat_zone_xba[z] = ba_z
        bat_zone_xslg[z] = slg_z

    rows = []
    for _, ar in arsenal.iterrows():
        sub = pit_vs_bat[pit_vs_bat["pitch_name"] == ar["pitch_name"]]
        if sub.empty:
            continue
        attack = (sub.groupby("zone")["weight"].sum()).reindex(ALL_ZONES, fill_value=0.0)
        total = attack.sum()
        if total == 0:
            continue
        attack_share = attack / total

        # Intersection metrics: sum_z (attack_z * batter_metric_z). Impute league
        # baselines on zones with no batter data so each score remains comparable
        # across batters with different coverage maps. Coverage % is shared
        # across all three metrics (same underlying mask).
        intersect_x = 0.0
        intersect_a = 0.0
        intersect_s = 0.0
        covered_share = 0.0
        for z in ALL_ZONES:
            share = float(attack_share[z])
            xw = bat_zone_xwoba[z]
            ba = bat_zone_xba[z]
            slg = bat_zone_xslg[z]
            if math.isnan(xw):
                intersect_x += share * LG_XWOBA
                intersect_a += share * LG_XBA
                intersect_s += share * LG_XSLG
            else:
                intersect_x += share * xw
                intersect_a += share * (ba if not math.isnan(ba) else LG_XBA)
                intersect_s += share * (slg if not math.isnan(slg) else LG_XSLG)
                covered_share += share

        in_zone_share = float(attack_share.loc[IN_ZONE].sum())
        # Top 3 zones by attack share for this pitch
        top_zones = attack_share.sort_values(ascending=False).head(3)
        top_zone_str = ", ".join(f"z{int(z)}={s*100:.0f}%" for z, s in top_zones.items() if s > 0)

        rows.append({
            "Pitch": ar["pitch_name"],
            "In-zone %": in_zone_share * 100,
            "Top zones": top_zone_str,
            "Intersection xwOBA": intersect_x,
            "Intersection xBA": intersect_a,
            "Intersection xSLG": intersect_s,
            "Coverage %": covered_share * 100,
        })

    return pd.DataFrame(rows)


# ---------- Layer 4: TTO curve --------------------------------------------

def tto_curve(pit_vs_bat: pd.DataFrame) -> pd.DataFrame:
    """xwOBA / xBA / xSLG / K% / Whiff% / HardHit% allowed by times through the order."""
    if pit_vs_bat.empty:
        return pd.DataFrame()

    rows = []
    for tto, group in pit_vs_bat.groupby("tto_bucket"):
        pa_rows = group[group["events"].notna()]
        n_pa_w = float(pa_rows["weight"].sum())
        if n_pa_w == 0:
            continue
        swings = group["description"].isin(SWING_DESCRIPTIONS)
        whiffs = group["description"].isin(WHIFF_DESCRIPTIONS)
        n_sw = float((swings.astype(float) * group["weight"]).sum())
        whiff = float((whiffs.astype(float) * group["weight"]).sum()) / n_sw if n_sw else 0.0
        bbe = group[(group["type"] == "X") & group["launch_speed"].notna()]
        n_bbe = float(bbe["weight"].sum())
        hh = (float(((bbe["launch_speed"] >= 95).astype(float) * bbe["weight"]).sum()) / n_bbe
              if n_bbe else 0.0)
        n_k_w = float((pa_rows["events"].isin(K_EVENTS).astype(float) * pa_rows["weight"]).sum())
        k_pct = n_k_w / n_pa_w if n_pa_w else 0.0
        xba_t, xslg_t = w_xba_xslg(group)
        rows.append({
            "TTO": int(tto),
            "PA (eff)": n_pa_w,
            "xwOBA allowed": w_xwoba(group),
            "xBA allowed": xba_t,
            "xSLG allowed": xslg_t,
            "K %": k_pct * 100,
            "Whiff %": whiff * 100,
            "Hard Hit %": hh * 100,
        })

    return pd.DataFrame(rows).sort_values("TTO").reset_index(drop=True)


def tto_projections(base_proj: dict, tto: pd.DataFrame, pitcher_overall: dict) -> pd.DataFrame:
    """Blend the headline projection with per-TTO pitcher deltas across all
    four metrics (xwOBA, K%, Whiff%, HardHit%). Previously only xwOBA actually
    moved across TTO rows; K%/Whiff%/HardHit% were copied flat from the
    headline, which made the report imply rate stats degraded across TTO when
    they were really pinned to PA1.
    """
    if tto.empty:
        return pd.DataFrame()
    base_pit_x = float(pitcher_overall.get("xwOBA", 0.0) or 0.0)
    base_pit_a = float(pitcher_overall.get("xBA", 0.0) or 0.0)
    base_pit_s = float(pitcher_overall.get("xSLG", 0.0) or 0.0)
    base_pit_k = float(pitcher_overall.get("K_pct", 0.0) or 0.0) * 100.0
    base_pit_whiff = float(pitcher_overall.get("Whiff_pct", 0.0) or 0.0) * 100.0
    base_pit_hh = float(pitcher_overall.get("HardHit_pct", 0.0) or 0.0) * 100.0

    def _shift(headline: float, allowed: float, base: float,
               lo: float, hi: float) -> float:
        if allowed is None or (isinstance(allowed, float) and math.isnan(allowed)):
            return float("nan")
        if base is None or (isinstance(base, float) and math.isnan(base)):
            return float("nan")
        return float(np.clip(headline + (float(allowed) - base), lo, hi))

    rows = []
    for _, r in tto.iterrows():
        rows.append({
            "TTO": int(r["TTO"]),
            "Projected xwOBA": _shift(base_proj["xwOBA"],
                                       r.get("xwOBA allowed"),
                                       base_pit_x, 0.0, 1.0),
            "Projected xBA": _shift(base_proj.get("xBA", float("nan")),
                                     r.get("xBA allowed"),
                                     base_pit_a, 0.0, 1.0),
            "Projected xSLG": _shift(base_proj.get("xSLG", float("nan")),
                                      r.get("xSLG allowed"),
                                      base_pit_s, 0.0, 4.0),
            "Projected K %": _shift(base_proj["K_pct"] * 100,
                                     r.get("K %"),
                                     base_pit_k, 0.0, 100.0),
            "Projected Whiff %": _shift(base_proj["Whiff_pct"] * 100,
                                         r.get("Whiff %"),
                                         base_pit_whiff, 0.0, 100.0),
            "Projected Hard Hit %": _shift(base_proj["HardHit_pct"] * 100,
                                            r.get("Hard Hit %"),
                                            base_pit_hh, 0.0, 100.0),
            "Sample (eff PA)": r["PA (eff)"],
        })
    return pd.DataFrame(rows)


# ---------- Layer 5: bat-tracking interaction -----------------------------

# Min effective swings on a pitch type before we trust per-pitch attack angle;
# below this we fall back to the batter's overall attack angle (asterisked).
MIN_SWINGS_PER_PITCH = 5.0

# Match-gap (attack_angle + VAA) interpretation thresholds, in degrees.
MATCH_THRESHOLD_DEG = 3.0


def _vaa_at_plate(vy0: float, vz0: float, ay: float, az: float) -> float:
    """Vertical approach angle at the front of home plate, from Statcast initial conditions.

    Statcast's vy0/vz0/ay/az are defined at y = 50 ft; the front of the plate is
    at y = 17/12 ft. Solve the quadratic for time-to-plate, then take the angle
    of the velocity vector. Returns degrees (negative for a descending pitch).
    """
    if any(math.isnan(x) for x in (vy0, vz0, ay, az)) or ay == 0:
        return float("nan")
    y_plate = 17.0 / 12.0
    # 50 + vy0*t + 0.5*ay*t² = y_plate  -->  0.5*ay*t² + vy0*t + (50 - y_plate) = 0
    a, b, c = 0.5 * ay, vy0, 50.0 - y_plate
    disc = b * b - 4 * a * c
    if disc < 0:
        return float("nan")
    # Pitch travels in -y, so vy0 < 0 and ay > 0; the smaller positive root is t-to-plate.
    sqrt_disc = math.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)
    candidates = [t for t in (t1, t2) if t > 0]
    if not candidates:
        return float("nan")
    t = min(candidates)
    vy_plate = vy0 + ay * t
    vz_plate = vz0 + az * t
    if vy_plate >= 0:
        return float("nan")
    return math.degrees(math.atan2(vz_plate, -vy_plate))


def _pitcher_vaa_per_pitch(pit_vs_bat: pd.DataFrame) -> dict[str, float]:
    """Weighted-mean VAA at the plate, per pitch_name."""
    needed = ["pitch_name", "vy0", "vz0", "ay", "az", "weight"]
    df = pit_vs_bat.dropna(subset=needed)
    if df.empty:
        return {}
    df = df.copy()
    df["vaa"] = [
        _vaa_at_plate(vy, vz, ay, az)
        for vy, vz, ay, az in zip(df["vy0"], df["vz0"], df["ay"], df["az"])
    ]
    df = df.dropna(subset=["vaa"])
    if df.empty:
        return {}
    out: dict[str, float] = {}
    for pitch_name, group in df.groupby("pitch_name"):
        out[pitch_name] = w_mean(group["vaa"], group["weight"])
    return out


def _batter_attack_per_pitch(bat_vs_pit: pd.DataFrame) -> pd.DataFrame:
    """Per-pitch-name weighted attack angle + effective sample (swings)."""
    needed = ["pitch_name", "attack_angle", "weight"]
    df = bat_vs_pit.dropna(subset=needed)
    if df.empty:
        return pd.DataFrame(columns=["pitch_name", "attack_angle", "eff_n"])
    rows = []
    for pitch_name, group in df.groupby("pitch_name"):
        rows.append({
            "pitch_name": pitch_name,
            "attack_angle": w_mean(group["attack_angle"], group["weight"]),
            "eff_n": float(group["weight"].sum()),
        })
    return pd.DataFrame(rows)


def _match_gap_note(gap: float) -> str:
    if math.isnan(gap):
        return ""
    if abs(gap) <= MATCH_THRESHOLD_DEG:
        return "on plane"
    if gap > 0:
        return "swing steeper than pitch — pop-up / swing-under risk"
    return "swing flatter than pitch — topped / grounder risk"


def bat_tracking(
    bat_vs_pit: pd.DataFrame,
    pit_vs_bat: pd.DataFrame,
    arsenal: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    """Per-arsenal-pitch swing-vs-pitch-plane match using real VAA and
    per-pitch-type batter attack angle (with overall fallback when the
    pitch-type sample is thin)."""
    if bat_vs_pit.empty:
        return {}, pd.DataFrame()

    bat_track = bat_vs_pit.dropna(subset=["attack_angle"])
    if bat_track.empty:
        return {"attack_angle": None, "swing_length": None, "bat_speed": None}, pd.DataFrame()

    overall = {
        "attack_angle": w_mean(bat_track["attack_angle"], bat_track["weight"]),
        "swing_length": w_mean(bat_track["swing_length"], bat_track["weight"]),
        "bat_speed": w_mean(bat_track["bat_speed"], bat_track["weight"]),
    }
    overall_attack = overall["attack_angle"]

    vaa_by_pitch = _pitcher_vaa_per_pitch(pit_vs_bat)
    bat_per_pitch = _batter_attack_per_pitch(bat_vs_pit)
    bat_lookup = {r["pitch_name"]: r for _, r in bat_per_pitch.iterrows()}

    rows = []
    for _, ar in arsenal.iterrows():
        pitch_name = ar["pitch_name"]
        vaa = vaa_by_pitch.get(pitch_name, float("nan"))
        if math.isnan(vaa):
            continue

        bat_row = bat_lookup.get(pitch_name)
        if bat_row is not None and bat_row["eff_n"] >= MIN_SWINGS_PER_PITCH:
            attack = float(bat_row["attack_angle"])
            eff_n = float(bat_row["eff_n"])
            fallback = False
        else:
            attack = overall_attack if overall_attack is not None else float("nan")
            eff_n = float(bat_row["eff_n"]) if bat_row is not None else 0.0
            fallback = True

        if math.isnan(attack):
            gap = float("nan")
        else:
            # Pitch plane is VAA (negative, descending). Bat plane is +attack_angle.
            # The signed gap = attack + VAA tells you whether the swing is
            # steeper (positive gap) or flatter (negative gap) than the pitch.
            gap = attack + vaa

        rows.append({
            "Pitch": pitch_name,
            "VAA (deg)": vaa,
            "Bat attack (deg)": attack,
            "Swings (n)": eff_n,
            "Fallback": fallback,
            "Match gap (deg)": gap,
            "Note": _match_gap_note(gap),
        })

    return overall, pd.DataFrame(rows)


# ---------- Layer 6: 1st-pitch + two-strike sub-profiles ------------------

def count_subprofiles(bat_vs_pit: pd.DataFrame, pit_vs_bat: pd.DataFrame) -> dict:
    """First-pitch tendencies and two-strike profile."""
    out = {}

    # First pitch
    fp_bat = bat_vs_pit[bat_vs_pit["pitch_number"] == 1]
    fp_pit = pit_vs_bat[pit_vs_bat["pitch_number"] == 1]
    if not fp_bat.empty and not fp_pit.empty:
        bat_swing = w_rate(fp_bat["description"].isin(SWING_DESCRIPTIONS), fp_bat["weight"])
        # First-pitch strike: any of called strike, swinging strike, foul, or in-play (description)
        strike_descs = {"called_strike", "swinging_strike", "swinging_strike_blocked",
                        "foul", "foul_tip", "hit_into_play"}
        pit_strike = w_rate(fp_pit["description"].isin(strike_descs), fp_pit["weight"])
        out["first_pitch"] = {
            "batter_swing_pct": bat_swing,
            "pitcher_strike_pct": pit_strike,
            "batter_xwoba_on_swing": w_xwoba(fp_bat[fp_bat["description"].isin(SWING_DESCRIPTIONS) & fp_bat["events"].notna()]),
        }

    # Two strikes
    ts_bat = bat_vs_pit[bat_vs_pit["strikes"] == 2]
    ts_pit = pit_vs_bat[pit_vs_bat["strikes"] == 2]
    if not ts_bat.empty and not ts_pit.empty:
        ts_bat_pa = ts_bat[ts_bat["events"].notna()]
        n_bat_pa_w = float(ts_bat_pa["weight"].sum())
        bat_k_w = float(
            (ts_bat_pa["events"].isin(K_EVENTS).astype(float) * ts_bat_pa["weight"]).sum()
        )
        # Pitcher putaway% must be computed entirely inside the pitcher frame:
        # (pitcher's two-strike strikeouts) / (pitcher's two-strike PA-ending events).
        # Previously this divided the BATTER's K count by the pitcher's PA weight,
        # which produced unbounded values like 429%.
        ts_pit_pa = ts_pit[ts_pit["events"].notna()]
        n_pit_pa_w = float(ts_pit_pa["weight"].sum())
        pit_k_w = float(
            (ts_pit_pa["events"].isin(K_EVENTS).astype(float) * ts_pit_pa["weight"]).sum()
        )
        # Pitcher's two-strike pitch mix
        ts_mix = (ts_pit.dropna(subset=["pitch_name"])
                        .groupby("pitch_name")["weight"].sum())
        ts_mix = (ts_mix / ts_mix.sum() * 100).round(1).sort_values(ascending=False).head(5)
        out["two_strike"] = {
            "batter_K_pct_2s": bat_k_w / n_bat_pa_w if n_bat_pa_w else 0.0,
            "batter_xwoba_2s": w_xwoba(ts_bat),
            "batter_chase_2s": w_rate(ts_bat["description"].isin(SWING_DESCRIPTIONS) & ts_bat["zone"].isin(OOZ_ZONE),
                                       ts_bat["weight"]) /
                                max(w_rate(ts_bat["zone"].isin(OOZ_ZONE), ts_bat["weight"]), 1e-9)
                                if w_rate(ts_bat["zone"].isin(OOZ_ZONE), ts_bat["weight"]) else 0.0,
            "pitcher_xwoba_2s": w_xwoba(ts_pit),
            "pitcher_putaway_pct": pit_k_w / n_pit_pa_w if n_pit_pa_w else 0.0,
            "two_strike_mix": ts_mix.to_dict(),
        }
    return out


# ---------- Layer 7: discipline panel -------------------------------------

def discipline_panel(b: dict, p: dict) -> pd.DataFrame:
    rows = [
        ("Chase %", b["Chase_pct"] * 100, p["Chase_pct"] * 100),
        ("Whiff %", b["Whiff_pct"] * 100, p["Whiff_pct"] * 100),
        ("K %", b["K_pct"] * 100, p["K_pct"] * 100),
        ("BB %", b["BB_pct"] * 100, p["BB_pct"] * 100),
        ("Hard Hit %", b["HardHit_pct"] * 100, p["HardHit_pct"] * 100),
        ("Barrel %", b["Barrel_pct"] * 100, p["Barrel_pct"] * 100),
        ("GB %", b["GB_pct"] * 100, p["GB_pct"] * 100),
        ("Air %", b["Air_pct"] * 100, p["Air_pct"] * 100),
        ("xwOBA", b["xwOBA"], p["xwOBA"]),
        ("xBA", b.get("xBA", float("nan")), p.get("xBA", float("nan"))),
        ("xSLG", b.get("xSLG", float("nan")), p.get("xSLG", float("nan"))),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Batter (vs same hand)", "Pitcher (vs same hand)"])


def discipline_notes(b: dict, p: dict) -> list[str]:
    notes = []
    if p["GB_pct"] > 0.50 and b["Air_pct"] > 0.60:
        notes.append("GB-heavy pitcher vs air-ball hitter — pitcher edge in BABIP suppression on contact.")
    if p["Whiff_pct"] > 0.27 and b["Whiff_pct"] > 0.27:
        notes.append("High-whiff pitcher meets a swing-and-miss prone hitter — strikeout odds elevated.")
    if p["Whiff_pct"] < 0.20 and b["Chase_pct"] < 0.25:
        notes.append(
            "Low-whiff pitcher meets a disciplined hitter — neither side wins the "
            "swing-and-miss battle, expect strikes in the zone and BIP-heavy ABs."
        )
    if b["HardHit_pct"] > 0.50 and p["HardHit_pct"] < 0.32:
        notes.append(
            "Elite hard-hit batter vs contact-suppressing pitcher — tension on quality "
            "of contact: leans pitcher in expectation but watch for barrels when the "
            "batter does square one up."
        )
    return notes


# ---------- Layer 7b: recent form (rolling-window override) ---------------

def recent_form_snapshot(df_pristine: pd.DataFrame, window_days: int | None,
                         min_pa_w: float = RECENT_FORM_MIN_PA,
                         ref_date: date | None = None) -> dict | None:
    """Slice the pristine (pre-recency-decay) blended frame to current-season
    rows in the last `window_days` and compute headline rates on a flat
    (weight=1) basis. Pass `window_days=None` for the year-to-date slice.

    Returns None if effective PA falls below `min_pa_w` so the panel can
    render a clean dash for thin samples (typical for relievers in 14d).
    """
    if df_pristine is None or df_pristine.empty or "game_date" not in df_pristine.columns:
        return None
    if ref_date is None:
        ref_date = _recency_reference_date()
    gd = df_pristine["game_date"]
    if not pd.api.types.is_datetime64_any_dtype(gd):
        gd = pd.to_datetime(gd, errors="coerce")
    year_mask = gd.dt.year.eq(ref_date.year)
    if window_days is None:
        mask = year_mask
    else:
        cutoff = pd.Timestamp(ref_date) - pd.Timedelta(days=int(window_days))
        mask = gd.ge(cutoff) & year_mask
    sub = df_pristine.loc[mask].copy()
    if sub.empty:
        return None
    # Use a flat weight so the snapshot is independent of season cascade and
    # recency decay; the panel's job is to show raw recent-window rates.
    sub["weight"] = 1.0
    pa = sub[sub["events"].notna()]
    n_pa_w = float(pa["weight"].sum())
    if n_pa_w < min_pa_w:
        return None
    n_k = float((pa["events"].isin(K_EVENTS).astype(float) * pa["weight"]).sum())
    n_bb = float((pa["events"].eq("walk").astype(float) * pa["weight"]).sum())
    swings = sub["description"].isin(SWING_DESCRIPTIONS)
    whiffs = sub["description"].isin(WHIFF_DESCRIPTIONS)
    n_sw = float((swings.astype(float) * sub["weight"]).sum())
    whiff = float((whiffs.astype(float) * sub["weight"]).sum()) / n_sw if n_sw else 0.0
    bbe = sub[(sub["type"] == "X") & sub["launch_speed"].notna()]
    n_bbe_w = float(bbe["weight"].sum())
    hh = (float(((bbe["launch_speed"] >= 95).astype(float) * bbe["weight"]).sum()) / n_bbe_w
          if n_bbe_w else 0.0)
    xba, xslg = w_xba_xslg(sub)
    return {
        "window_days": (int(window_days) if window_days is not None else None),
        "n_pa": n_pa_w,
        "xwOBA": w_xwoba(sub),
        "xBA": xba,
        "xSLG": xslg,
        "K_pct": (n_k / n_pa_w) if n_pa_w else 0.0,
        "BB_pct": (n_bb / n_pa_w) if n_pa_w else 0.0,
        "Whiff_pct": whiff,
        "HardHit_pct": hh,
    }


def _build_recent_form_view(bat_pristine: pd.DataFrame, pit_pristine: pd.DataFrame,
                            windows: tuple[int, ...],
                            ref_date: date | None) -> dict:
    """Compute one side of the recent-form panel (either overall or platoon)."""
    out = {
        "batter": {"season": recent_form_snapshot(bat_pristine, window_days=None,
                                                    min_pa_w=1.0, ref_date=ref_date)},
        "pitcher": {"season": recent_form_snapshot(pit_pristine, window_days=None,
                                                     min_pa_w=1.0, ref_date=ref_date)},
    }
    for w in windows:
        out["batter"][f"d{int(w)}"] = recent_form_snapshot(bat_pristine, w, ref_date=ref_date)
        out["pitcher"][f"d{int(w)}"] = recent_form_snapshot(pit_pristine, w, ref_date=ref_date)
    return out


def recent_form_panel(bat_pristine: pd.DataFrame, pit_pristine: pd.DataFrame,
                      batter_meta: dict | None = None,
                      pitcher_meta: dict | None = None,
                      windows: tuple[int, ...] = RECENT_FORM_WINDOWS,
                      ref_date: date | None = None) -> dict:
    """Build the side-by-side recent-form readout for the report.

    Returns two parallel views over current-season, flat-weight, pre-decay rows:

    * `overall`  — no platoon filter; matches BR / Savant year-to-date totals
                   and is the natural read on "how is this player going right now".
    * `platoon`  — restricted to the matchup hand (batter vs `pitcher_meta.p_throws`,
                   pitcher vs `batter_meta.stand`), so it's apples-to-apples with
                   the rest of the report.

    Both views share the same row schema (season + each rolling window).
    """
    overall = _build_recent_form_view(bat_pristine, pit_pristine, windows, ref_date)

    p_throws = (pitcher_meta or {}).get("p_throws")
    stand = (batter_meta or {}).get("stand")
    bat_pl = (bat_pristine[bat_pristine["p_throws"] == p_throws].copy()
              if p_throws and "p_throws" in bat_pristine.columns else bat_pristine.iloc[0:0].copy())
    pit_pl = (pit_pristine[pit_pristine["stand"] == stand].copy()
              if stand and "stand" in pit_pristine.columns else pit_pristine.iloc[0:0].copy())
    platoon = _build_recent_form_view(bat_pl, pit_pl, windows, ref_date)

    return {
        "windows": tuple(int(w) for w in windows),
        "overall": overall,
        "platoon": platoon,
        "platoon_labels": {
            "batter": (f"vs {p_throws}HP" if p_throws else None),
            "pitcher": (f"vs {stand}HB" if stand else None),
        },
    }


def recent_form_summary(form: dict | None) -> dict | None:
    """Distill the panel's overall (un-platoon-filtered) view down to the
    deltas the narrative needs. Anchored on the overall view because it's the
    natural "how is this player going right now" read.
    """
    if not form:
        return None
    overall = form.get("overall") or form  # tolerate legacy shape
    def _delta(side_key: str) -> float | None:
        side = overall.get(side_key) or {}
        season = (side.get("season") or {}).get("xwOBA")
        snap = side.get("d14")
        if not snap or season is None or math.isnan(season) or math.isnan(snap.get("xwOBA", float("nan"))):
            return None
        return float(snap["xwOBA"]) - float(season)
    bat = _delta("batter")
    pit = _delta("pitcher")
    return {
        "batter_d14_xwoba": (overall.get("batter") or {}).get("d14", {}).get("xwOBA")
            if (overall.get("batter") or {}).get("d14") else None,
        "pitcher_d14_xwoba": (overall.get("pitcher") or {}).get("d14", {}).get("xwOBA")
            if (overall.get("pitcher") or {}).get("d14") else None,
        "batter_d14_xwoba_delta": bat,
        "pitcher_d14_xwoba_delta": pit,
        "batter_d14_n_pa": (overall.get("batter") or {}).get("d14", {}).get("n_pa")
            if (overall.get("batter") or {}).get("d14") else None,
        "pitcher_d14_n_pa": (overall.get("pitcher") or {}).get("d14", {}).get("n_pa")
            if (overall.get("pitcher") or {}).get("d14") else None,
    }


# ---------- Layer 8: edge analysis ----------------------------------------

def edge_analysis(batter_pt: pd.DataFrame, pitcher_pt: pd.DataFrame,
                  marginal: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    if batter_pt.empty or pitcher_pt.empty or marginal.empty:
        return pd.DataFrame(), pd.DataFrame()

    bat_idx = batter_pt.set_index("pitch_name")
    pit_idx = pitcher_pt.set_index("pitch_name")

    def _cell(idx, name, col):
        if name not in idx.index:
            return float("nan")
        v = idx.loc[name, col]
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")

    rows = []
    for pitch_name, p in marginal.items():
        b_x = _cell(bat_idx, pitch_name, "xwOBA")
        pi_x = _cell(pit_idx, pitch_name, "xwOBA")
        if math.isnan(b_x) or math.isnan(pi_x):
            continue
        b_a = _cell(bat_idx, pitch_name, "xBA")
        pi_a = _cell(pit_idx, pitch_name, "xBA")
        b_s = _cell(bat_idx, pitch_name, "xSLG")
        pi_s = _cell(pit_idx, pitch_name, "xSLG")
        # Use the additive(b, p, lg) projection's distance from league as the edge
        # metric. Positive = the per-pitch matchup pulls projection above league
        # (batter edge); negative = pulls it below (pitcher edge). Weight by usage
        # so a low-usage pitch with a huge edge doesn't outrank a primary pitch.
        # Edge ranking is anchored on xwOBA; xBA / xSLG are surfaced for context.
        proj_x = additive(b_x, pi_x, LG_XWOBA)
        edge = proj_x - LG_XWOBA
        proj_a = (additive(b_a, pi_a, LG_XBA)
                  if not (math.isnan(b_a) or math.isnan(pi_a)) else float("nan"))
        proj_s = (additive(b_s, pi_s, LG_XSLG)
                  if not (math.isnan(b_s) or math.isnan(pi_s)) else float("nan"))
        rows.append({
            "Pitch": pitch_name,
            "Usage %": p * 100,
            "Batter xwOBA": b_x,
            "Pitcher xwOBA allowed": pi_x,
            "Projected xwOBA": proj_x,
            "Projected xBA": proj_a,
            "Projected xSLG": proj_s,
            "Edge (pts)": edge * 1000,
            "edge_score": p * edge,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df, df

    # Sign-filter each table so we don't manufacture an "edge" out of pitches
    # that are actually neutral or favor the other side.
    bat_fav = (df[df["edge_score"] > 0]
               .sort_values("edge_score", ascending=False)
               .head(3).reset_index(drop=True))
    pit_fav = (df[df["edge_score"] < 0]
               .sort_values("edge_score", ascending=True)
               .head(3).reset_index(drop=True))
    return bat_fav, pit_fav


# ---------- Layer 9: deception, spin axis, handedness ---------------------

def deception(arsenal: pd.DataFrame) -> dict:
    if arsenal.empty:
        return {}
    rel = arsenal.dropna(subset=["rel_x", "rel_z"])
    if len(rel) < 2:
        return {"cluster_stdev_in": 0.0, "per_pitch": pd.DataFrame()}

    # Convert release feet to inches; centroid + per-pitch deviation.
    cx = rel["rel_x"].mean()
    cz = rel["rel_z"].mean()
    devs = []
    for _, r in rel.iterrows():
        d_in = math.sqrt((r["rel_x"] - cx) ** 2 + (r["rel_z"] - cz) ** 2) * 12.0
        devs.append({"Pitch": r["pitch_name"], "Δ from centroid (in)": d_in,
                     "Spin axis (deg)": r["spin_axis"]})
    per_pitch = pd.DataFrame(devs).sort_values("Δ from centroid (in)").reset_index(drop=True)
    cluster_in = math.sqrt(((rel["rel_x"] - cx) ** 2 + (rel["rel_z"] - cz) ** 2).mean()) * 12.0

    deception_label = "tight (deceptive)" if cluster_in < 1.5 else "loose" if cluster_in > 3.5 else "moderate"
    return {
        "cluster_stdev_in": cluster_in,
        "deception_label": deception_label,
        "per_pitch": per_pitch,
    }


def handedness_verdict(stand: str, p_throws: str) -> str:
    if not stand or not p_throws:
        return ""
    if stand == p_throws:
        return f"{stand}HB vs {p_throws}HP — same-side platoon disadvantage to the hitter."
    return f"{stand}HB vs {p_throws}HP — opposite-side platoon advantage to the hitter."


# ---------- Layer 10: per-PA outcome distribution -------------------------

# ---------- Layer 11: contact-quality projection -------------------------

# Statcast quality-of-contact buckets (`launch_speed_angle`, 1-6).
QOC_LABELS = {
    1: "Weak",
    2: "Topped",
    3: "Under",
    4: "Flare/Burner",
    5: "Solid",
    6: "Barrel",
}

# Exit-velocity histogram bin edges (mph).
EV_BINS = [-float("inf"), 70.0, 80.0, 87.5, 92.5, 95.0, 100.0, 105.0, float("inf")]
EV_LABELS = ["<70", "70-80", "80-87", "87-92", "92-95", "95-100", "100-105", "105+"]

# Launch-angle histogram bin edges (degrees).
LA_BINS = [-float("inf"), -10.0, 0.0, 10.0, 19.0, 26.0, 35.0, 50.0, float("inf")]
LA_LABELS = ["<-10\u00b0", "-10-0\u00b0", "0-10\u00b0", "10-19\u00b0",
             "19-26\u00b0", "26-35\u00b0", "35-50\u00b0", "50\u00b0+"]
# Indices in LA_BINS marked as the launch-angle "barrel zone" in the report.
# We highlight the 10-19 and 19-26 bins (indices 3 and 4); these are the
# narrowest standard cut that lines up cleanly with our bin edges. Statcast's
# broader 8-32 deg "sweet spot" leaks into bins 2 and 5, but partially marking
# them would visually exaggerate coverage, so the copy below names the
# 10-26 deg highlighted range explicitly.
LA_SWEET_SPOT_BINS = (3, 4)


def _weighted_hist(values: pd.Series, weights: pd.Series,
                   bins: list[float]) -> list[float]:
    """Normalized weighted histogram of `values` over `bins`. Skips NaNs.
    Returns a list of bin probabilities summing to 1 (or all-zeros if empty).
    """
    if values is None or len(values) == 0:
        return [0.0] * (len(bins) - 1)
    s = pd.to_numeric(values, errors="coerce")
    mask = s.notna()
    if not mask.any():
        return [0.0] * (len(bins) - 1)
    v = s[mask].to_numpy()
    w = pd.to_numeric(weights, errors="coerce").reindex(s.index)[mask].fillna(0.0).to_numpy()
    counts, _ = np.histogram(v, bins=bins, weights=w)
    total = float(counts.sum())
    if total <= 0:
        return [0.0] * (len(bins) - 1)
    return [float(c) / total for c in counts]


def _qoc_dist(bbe: pd.DataFrame) -> dict[int, float]:
    """P(launch_speed_angle bucket) over weighted BBEs (renormalized over
    rows that have a non-null bucket value)."""
    out = {q: 0.0 for q in QOC_LABELS}
    if bbe.empty or "launch_speed_angle" not in bbe.columns:
        return out
    sub = bbe[bbe["launch_speed_angle"].notna()]
    total = float(sub["weight"].sum()) if "weight" in sub.columns else float(len(sub))
    if total <= 0:
        return out
    weights = sub["weight"] if "weight" in sub.columns else pd.Series(1.0, index=sub.index)
    for q in QOC_LABELS:
        m = (sub["launch_speed_angle"] == q).astype(float)
        out[q] = float((m * weights).sum()) / total
    return out


def contact_quality_projection(batter_blended: pd.DataFrame,
                                marginal: pd.Series) -> dict:
    """Project the batter's BBE distribution under the pitcher's pitch mix.

    Returns a dict with the projected and career baseline distributions for
    the Statcast quality-of-contact 6-bin classification, the EV histogram,
    and the LA histogram, plus weighted-mean EV/LA for the section subtitle.

    The career baseline is the batter's full platoon-filtered BBE
    distribution. The projected distribution re-weights each pitch-specific
    sub-distribution by the pitcher's marginal pitch usage (same `marginal`
    that drives the headline xwOBA projection); pitch types where the batter
    has fewer than 5 weighted BBE fall back to the batter's overall BBE
    distribution and the projection is renormalized over the used weight.
    """
    empty = {
        "available": False,
        "n_career_bbe": 0.0,
        "qoc_proj": {q: 0.0 for q in QOC_LABELS},
        "qoc_career": {q: 0.0 for q in QOC_LABELS},
        "ev_proj": [0.0] * (len(EV_BINS) - 1),
        "ev_career": [0.0] * (len(EV_BINS) - 1),
        "la_proj": [0.0] * (len(LA_BINS) - 1),
        "la_career": [0.0] * (len(LA_BINS) - 1),
        "mean_ev_career": float("nan"), "mean_ev_proj": float("nan"),
        "mean_la_career": float("nan"), "mean_la_proj": float("nan"),
    }
    if batter_blended is None or batter_blended.empty:
        return empty
    if "type" not in batter_blended.columns or "launch_speed" not in batter_blended.columns:
        return empty

    bbe = batter_blended[(batter_blended["type"] == "X")
                         & batter_blended["launch_speed"].notna()].copy()
    if bbe.empty:
        return empty
    if "weight" not in bbe.columns:
        bbe["weight"] = 1.0

    # Career baseline (across the batter's natural pitch usage).
    qoc_career = _qoc_dist(bbe)
    ev_career = _weighted_hist(bbe["launch_speed"], bbe["weight"], EV_BINS)
    la_career = _weighted_hist(bbe["launch_angle"], bbe["weight"], LA_BINS)

    n_career_bbe = float(bbe["weight"].sum())
    mean_ev_career = w_mean(bbe["launch_speed"], bbe["weight"])
    mean_la_career = w_mean(bbe["launch_angle"], bbe["weight"])

    # Projected distribution under the pitcher's marginal pitch usage.
    qoc_proj = {q: 0.0 for q in QOC_LABELS}
    ev_proj = [0.0] * (len(EV_BINS) - 1)
    la_proj = [0.0] * (len(LA_BINS) - 1)
    mean_ev_proj_acc = 0.0
    mean_la_proj_acc = 0.0
    used_weight = 0.0

    if marginal is None or len(marginal) == 0:
        # No pitcher marginal to reweight by; projected == career.
        return {
            "available": True,
            "n_career_bbe": n_career_bbe,
            "qoc_proj": dict(qoc_career), "qoc_career": qoc_career,
            "ev_proj": list(ev_career), "ev_career": ev_career,
            "la_proj": list(la_career), "la_career": la_career,
            "mean_ev_career": mean_ev_career, "mean_ev_proj": mean_ev_career,
            "mean_la_career": mean_la_career, "mean_la_proj": mean_la_career,
        }

    bbe_by_pitch = dict(tuple(bbe.groupby("pitch_name"))) if "pitch_name" in bbe.columns else {}
    for pitch_name, p in marginal.items():
        sub = bbe_by_pitch.get(pitch_name)
        if sub is None or float(sub["weight"].sum()) < 5.0:
            sub_use = bbe  # Fall back to overall BBE distribution.
        else:
            sub_use = sub

        q_p = _qoc_dist(sub_use)
        e_p = _weighted_hist(sub_use["launch_speed"], sub_use["weight"], EV_BINS)
        l_p = _weighted_hist(sub_use["launch_angle"], sub_use["weight"], LA_BINS)
        for q in QOC_LABELS:
            qoc_proj[q] += float(p) * q_p[q]
        for i in range(len(ev_proj)):
            ev_proj[i] += float(p) * e_p[i]
        for i in range(len(la_proj)):
            la_proj[i] += float(p) * l_p[i]
        mean_ev_proj_acc += float(p) * w_mean(sub_use["launch_speed"], sub_use["weight"])
        mean_la_proj_acc += float(p) * w_mean(sub_use["launch_angle"], sub_use["weight"])
        used_weight += float(p)

    if used_weight > 0:
        for q in QOC_LABELS:
            qoc_proj[q] /= used_weight
        ev_proj = [v / used_weight for v in ev_proj]
        la_proj = [v / used_weight for v in la_proj]
        mean_ev_proj = mean_ev_proj_acc / used_weight
        mean_la_proj = mean_la_proj_acc / used_weight
    else:
        mean_ev_proj = mean_ev_career
        mean_la_proj = mean_la_career

    return {
        "available": True,
        "n_career_bbe": n_career_bbe,
        "qoc_proj": qoc_proj, "qoc_career": qoc_career,
        "ev_proj": ev_proj, "ev_career": ev_career,
        "la_proj": la_proj, "la_career": la_career,
        "mean_ev_career": mean_ev_career, "mean_ev_proj": mean_ev_proj,
        "mean_la_career": mean_la_career, "mean_la_proj": mean_la_proj,
    }


def outcome_distribution(proj: dict, batter_pt: pd.DataFrame,
                          marginal: pd.Series) -> pd.DataFrame:
    """Outcome shares (K, BB, HBP, 1B, 2B, 3B, HR, BIP_out) summing to 1.

    Uses Layer 1 K/BB/HBP/xBA, plus a marginal-usage-weighted hit-type mix
    pulled from the batter's per-pitch-type distribution. Cross-checks against
    projected xwOBA and rescales hit components if they drift more than 5 pts.
    """
    K = proj["K_pct"]
    BB = proj["BB_pct"]
    HBP = proj["HBP_pct"]
    BIP = max(0.0, 1.0 - K - BB - HBP)

    AB_share = max(1e-9, 1.0 - BB - HBP)
    hits_per_PA = proj["xBA"] * AB_share
    BIP_hits = max(0.0, hits_per_PA)

    # Hit-type mix (1B/2B/3B/HR shares within a hit) weighted by marginal usage.
    if batter_pt.empty or marginal.empty:
        # Fall back to league shares.
        lg_hits = LG_OUTCOMES["1B"] + LG_OUTCOMES["2B"] + LG_OUTCOMES["3B"] + LG_OUTCOMES["HR"]
        share_1b = LG_OUTCOMES["1B"] / lg_hits
        share_2b = LG_OUTCOMES["2B"] / lg_hits
        share_3b = LG_OUTCOMES["3B"] / lg_hits
        share_hr = LG_OUTCOMES["HR"] / lg_hits
    else:
        bat_idx = batter_pt.set_index("pitch_name")
        num = {"1B": 0.0, "2B": 0.0, "3B": 0.0, "HR": 0.0}
        for pitch_name, p in marginal.items():
            if pitch_name not in bat_idx.index:
                continue
            row = bat_idx.loc[pitch_name]
            n_hits = row["1B_w"] + row["2B_w"] + row["3B_w"] + row["HR_w"]
            if n_hits <= 0:
                continue
            num["1B"] += p * (row["1B_w"] / n_hits)
            num["2B"] += p * (row["2B_w"] / n_hits)
            num["3B"] += p * (row["3B_w"] / n_hits)
            num["HR"] += p * (row["HR_w"] / n_hits)
        total = sum(num.values())
        if total <= 0:
            lg_hits = LG_OUTCOMES["1B"] + LG_OUTCOMES["2B"] + LG_OUTCOMES["3B"] + LG_OUTCOMES["HR"]
            share_1b = LG_OUTCOMES["1B"] / lg_hits
            share_2b = LG_OUTCOMES["2B"] / lg_hits
            share_3b = LG_OUTCOMES["3B"] / lg_hits
            share_hr = LG_OUTCOMES["HR"] / lg_hits
        else:
            share_1b = num["1B"] / total
            share_2b = num["2B"] / total
            share_3b = num["3B"] / total
            share_hr = num["HR"] / total

    H1 = BIP_hits * share_1b
    H2 = BIP_hits * share_2b
    H3 = BIP_hits * share_3b
    HR = BIP_hits * share_hr
    BIP_out = max(0.0, BIP - (H1 + H2 + H3 + HR))

    # Cross-check: reconstruct wOBA from outcome shares; if it disagrees with
    # projected xwOBA by >5 pts, scale 1B/2B/3B/HR proportionally to reconcile.
    def _reconstructed_woba(_h1, _h2, _h3, _hr):
        return (WOBA_BB * BB + WOBA_HBP * HBP +
                WOBA_1B * _h1 + WOBA_2B * _h2 + WOBA_3B * _h3 + WOBA_HR * _hr)

    target = proj["xwOBA"]
    woba0 = _reconstructed_woba(H1, H2, H3, HR)
    # Skip reconciliation when the headline projection itself is unusable
    # (NaN can now arrive here via w_xwoba on tiny samples).
    target_ok = target is not None and not (isinstance(target, float) and math.isnan(target))
    if target_ok and abs(woba0 - target) > 0.005 and (H1 + H2 + H3 + HR) > 0:
        contact_woba = WOBA_1B * H1 + WOBA_2B * H2 + WOBA_3B * H3 + WOBA_HR * HR
        residual = target - (WOBA_BB * BB + WOBA_HBP * HBP)
        if contact_woba > 0:
            if residual > 0:
                # Scale hits up or down toward the residual; clamp the
                # multiplier so a near-zero contact_woba can't blow up.
                scale = residual / contact_woba
                scale = float(np.clip(scale, 0.5, 2.0))
            else:
                # BB + HBP already account for (or overshoot) the entire
                # projection. Squeeze hits toward zero rather than leaving the
                # original (drifted) values in place.
                scale = 0.5
            H1 *= scale
            H2 *= scale
            H3 *= scale
            HR *= scale
            total_hits = H1 + H2 + H3 + HR
            if total_hits > BIP:
                # Don't let hits exceed BIP%.
                k = BIP / total_hits
                H1 *= k
                H2 *= k
                H3 *= k
                HR *= k
            BIP_out = max(0.0, BIP - (H1 + H2 + H3 + HR))

    rows = [
        ("Strikeout", K, LG_OUTCOMES["K"]),
        ("Walk", BB, LG_OUTCOMES["BB"]),
        ("HBP", HBP, LG_OUTCOMES["HBP"]),
        ("Single", H1, LG_OUTCOMES["1B"]),
        ("Double", H2, LG_OUTCOMES["2B"]),
        ("Triple", H3, LG_OUTCOMES["3B"]),
        ("Home Run", HR, LG_OUTCOMES["HR"]),
        ("In-play out", BIP_out, LG_OUTCOMES["BIP_out"]),
    ]
    df = pd.DataFrame(rows, columns=["Outcome", "Prob", "League"])

    # Convenience rollups (not part of the sum).
    hit_any = H1 + H2 + H3 + HR
    on_base = hit_any + BB + HBP
    df_extra = pd.DataFrame([
        ("Hit (any)", hit_any, sum(LG_OUTCOMES[k] for k in ("1B", "2B", "3B", "HR"))),
        ("On-base", on_base, sum(LG_OUTCOMES[k] for k in ("1B", "2B", "3B", "HR", "BB", "HBP"))),
    ], columns=["Outcome", "Prob", "League"])
    return pd.concat([df, df_extra], ignore_index=True)


# ---------- Multi-PA outlook ----------------------------------------------

def multi_pa_outlook(outcomes: pd.DataFrame, ns: tuple[int, ...] = (2, 3, 4)) -> pd.DataFrame:
    """For each outcome, compute P(>=1 in N PAs) and expected count over N PAs.

    Assumes per-PA outcomes are independent draws from the projected distribution
    (a simplifying assumption — real PAs share context like pitcher fatigue, but
    is the standard quick approximation for slate-style outlooks).
    """
    if outcomes.empty:
        return pd.DataFrame()

    rows = []
    for _, r in outcomes.iterrows():
        p = float(r["Prob"])
        row = {"Outcome": r["Outcome"]}
        for n in ns:
            at_least_one = 1.0 - (1.0 - p) ** n
            expected = n * p
            row[f"P(>=1) in {n} PA"] = at_least_one
            row[f"E[count] in {n} PA"] = expected
        rows.append(row)
    return pd.DataFrame(rows)


# ---------- Defensive alignment polish ------------------------------------

def alignment_split(bat_vs_pit: pd.DataFrame, pit_vs_bat: pd.DataFrame) -> dict:
    """Batter grounder BABIP by infield alignment, plus pitcher's typical alignment."""
    if "if_fielding_alignment" not in bat_vs_pit.columns or bat_vs_pit.empty:
        return {}

    bbe = bat_vs_pit[(bat_vs_pit["type"] == "X") &
                      (bat_vs_pit["bb_type"] == "ground_ball") &
                      bat_vs_pit["if_fielding_alignment"].notna()].copy()
    if bbe.empty:
        return {}

    bbe["is_hit"] = bbe["events"].isin(HIT_EVENTS)
    rows = []
    for align, group in bbe.groupby("if_fielding_alignment"):
        n_w = float(group["weight"].sum())
        if n_w == 0:
            continue
        babip = float((group["is_hit"].astype(float) * group["weight"]).sum()) / n_w
        rows.append({"Alignment": align, "GB BABIP": babip, "Sample (eff GB)": n_w})

    by_align = pd.DataFrame(rows).sort_values("GB BABIP", ascending=False).reset_index(drop=True)

    pitcher_align = (pit_vs_bat["if_fielding_alignment"].dropna().value_counts(normalize=True) * 100).round(1)

    return {"batter_gb_babip": by_align, "pitcher_alignment_mix": pitcher_align}


# ---------- verdict + narrative ------------------------------------------

def verdict(proj_xwoba: float, lg: float, pit_baseline: float,
            bat_baseline: float) -> dict:
    def _v(diff_pts):
        if diff_pts > 5:
            return "Edge: Hitter"
        if diff_pts < -5:
            return "Edge: Pitcher"
        return "Even"

    lg_diff = (proj_xwoba - lg) * 1000
    pit_diff = (proj_xwoba - pit_baseline) * 1000
    bat_diff = (proj_xwoba - bat_baseline) * 1000
    return {
        "vs_league": (proj_xwoba, lg, lg_diff, _v(lg_diff)),
        "vs_pitcher": (proj_xwoba, pit_baseline, pit_diff,
                        "Edge: Hitter (vs Pitcher norm)" if pit_diff > 5 else
                        "Edge: Pitcher (vs Pitcher norm)" if pit_diff < -5 else
                        "Pitcher norm"),
        "vs_batter": (proj_xwoba, bat_baseline, bat_diff,
                       "Edge: Hitter (vs Batter norm)" if bat_diff > 5 else
                       "Edge: Pitcher (vs Batter norm)" if bat_diff < -5 else
                       "Batter norm"),
    }


def narrative(batter_meta: dict, pitcher_meta: dict, proj: dict,
              v: dict, bat_fav: pd.DataFrame, pit_fav: pd.DataFrame,
              handed: str,
              recent_summary: dict | None = None) -> str:
    """Generate a 1-2 sentence summary."""
    proj_x = proj["xwOBA"]

    top_pitcher_pitch = pit_fav.iloc[0]["Pitch"] if not pit_fav.empty else None
    top_batter_pitch = bat_fav.iloc[0]["Pitch"] if not bat_fav.empty else None

    lg_diff = v["vs_league"][2]
    edge = v["vs_league"][3]

    parts = []
    if top_pitcher_pitch:
        parts.append(f"{pitcher_meta['possessive']} {top_pitcher_pitch.lower()} is the projected matchup advantage")
        if top_batter_pitch and top_batter_pitch != top_pitcher_pitch:
            parts.append(f"while {batter_meta['name']} should look to drive the {top_batter_pitch.lower()}")
    summary = ", ".join(parts) + (". " if parts else "")
    if not summary:
        summary = ""

    summary += (f"Projection: {proj_x:.3f} xwOBA "
                f"({lg_diff:+.0f} pts vs league, {edge.split(': ')[-1] if ':' in edge else edge}).")

    adj_pts = float(proj.get("xwOBA_adj_pts", 0.0) or 0.0)
    raw_x = float(proj.get("xwOBA_raw", proj_x) or proj_x)
    bbtype_x = float(proj.get("xwOBA_bbtype", proj_x) or proj_x)
    if abs(adj_pts) >= 5:
        direction = (
            "skews toward weaker contact" if adj_pts < 0
            else "skews toward stronger contact"
        )
        summary += (
            f" GB-mix adjustment: {adj_pts:+.0f} pts "
            f"(raw {raw_x:.3f} -> adj {bbtype_x:.3f}); pitcher's induced bb_type mix "
            f"{direction} than the batter's career mix would suggest."
        )

    count_pts = float(proj.get("xwOBA_count_pts", 0.0) or 0.0)
    if abs(count_pts) >= 5:
        cdir = (
            "pushes the batter into worse counts" if count_pts < 0
            else "pushes the batter into more favorable counts"
        )
        summary += (
            f" Count-state blend: {count_pts:+.0f} pts (bb_type-adj {bbtype_x:.3f} -> "
            f"final {proj_x:.3f}); pitcher's count distribution {cdir} relative to "
            f"how the batter performs across counts."
        )

    fb = proj.get("marginal_fallback")
    if fb == "flat_pitcher_pt":
        summary += (" NOTE: count-conditional pitch mix unavailable, used pitcher's "
                    "flat per-pitch usage instead — treat headline as low confidence.")
    elif fb == "overall_only":
        summary += (" NOTE: no per-pitch arsenal data, projection reduced to a single "
                    "overall additive — treat as low confidence.")

    if recent_summary:
        bat_d = recent_summary.get("batter_d14_xwoba_delta")
        if bat_d is not None and abs(bat_d) >= RECENT_FORM_STREAK_THRESHOLD:
            tag = "heater" if bat_d > 0 else "slump"
            summary += (
                f" {batter_meta['name']} is on a 14-day {tag} "
                f"({bat_d*1000:+.0f} pts vs season)."
            )
        pit_d = recent_summary.get("pitcher_d14_xwoba_delta")
        if pit_d is not None and abs(pit_d) >= RECENT_FORM_STREAK_THRESHOLD:
            # For pitchers, lower xwOBA-allowed = better; flip the wording.
            tag = "rough patch" if pit_d > 0 else "hot stretch"
            summary += (
                f" {pitcher_meta['name']} is in a 14-day {tag} "
                f"({pit_d*1000:+.0f} pts xwOBA-allowed vs season)."
            )

    if handed:
        summary += f" {handed}"
    return summary


# ---------- markdown rendering -------------------------------------------

def fmt_pct(x: float, dp: int = 1) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{dp}f}%"


def fmt3(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.3f}"


def fmt1(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.1f}"


def to_markdown(
    batter_meta: dict, pitcher_meta: dict,
    season: int,
    bat_blended: pd.DataFrame, pit_blended: pd.DataFrame,
    proj: dict,
    v: dict, narrative_text: str,
    outcomes: pd.DataFrame,
    multi_pa: pd.DataFrame,
    tto_proj: pd.DataFrame,
    panel: pd.DataFrame, panel_notes: list[str],
    pitch_table: pd.DataFrame,
    count_mix_summary: pd.DataFrame,
    comps: pd.DataFrame,
    zone_overlay_df: pd.DataFrame,
    bat_track_overall: dict, bat_track_pitch: pd.DataFrame,
    sub: dict,
    bat_fav: pd.DataFrame, pit_fav: pd.DataFrame,
    decep: dict, handed_note: str,
    align: dict,
    contact_quality: dict | None = None,
    recent_form: dict | None = None,
    body_only: bool = False,
) -> str:
    lines: list[str] = []

    # ----- Header -----
    lines.append(f"# {batter_meta['name']} vs {pitcher_meta['name']} — matchup")
    lines.append("")
    bat_counts = _per_season_counts(bat_blended)
    pit_counts = _per_season_counts(pit_blended)
    weight_str = ", ".join(
        f"{season - off} \u00d7{w:g}" for off, w in enumerate(SEASON_WEIGHTS)
    )
    bat_rows_str = "+".join(str(bat_counts.get(w, 0)) for w in SEASON_WEIGHTS)
    pit_rows_str = "+".join(str(pit_counts.get(w, 0)) for w in SEASON_WEIGHTS)
    lines.append(
        f"_window: {weight_str}; batter rows: {bat_rows_str}, "
        f"pitcher rows: {pit_rows_str}_"
    )
    lines.append(
        f"_handedness: {batter_meta['stand']}HB vs {pitcher_meta['p_throws']}HP_"
    )
    lines.append("")

    # ----- Narrative + verdict -----
    lines.append("## Verdict")
    lines.append("")
    lines.append(f"**{narrative_text}**")
    lines.append("")
    proj_a = proj.get("xBA", float("nan"))
    proj_s = proj.get("xSLG", float("nan"))
    lines.append(
        f"_Projected slash: **xwOBA {fmt3(proj['xwOBA'])}** "
        f"(lg {fmt3(LG_XWOBA)}) · **xBA {fmt3(proj_a)}** (lg {fmt3(LG_XBA)}) · "
        f"**xSLG {fmt3(proj_s)}** (lg {fmt3(LG_XSLG)})._"
    )
    lines.append("")
    lines.append("| Frame | Projected xwOBA | Baseline | Δ (wOBA pts) | Read |")
    lines.append("|---|---:|---:|---:|---|")
    for label, key in (("vs league avg", "vs_league"),
                       (f"vs {pitcher_meta['possessive']} baseline", "vs_pitcher"),
                       (f"vs {batter_meta['possessive']} baseline", "vs_batter")):
        proj_x, base, diff, read = v[key]
        lines.append(f"| {label} | {fmt3(proj_x)} | {fmt3(base)} | {diff:+.0f} | {read} |")
    lines.append("")

    # ----- PA outcome table -----
    lines.append("## Per-PA outcome distribution")
    lines.append("")
    lines.append("| Outcome | Prob | American | League |")
    lines.append("|---|---:|---:|---:|")
    for _, r in outcomes.iterrows():
        prob = r["Prob"]
        lg = r["League"]
        am = american_odds(prob)
        lines.append(f"| {r['Outcome']} | {prob*100:.1f}% | {am} | {lg*100:.1f}% |")
        if r["Outcome"] == "In-play out":
            lines.append("| --- | --- | --- | --- |")
    lines.append("")

    # ----- Multi-PA outlook -----
    lines.append("## Multi-PA outlook (2 / 3 / 4 PAs)")
    lines.append("")
    if multi_pa.empty:
        lines.append("_no outcome distribution available_")
    else:
        lines.append(
            "_For each outcome: chance it happens **at least once** across N PAs "
            "(with American odds), and **expected count** across N PAs "
            "(assumes independent PAs)._"
        )
        lines.append("")
        lines.append(
            "| Outcome "
            "| Over 2 PA: chance ≥1 | Odds | Avg # "
            "| Over 3 PA: chance ≥1 | Odds | Avg # "
            "| Over 4 PA: chance ≥1 | Odds | Avg # |"
        )
        lines.append(
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        )
        for _, r in multi_pa.iterrows():
            cells = [r["Outcome"]]
            for n in (2, 3, 4):
                p = float(r[f"P(>=1) in {n} PA"])
                ec = float(r[f"E[count] in {n} PA"])
                cells += [f"{p*100:.1f}%", american_odds(p), f"{ec:.2f}"]
            lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # ----- Headline by TTO -----
    lines.append("## Projection by times through the order")
    lines.append("")
    if tto_proj.empty:
        lines.append("_no TTO data available_")
    else:
        lines.append("| TTO | Proj xwOBA | Proj xBA | Proj xSLG | Proj K % | Proj Whiff % | Proj Hard Hit % | Sample (eff PA) |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
        for _, r in tto_proj.iterrows():
            lines.append(
                f"| {int(r['TTO'])} | {fmt3(r['Projected xwOBA'])} | "
                f"{fmt3(r.get('Projected xBA'))} | {fmt3(r.get('Projected xSLG'))} | "
                f"{r['Projected K %']:.1f}% | {r['Projected Whiff %']:.1f}% | "
                f"{r['Projected Hard Hit %']:.1f}% | {r['Sample (eff PA)']:.1f} |"
            )
    lines.append("")

    # ----- Discipline panel -----
    lines.append("## Side-by-side profile (platoon-filtered)")
    lines.append("")
    lines.append("| Metric | Batter | Pitcher (allowed) |")
    lines.append("|---|---:|---:|")
    for _, r in panel.iterrows():
        b_val = r["Batter (vs same hand)"]
        p_val = r["Pitcher (vs same hand)"]
        if r["Metric"] in ("xwOBA", "xBA", "xSLG"):
            lines.append(f"| {r['Metric']} | {fmt3(b_val)} | {fmt3(p_val)} |")
        else:
            lines.append(f"| {r['Metric']} | {b_val:.1f}% | {p_val:.1f}% |")
    if panel_notes:
        lines.append("")
        for n in panel_notes:
            lines.append(f"- {n}")
    lines.append("")

    # ----- Recent form -----
    if recent_form:
        windows = recent_form.get("windows", RECENT_FORM_WINDOWS)
        platoon_labels = recent_form.get("platoon_labels", {})
        lines.append("## Recent form (rolling)")
        lines.append("")

        def _row_md(side_label: str, snap: dict | None, is_season: bool = False) -> str:
            window_label = "season" if is_season else (
                f"last {snap['window_days']}d" if snap else f"last \u2014d"
            )
            if snap is None:
                return (
                    f"| {side_label} | {window_label} | \u2014 | \u2014 | \u2014 | \u2014 "
                    "| \u2014 | \u2014 | \u2014 | \u2014 |"
                )
            n_pa = snap.get("n_pa", 0)
            return (
                f"| {side_label} | {window_label} | {n_pa:.0f} | "
                f"{fmt3(snap['xwOBA'])} | {fmt3(snap.get('xBA'))} | "
                f"{fmt3(snap.get('xSLG'))} | "
                f"{snap['K_pct']*100:.1f} | {snap['BB_pct']*100:.1f} | "
                f"{snap['Whiff_pct']*100:.1f} | {snap['HardHit_pct']*100:.1f} |"
            )

        def _emit_table(view: dict, sub_heading: str) -> None:
            lines.append(f"#### {sub_heading}")
            lines.append("")
            lines.append("| Side | Window | n PA | xwOBA | xBA | xSLG | K% | BB% | Whiff% | Hard Hit% |")
            lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
            for side_key, side_label in (("batter", "Batter"), ("pitcher", "Pitcher")):
                side = view.get(side_key, {}) or {}
                lines.append(_row_md(side_label, side.get("season"), is_season=True))
                for w in windows:
                    lines.append(_row_md(side_label, side.get(f"d{int(w)}")))
            lines.append("")

        overall_view = recent_form.get("overall") or {
            "batter": {k: recent_form.get("batter", {}).get(k) for k in ("season",) + tuple(f"d{int(w)}" for w in windows)},
            "pitcher": {k: recent_form.get("pitcher", {}).get(k) for k in ("season",) + tuple(f"d{int(w)}" for w in windows)},
        }
        _emit_table(overall_view, "Overall (no platoon filter)")

        platoon_view = recent_form.get("platoon")
        if platoon_view:
            bat_pl = platoon_labels.get("batter") or "vs same hand"
            pit_pl = platoon_labels.get("pitcher") or "vs same hand"
            _emit_table(platoon_view, f"Platoon — Batter {bat_pl}, Pitcher {pit_pl}")

        lines.append(
            "_Both tables use raw current-season PA (no season-blend, no recency "
            "decay). **Overall** is un-platoon-filtered and matches the player's "
            "year-to-date totals on BR / Savant — the natural \"how is this player "
            "going right now\" read. **Platoon** restricts to tonight's matchup hand, "
            "so it's apples-to-apples with the rest of the report (which is also "
            f"platoon-filtered). Rolling rows are shown only when n PA >= {RECENT_FORM_MIN_PA:.0f}; "
            "the season row is suppressed if there's no qualifying data._"
        )
        lines.append("")

    # ----- Pitch-mix projection -----
    lines.append("## Pitch-mix projection")
    lines.append("")
    if pitch_table.empty:
        lines.append("_no arsenal data_")
    else:
        lines.append("| Pitch | Usage % | Batter xwOBA | Pitcher xwOBA allowed | Pitcher BB-mix | Adj batter xwOBA | Adj Δ (pts) | Projected xwOBA | Projected xBA | Projected xSLG | Projected Whiff % |")
        lines.append("|---|---:|---:|---:|:---|---:|---:|---:|---:|---:|---:|")
        for _, r in pitch_table.iterrows():
            adj_pts = float(r.get("Adj delta (pts)", 0.0))
            lines.append(
                f"| {r['Pitch']} | {r['Marginal Usage %']:.1f} | "
                f"{fmt3(r['Batter xwOBA'])} | {fmt3(r['Pitcher xwOBA allowed'])} | "
                f"{r.get('Pitcher BB-mix', '')} | "
                f"{fmt3(r.get('Adj batter xwOBA', r['Batter xwOBA']))} | "
                f"{adj_pts:+.0f} | "
                f"{fmt3(r['Projected xwOBA'])} | "
                f"{fmt3(r.get('Projected xBA'))} | {fmt3(r.get('Projected xSLG'))} | "
                f"{r['Projected Whiff %']:.1f} |"
            )
        lines.append("")
        lines.append(
            "_Adj Δ (pts) is the bb_type mix discount applied to the batter's per-pitch "
            "xwOBA: when the pitcher's induced bb_type mix on a pitch is more grounder-heavy "
            "than the batter's career mix on that pitch, the per-PA xwOBA contribution from "
            "that pitch is shifted down by the equivalent amount. Negative values favor the "
            "pitcher; positive values mean the pitcher's air-ball-heavy mix actually plays into "
            "the hitter's strengths._"
        )
    lines.append("")

    # ----- Contact-quality projection -----
    if contact_quality and contact_quality.get("available"):
        cq = contact_quality
        lines.append("## Contact-quality projection")
        lines.append("")
        n_bbe = cq.get("n_career_bbe", 0.0)
        mev_c = cq.get("mean_ev_career", float("nan"))
        mev_p = cq.get("mean_ev_proj", float("nan"))
        mla_c = cq.get("mean_la_career", float("nan"))
        mla_p = cq.get("mean_la_proj", float("nan"))
        sub_bits = [f"weighted from **{n_bbe:.0f}** career BBE"]
        if not math.isnan(mev_c):
            d_ev = mev_p - mev_c
            sub_bits.append(
                f"mean EV career **{mev_c:.1f}** \u2192 proj **{mev_p:.1f}** "
                f"({d_ev:+.1f} mph)"
            )
        if not math.isnan(mla_c):
            d_la = mla_p - mla_c
            sub_bits.append(
                f"mean LA career **{mla_c:.1f}\u00b0** \u2192 proj **{mla_p:.1f}\u00b0** "
                f"({d_la:+.1f}\u00b0)"
            )
        lines.append("_" + " · ".join(sub_bits) + "_")
        lines.append("")

        lines.append("**Quality of contact (Statcast 6-bin)**")
        lines.append("")
        lines.append("| Bucket | Career % | Projected % | \u0394 (pp) |")
        lines.append("|---|---:|---:|---:|")
        for q in sorted(QOC_LABELS):
            c = cq["qoc_career"].get(q, 0.0) * 100
            p = cq["qoc_proj"].get(q, 0.0) * 100
            lines.append(
                f"| {QOC_LABELS[q]} | {c:.1f} | {p:.1f} | {p-c:+.1f} |"
            )
        lines.append("")

        lines.append("**Exit velocity (mph)**")
        lines.append("")
        lines.append("| Bin | Career % | Projected % | \u0394 (pp) |")
        lines.append("|---|---:|---:|---:|")
        for lab, c, p in zip(EV_LABELS, cq["ev_career"], cq["ev_proj"]):
            c100, p100 = c * 100, p * 100
            lines.append(f"| {lab} | {c100:.1f} | {p100:.1f} | {p100-c100:+.1f} |")
        lines.append("")

        lines.append("**Launch angle (deg)**")
        lines.append("")
        lines.append("| Bin | Career % | Projected % | \u0394 (pp) |")
        lines.append("|---|---:|---:|---:|")
        for i, (lab, c, p) in enumerate(zip(LA_LABELS, cq["la_career"], cq["la_proj"])):
            c100, p100 = c * 100, p * 100
            star = " \u2605" if i in LA_SWEET_SPOT_BINS else ""
            lines.append(f"| {lab}{star} | {c100:.1f} | {p100:.1f} | {p100-c100:+.1f} |")
        lines.append("")
        lines.append(
            "_Career = batter's full platoon-filtered BBE distribution. Projected re-weights "
            "each pitch-specific contact distribution by the pitcher's marginal pitch usage; "
            "pitches with <5 weighted BBE fall back to the batter's overall mix. \u0394 (pp) "
            "is the projected share minus career share, in percentage points. \u2605 rows "
            "mark the highlighted 10\u201326\u00b0 barrel zone (overlaps Statcast's "
            "broader 8\u201332\u00b0 sweet-spot range)._"
        )
        lines.append("")

    # ----- Count-state mix summary -----
    lines.append("## Count-state pitch mix (pitcher, vs same hand)")
    lines.append("")
    if count_mix_summary.empty:
        lines.append("_no count-state data_")
    else:
        lines.append(count_mix_summary.to_markdown(index=False))
    lines.append("")

    # ----- Shape comps -----
    lines.append("## Shape-aware comps from batter history")
    lines.append("")
    if comps.empty:
        lines.append("_no comps available_")
    else:
        lines.append("| Pitch | Shape (eff velo / IVB / HB-in) | n comps (eff) | Whiff % | xwOBA | xBA | xSLG | Hard Hit % | Confidence |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|")
        for _, r in comps.iterrows():
            lines.append(
                f"| {r['Pitch']} | {r['Shape (eff velo / IVB / HB-in)']} | "
                f"{r['n comps (eff)']:.1f} | {fmt_pct(r['Whiff %'])} | "
                f"{fmt3(r['xwOBA'])} | {fmt3(r.get('xBA'))} | "
                f"{fmt3(r.get('xSLG'))} | {fmt_pct(r['Hard Hit %'])} | {r['Confidence']} |"
            )
    lines.append("")

    # ----- Zone overlay -----
    lines.append("## Zone overlay")
    lines.append("")
    if zone_overlay_df.empty:
        lines.append("_no zone data_")
    else:
        lines.append("| Pitch | In-zone % | Top zones (attack share) | Intersection xwOBA | Intersection xBA | Intersection xSLG | Coverage % |")
        lines.append("|---|---:|---|---:|---:|---:|---:|")
        for _, r in zone_overlay_df.iterrows():
            cov = r.get("Coverage %")
            cov_str = f"{cov:.0f}" if cov is not None and not (isinstance(cov, float) and math.isnan(cov)) else "—"
            lines.append(
                f"| {r['Pitch']} | {r['In-zone %']:.1f} | "
                f"{r['Top zones']} | {fmt3(r['Intersection xwOBA'])} | "
                f"{fmt3(r.get('Intersection xBA'))} | {fmt3(r.get('Intersection xSLG'))} | "
                f"{cov_str} |"
            )
        lines.append("")
        lines.append("_Coverage % = share of the pitcher's attack on this pitch landing in zones with batter data; remainder imputed at league baselines (xwOBA 0.315, xBA 0.245, xSLG 0.405)._")
    lines.append("")

    # ----- Bat tracking -----
    lines.append("## Bat-tracking interaction")
    lines.append("")
    if not bat_track_overall:
        lines.append("_no bat-tracking data in this window_")
    else:
        if bat_track_overall.get("attack_angle") is not None:
            lines.append(
                f"Batter avg: bat speed **{bat_track_overall['bat_speed']:.1f} mph**, "
                f"swing length **{bat_track_overall['swing_length']:.1f} ft**, "
                f"attack angle **{bat_track_overall['attack_angle']:.1f}°**"
            )
            lines.append("")
        if not bat_track_pitch.empty:
            lines.append(
                "| Pitch | VAA (deg) | Bat attack (deg) | Swings (n) | Match gap (deg) | Note |"
            )
            lines.append("|---|---:|---:|---:|---:|---|")
            for _, r in bat_track_pitch.iterrows():
                attack_str = f"{r['Bat attack (deg)']:+.1f}" if not math.isnan(r['Bat attack (deg)']) else "—"
                if r["Fallback"]:
                    attack_str += "*"
                gap_str = f"{r['Match gap (deg)']:+.1f}" if not math.isnan(r['Match gap (deg)']) else "—"
                lines.append(
                    f"| {r['Pitch']} | {r['VAA (deg)']:+.1f} | "
                    f"{attack_str} | {r['Swings (n)']:.0f} | "
                    f"{gap_str} | {r['Note']} |"
                )
            if bat_track_pitch["Fallback"].any():
                lines.append("")
                lines.append(
                    f"_\\* fewer than {MIN_SWINGS_PER_PITCH:.0f} swings on this pitch type; "
                    "showing batter's overall attack angle as a fallback._"
                )
            lines.append("")
            lines.append(
                "_Match gap = bat attack angle + pitch VAA. "
                f"|gap| ≤ {MATCH_THRESHOLD_DEG:.0f}° = on plane; "
                "positive = swing steeper than pitch (under risk); "
                "negative = swing flatter than pitch (top risk)._"
            )
    lines.append("")

    # ----- Sub-profiles -----
    lines.append("## First-pitch & two-strike sub-profiles")
    lines.append("")
    fp = sub.get("first_pitch")
    ts = sub.get("two_strike")
    if fp:
        lines.append(
            f"**First pitch**: pitcher first-pitch strike% {fp['pitcher_strike_pct']*100:.1f}%, "
            f"batter first-pitch swing% {fp['batter_swing_pct']*100:.1f}%, "
            f"batter xwOBA on first-pitch swings {fmt3(fp['batter_xwoba_on_swing'])}"
        )
    if ts:
        lines.append("")
        lines.append(
            f"**Two-strike**: pitcher putaway% {ts['pitcher_putaway_pct']*100:.1f}%, "
            f"batter K% in 2-strike counts {ts['batter_K_pct_2s']*100:.1f}%, "
            f"batter xwOBA in 2-strike counts {fmt3(ts['batter_xwoba_2s'])}"
        )
        if ts.get("two_strike_mix"):
            mix_str = ", ".join(f"{k} {v:.1f}%" for k, v in ts["two_strike_mix"].items())
            lines.append("")
            lines.append(f"Pitcher's two-strike mix: {mix_str}")
    lines.append("")

    # ----- Edge analysis -----
    lines.append("## Edge analysis")
    lines.append("")
    lines.append(
        "_Per-pitch matchup result vs league (additive batter + pitcher projection "
        "minus 0.315). **Edge (pts)** is the projection delta in wOBA points; "
        "positive favors the hitter, negative favors the pitcher. Tables only list "
        "pitches whose net projection actually leans in that direction._"
    )
    lines.append("")
    lines.append("**Pitches favoring the hitter**")
    lines.append("")
    edge_header = (
        "| Pitch | Usage % | Batter xwOBA | Pitcher xwOBA allowed | "
        "Projected xwOBA | Projected xBA | Projected xSLG | Edge (pts) |"
    )
    edge_sep = "|---|---:|---:|---:|---:|---:|---:|---:|"

    def _edge_md(r) -> str:
        return (
            f"| {r['Pitch']} | {r['Usage %']:.1f} | "
            f"{fmt3(r['Batter xwOBA'])} | {fmt3(r['Pitcher xwOBA allowed'])} | "
            f"{fmt3(r['Projected xwOBA'])} | {fmt3(r.get('Projected xBA'))} | "
            f"{fmt3(r.get('Projected xSLG'))} | {r['Edge (pts)']:+.0f} |"
        )

    if bat_fav.empty:
        lines.append("_no pitch in the arsenal projects above league for this hitter_")
    else:
        lines.append(edge_header)
        lines.append(edge_sep)
        for _, r in bat_fav.iterrows():
            lines.append(_edge_md(r))
    lines.append("")
    lines.append("**Pitches favoring the pitcher**")
    lines.append("")
    if pit_fav.empty:
        lines.append("_no pitch in the arsenal projects below league for this hitter_")
    else:
        lines.append(edge_header)
        lines.append(edge_sep)
        for _, r in pit_fav.iterrows():
            lines.append(_edge_md(r))
    lines.append("")

    # ----- Deception / spin axis / handedness -----
    lines.append("## Deception & shape signature")
    lines.append("")
    if decep:
        lines.append(f"Release-point cluster: **{decep['cluster_stdev_in']:.2f} in** "
                     f"({decep['deception_label']})")
        lines.append("")
        if not decep["per_pitch"].empty:
            lines.append("| Pitch | Δ from release centroid (in) | Spin axis (deg) |")
            lines.append("|---|---:|---:|")
            for _, r in decep["per_pitch"].iterrows():
                axis_v = "—" if pd.isna(r["Spin axis (deg)"]) else f"{r['Spin axis (deg)']:.0f}"
                lines.append(f"| {r['Pitch']} | {r['Δ from centroid (in)']:.2f} | {axis_v} |")
    if handed_note:
        lines.append("")
        lines.append(f"_{handed_note}_")
    lines.append("")

    # ----- Defensive alignment -----
    lines.append("## Defensive alignment")
    lines.append("")
    if not align:
        lines.append("_no alignment data_")
    else:
        ba = align.get("batter_gb_babip")
        if ba is not None and not ba.empty:
            lines.append("Batter ground-ball BABIP by infield alignment:")
            lines.append("")
            lines.append("| Alignment | GB BABIP | Sample (eff GB) |")
            lines.append("|---|---:|---:|")
            for _, r in ba.iterrows():
                lines.append(f"| {r['Alignment']} | {fmt3(r['GB BABIP'])} | {r['Sample (eff GB)']:.1f} |")
        pa_mix = align.get("pitcher_alignment_mix")
        if pa_mix is not None and len(pa_mix):
            lines.append("")
            lines.append("Pitcher's typical infield alignment usage:")
            lines.append("")
            mix_str = ", ".join(f"{idx} {val:.1f}%" for idx, val in pa_mix.items())
            lines.append(mix_str)
    lines.append("")

    # ----- Notes -----
    if not body_only:
        lines.append("## Notes & caveats")
        lines.append("")
        lines.append(
            "- Headline projection combines a count-conditional pitch mix from the "
            "pitcher with the batter's per-pitch-type xwOBA / xBA / xSLG (additive vs "
            "league) and Whiff% / Hard Hit% (log5)."
        )
        lines.append(
            "- All inputs are platoon-filtered: batter rows are restricted to pitches "
            f"from {pitcher_meta['p_throws']}HP, pitcher rows to pitches against {batter_meta['stand']}HB."
        )
        lines.append(
            "- Window blends "
            + ", ".join(
                f"{season - off} (weight {w:g})" for off, w in enumerate(SEASON_WEIGHTS)
            )
            + ". Adjust `SEASON_WEIGHTS` at the top of `matchup.py` to taste."
        )
        lines.append(
            "- Per-PA outcome shares are derived from the projection plus the batter's hit-type mix; "
            "they're cross-checked against projected xwOBA and reconciled if drift exceeds 5 pts."
        )
        lines.append(
            "- Not modeled: park / weather, catcher framing, umpire zone, fatigue beyond TTO. "
            "Recent form is modeled (current-season row weights decay with a 30-day half-life, and the "
            "Recent form panel above shows 14-day / 30-day rolling rates). Treat single-season cells as noisy."
        )
    return "\n".join(lines)


# ---------- HTML rendering -----------------------------------------------

_HTML_CSS = """
:root {
  --bg: #f7f8fa;
  --card: #ffffff;
  --ink: #1f2937;
  --muted: #64748b;
  --line: #e5e7eb;
  --accent: #2563eb;
  --bat-strong-bg: #d1fae5;
  --bat-strong-fg: #065f46;
  --bat-mild-bg: #ecfdf5;
  --bat-mild-fg: #047857;
  --pit-strong-bg: #fee2e2;
  --pit-strong-fg: #991b1b;
  --pit-mild-bg: #fef2f2;
  --pit-mild-fg: #b91c1c;
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; background: var(--bg); color: var(--ink);
             font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI',
                          Helvetica, Arial, sans-serif;
             font-size: 14px; line-height: 1.45; }
.container { max-width: 1180px; margin: 0 auto; padding: 24px 20px 80px; }
.page-head { padding-bottom: 16px; margin-bottom: 16px; border-bottom: 1px solid var(--line); }
.page-head h1 { margin: 0 0 6px 0; font-size: 26px; font-weight: 700; }
.page-head .meta { color: var(--muted); font-size: 13px; }
.badge { display: inline-block; padding: 2px 8px; margin-left: 8px;
         font-size: 11px; font-weight: 600; border-radius: 999px;
         background: #e0e7ff; color: #1e3a8a; vertical-align: middle; letter-spacing: 0.5px; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 10px;
        padding: 18px 22px; margin-bottom: 16px; box-shadow: 0 1px 2px rgba(0,0,0,0.03); }
.card h2 { margin: 0 0 12px 0; font-size: 16px; font-weight: 700; color: var(--ink);
           letter-spacing: 0.2px; text-transform: uppercase; }
.narrative { font-size: 15px; line-height: 1.55; padding: 12px 14px;
             background: #f1f5f9; border-left: 4px solid var(--accent);
             border-radius: 4px; margin: 0 0 14px 0; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
table thead th { text-align: right; font-weight: 600; padding: 8px 10px;
                 color: var(--muted); border-bottom: 1px solid var(--line);
                 background: #fafbfc; }
table thead th:first-child, table tbody td:first-child { text-align: left; }
table tbody td { padding: 7px 10px; text-align: right; border-bottom: 1px solid #f1f5f9;
                 font-variant-numeric: tabular-nums; }
table tbody tr:last-child td { border-bottom: none; }
.bat-edge-strong { background: var(--bat-strong-bg); color: var(--bat-strong-fg); font-weight: 700; }
.bat-edge-mild { background: var(--bat-mild-bg); color: var(--bat-mild-fg); }
.pit-edge-strong { background: var(--pit-strong-bg); color: var(--pit-strong-fg); font-weight: 700; }
.pit-edge-mild { background: var(--pit-mild-bg); color: var(--pit-mild-fg); }
.note { color: var(--muted); font-size: 12px; margin: 6px 0 0 0; }
.note-list { margin: 6px 0 0 0; padding-left: 20px; color: #334155; }
.note-list li { margin-bottom: 4px; font-size: 13px; }
.subtitle { font-size: 13px; color: var(--muted); margin: 0 0 10px 0; }
.divider-row td { background: #f8fafc; color: var(--muted); font-style: italic; }

/* ----- Multi-PA outlook ----- */
table.multi-pa thead tr:first-child th.pa-group {
  text-align: center; text-transform: uppercase; font-size: 11px;
  letter-spacing: 0.4px; color: var(--ink); background: #eef2f7;
  border-bottom: 1px solid var(--line);
}
table.multi-pa thead tr:first-child th.pa-group + th.pa-group {
  border-left: 1px solid var(--line);
}
table.multi-pa thead tr:nth-child(2) th { font-size: 11px; }
table.multi-pa th.outcome-col { text-align: left; vertical-align: bottom; }
table.multi-pa tbody td { white-space: nowrap; }
table.multi-pa tbody td.odds { color: var(--muted); font-size: 12px; }
table.multi-pa tbody td.avg { color: var(--muted); font-size: 12px; }
table.multi-pa tbody tr td:nth-child(5),
table.multi-pa tbody tr td:nth-child(8) { border-left: 1px solid #f1f5f9; }
table.multi-pa thead tr:nth-child(2) th:nth-child(4),
table.multi-pa thead tr:nth-child(2) th:nth-child(7) { border-left: 1px solid var(--line); }
.kv { display: flex; flex-wrap: wrap; gap: 14px 24px; padding: 4px 0 8px 0; font-size: 13px; }
.kv > div { color: var(--muted); }
.kv > div b { color: var(--ink); font-weight: 600; }
.pill { display: inline-block; padding: 2px 8px; border-radius: 999px;
        background: #f1f5f9; color: #334155; font-size: 11px; }
footer { margin-top: 24px; color: var(--muted); font-size: 12px; }

/* ----- Lineup view ----- */
.lineup-hero { display: flex; flex-wrap: wrap; gap: 16px 32px; align-items: baseline;
               padding: 14px 18px; background: #eff6ff; border-left: 4px solid var(--accent);
               border-radius: 6px; margin-bottom: 16px; }
.lineup-hero .stat { font-size: 13px; color: var(--muted); }
.lineup-hero .stat b { display: block; font-size: 18px; color: var(--ink);
                       font-variant-numeric: tabular-nums; font-weight: 700; margin-top: 2px; }
table.pitcher-bf tbody td { font-variant-numeric: tabular-nums; }
table.pitcher-bf tbody td:first-child { text-align: center; color: var(--muted);
                                        background: #fff7ed; }
.lineup-grid td.spot { text-align: center; font-weight: 700; color: var(--muted); width: 32px; }
.lineup-grid td.name { font-weight: 600; text-align: left; }
.lineup-grid td.handpill { text-align: center; }
.lineup-grid td.handpill .pill { font-weight: 600; }
.lineup-grid td.pitch-cell { text-align: left; font-size: 12px; color: var(--muted); }
.lineup-grid td.verdict { text-align: center; font-weight: 600; border-radius: 4px; }
.lineup-grid .neutral { background: #f1f5f9; color: #334155; }
.batter-block { margin-top: 14px; border: 1px solid var(--line); border-radius: 10px;
                background: var(--card); box-shadow: 0 1px 2px rgba(0,0,0,0.03); }
.batter-block > summary { list-style: none; cursor: pointer; padding: 14px 20px;
                          font-size: 15px; display: flex; align-items: center;
                          gap: 10px 16px; flex-wrap: wrap; }
.batter-block > summary::-webkit-details-marker { display: none; }
.batter-block > summary::before { content: '\\25B8'; font-size: 11px; color: var(--muted);
                                  transition: transform 0.15s ease; display: inline-block; }
.batter-block[open] > summary::before { transform: rotate(90deg); }
.batter-block > summary .spot { font-weight: 700; color: var(--muted); width: 22px; text-align: right; }
.batter-block > summary .name { font-weight: 700; color: var(--ink); }
.batter-block > summary .summary-stat { color: var(--muted); font-size: 12px;
                                        font-variant-numeric: tabular-nums; }
.batter-block > summary .summary-stat b { color: var(--ink); font-weight: 700; }
.batter-block > summary .verdict-pill { margin-left: auto; padding: 3px 10px; border-radius: 999px;
                                        font-size: 11px; font-weight: 700; letter-spacing: 0.4px; }
.batter-block .batter-body { padding: 0 20px 18px 20px; border-top: 1px solid var(--line); }
.batter-block .batter-body section.card { border: none; padding: 14px 0; margin-bottom: 0;
                                          box-shadow: none; border-bottom: 1px dashed var(--line); }
.batter-block .batter-body section.card:last-child { border-bottom: none; }
.batter-block .batter-body .page-head { display: none; }

/* ----- Contact-quality projection ----- */
.cq-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
           gap: 14px; margin-top: 4px; }
.cq-panel h3 { margin: 0 0 6px 0; font-size: 12px; font-weight: 700; color: var(--muted);
               text-transform: uppercase; letter-spacing: 0.4px; }
.cq-panel table { font-size: 12px; }
.cq-panel table thead th { padding: 5px 8px; font-size: 11px; }
.cq-panel table tbody td { padding: 5px 8px; }
.cq-panel .sweet td:first-child { background: #f1f5f9; font-weight: 600; }
.cq-panel .sweet td:first-child::before { content: '\\2605  '; color: #2563eb; }
"""


def _h(text) -> str:
    """HTML-escape a value (None / NaN -> em dash)."""
    if text is None:
        return "&mdash;"
    if isinstance(text, float) and math.isnan(text):
        return "&mdash;"
    return html.escape(str(text))


def edge_class(value, baseline, scale, batter_favors_high: bool = True) -> str:
    """Return a CSS class based on how far `value` deviates from `baseline`.

    `scale` sets the unit: 0.5*scale = mild edge, 1.5*scale = strong edge.
    `batter_favors_high=True` means high values favor the hitter (xwOBA, HR%).
    `False` means high values favor the pitcher (K%, In-play out%, Whiff%).
    """
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    delta = (value - baseline) * (1 if batter_favors_high else -1)
    if delta >= 1.5 * scale:
        return "bat-edge-strong"
    if delta >= 0.5 * scale:
        return "bat-edge-mild"
    if delta <= -1.5 * scale:
        return "pit-edge-strong"
    if delta <= -0.5 * scale:
        return "pit-edge-mild"
    return ""


def _read_class(read_str: str) -> str:
    if "Edge: Hitter" in read_str:
        return "bat-edge-strong" if "vs Pitcher norm" not in read_str else "bat-edge-mild"
    if "Edge: Pitcher" in read_str:
        return "pit-edge-strong" if "vs Batter norm" not in read_str else "pit-edge-mild"
    return ""


# Per-outcome direction + scale for highlighting cells against the league baseline.
_OUTCOME_RULES = {
    "Strikeout":   (False, 0.04),
    "Walk":        (True,  0.025),
    "HBP":         (True,  0.005),
    "Single":      (True,  0.025),
    "Double":      (True,  0.012),
    "Triple":      (True,  0.005),
    "Home Run":    (True,  0.012),
    "In-play out": (False, 0.04),
    "Hit (any)":   (True,  0.04),
    "On-base":     (True,  0.04),
}


def _td(value: str, css_class: str = "") -> str:
    return f'<td class="{css_class}">{value}</td>' if css_class else f'<td>{value}</td>'


def to_html(
    batter_meta: dict, pitcher_meta: dict,
    season: int,
    bat_blended: pd.DataFrame, pit_blended: pd.DataFrame,
    proj: dict,
    v: dict, narrative_text: str,
    outcomes: pd.DataFrame,
    multi_pa: pd.DataFrame,
    tto_proj: pd.DataFrame,
    panel: pd.DataFrame, panel_notes: list[str],
    pitch_table: pd.DataFrame,
    count_mix_summary: pd.DataFrame,
    comps: pd.DataFrame,
    zone_overlay_df: pd.DataFrame,
    bat_track_overall: dict, bat_track_pitch: pd.DataFrame,
    sub: dict,
    bat_fav: pd.DataFrame, pit_fav: pd.DataFrame,
    decep: dict, handed_note: str,
    align: dict,
    contact_quality: dict | None = None,
    recent_form: dict | None = None,
    body_only: bool = False,
) -> str:
    parts: list[str] = []
    title = f"{batter_meta['name']} vs {pitcher_meta['name']} - matchup"

    if not body_only:
        parts.append("<!doctype html>")
        parts.append('<html lang="en">')
        parts.append("<head>")
        parts.append('<meta charset="utf-8">')
        parts.append(f"<title>{_h(title)}</title>")
        parts.append(f"<style>{_HTML_CSS}</style>")
        parts.append("</head>")
        parts.append("<body>")
        parts.append('<main class="container">')

    # ----- Header -----
    bat_counts = _per_season_counts(bat_blended)
    pit_counts = _per_season_counts(pit_blended)
    weight_str = ", ".join(
        f"{season - off} &times;{w:g}" for off, w in enumerate(SEASON_WEIGHTS)
    )
    bat_rows_str = "+".join(str(bat_counts.get(w, 0)) for w in SEASON_WEIGHTS)
    pit_rows_str = "+".join(str(pit_counts.get(w, 0)) for w in SEASON_WEIGHTS)

    parts.append('<header class="page-head">')
    parts.append(
        f'<h1>{_h(batter_meta["name"])} '
        f'<span class="badge">{_h(batter_meta["stand"])}HB</span> '
        f'vs {_h(pitcher_meta["name"])} '
        f'<span class="badge">{_h(pitcher_meta["p_throws"])}HP</span></h1>'
    )
    parts.append(
        f'<div class="meta">window: {weight_str} &middot; '
        f'batter rows {bat_rows_str}, pitcher rows {pit_rows_str}</div>'
    )
    parts.append("</header>")

    # ----- Verdict -----
    parts.append('<section class="card">')
    parts.append("<h2>Verdict</h2>")
    parts.append(f'<p class="narrative">{_h(narrative_text)}</p>')

    proj_x_h = float(proj.get("xwOBA", float("nan")))
    proj_a_h = float(proj.get("xBA", float("nan"))) if proj.get("xBA") is not None else float("nan")
    proj_s_h = float(proj.get("xSLG", float("nan"))) if proj.get("xSLG") is not None else float("nan")
    cls_x_h = edge_class(proj_x_h, LG_XWOBA, 0.030, True)
    cls_a_h = edge_class(proj_a_h, LG_XBA, 0.025, True)
    cls_s_h = edge_class(proj_s_h, LG_XSLG, 0.040, True)
    parts.append(
        '<table class="proj-slash" style="margin-bottom:10px"><thead><tr>'
        '<th>Projected slash</th><th>xwOBA</th><th>xBA</th><th>xSLG</th>'
        '</tr></thead><tbody><tr>'
        f'<td style="text-align:left;color:var(--muted)">'
        f'lg {fmt3(LG_XWOBA)} / {fmt3(LG_XBA)} / {fmt3(LG_XSLG)}</td>'
        f'{_td(fmt3(proj_x_h), cls_x_h)}'
        f'{_td(fmt3(proj_a_h), cls_a_h)}'
        f'{_td(fmt3(proj_s_h), cls_s_h)}'
        '</tr></tbody></table>'
    )

    parts.append("<table>")
    parts.append("<thead><tr><th>Frame</th><th>Projected xwOBA</th>"
                 "<th>Baseline</th><th>&Delta; (wOBA pts)</th><th>Read</th></tr></thead>")
    parts.append("<tbody>")
    for label, key in (("vs league avg", "vs_league"),
                       (f"vs {pitcher_meta['possessive']} baseline", "vs_pitcher"),
                       (f"vs {batter_meta['possessive']} baseline", "vs_batter")):
        proj_x, base, diff, read = v[key]
        cls = _read_class(read)
        sign = "+" if diff >= 0 else ""
        parts.append(
            "<tr>"
            f"<td>{_h(label)}</td>"
            f"<td>{proj_x:.3f}</td>"
            f"<td>{base:.3f}</td>"
            f"{_td(f'{sign}{diff:.0f}', cls)}"
            f"{_td(_h(read), cls)}"
            "</tr>"
        )
    parts.append("</tbody></table></section>")

    # ----- Per-PA outcomes -----
    parts.append('<section class="card">')
    parts.append("<h2>Per-PA outcome distribution</h2>")
    parts.append("<table><thead><tr>"
                 "<th>Outcome</th><th>Prob</th><th>American</th><th>League</th>"
                 "</tr></thead><tbody>")
    for _, r in outcomes.iterrows():
        outcome = r["Outcome"]
        prob = r["Prob"]
        lg = r["League"]
        am = american_odds(prob)
        favor_high, scale = _OUTCOME_RULES.get(outcome, (True, 0.04))
        cls = edge_class(prob, lg, scale, favor_high)
        if outcome == "Hit (any)":
            parts.append(
                '<tr class="divider-row"><td colspan="4">rollups</td></tr>'
            )
        parts.append(
            "<tr>"
            f"<td>{_h(outcome)}</td>"
            f"{_td(f'{prob*100:.1f}%', cls)}"
            f"<td>{_h(am)}</td>"
            f"<td>{lg*100:.1f}%</td>"
            "</tr>"
        )
    parts.append("</tbody></table></section>")

    # ----- Multi-PA outlook -----
    parts.append('<section class="card">')
    parts.append("<h2>Multi-PA outlook (2 / 3 / 4 PAs)</h2>")
    parts.append('<p class="subtitle">Chance the outcome happens at least once across N PAs '
                 '(with American odds), and expected count across N PAs (assumes independent PAs).</p>')
    if multi_pa.empty:
        parts.append('<p class="note">no outcome distribution available</p>')
    else:
        parts.append('<table class="multi-pa">')
        parts.append(
            "<thead>"
            "<tr>"
            '<th rowspan="2" class="outcome-col">Outcome</th>'
            '<th colspan="3" class="pa-group">Over 2 PAs</th>'
            '<th colspan="3" class="pa-group">Over 3 PAs</th>'
            '<th colspan="3" class="pa-group">Over 4 PAs</th>'
            "</tr>"
            "<tr>"
            "<th>Chance &ge;1</th><th>Odds</th><th>Avg #</th>"
            "<th>Chance &ge;1</th><th>Odds</th><th>Avg #</th>"
            "<th>Chance &ge;1</th><th>Odds</th><th>Avg #</th>"
            "</tr>"
            "</thead><tbody>"
        )
        for _, r in multi_pa.iterrows():
            outcome = r["Outcome"]
            favor_high, scale = _OUTCOME_RULES.get(outcome, (True, 0.04))
            lg_p = LG_OUTCOMES.get({
                "Strikeout": "K", "Walk": "BB", "HBP": "HBP",
                "Single": "1B", "Double": "2B", "Triple": "3B",
                "Home Run": "HR", "In-play out": "BIP_out",
            }.get(outcome, ""), 0.0)
            row_cells = [f"<td>{_h(outcome)}</td>"]
            for n in (2, 3, 4):
                p_ge1 = float(r[f"P(>=1) in {n} PA"])
                ec = float(r[f"E[count] in {n} PA"])
                if outcome in ("Hit (any)", "On-base"):
                    lg_n = 1.0 - (1.0 - {"Hit (any)": 0.22, "On-base": 0.317}.get(outcome, 0.0)) ** n
                else:
                    lg_n = 1.0 - (1.0 - lg_p) ** n
                cls = edge_class(p_ge1, lg_n, scale * n / 1.5, favor_high)
                row_cells.append(_td(f"{p_ge1*100:.1f}%", cls))
                row_cells.append(f'<td class="odds">{_h(american_odds(p_ge1))}</td>')
                row_cells.append(f'<td class="avg">{ec:.2f}</td>')
            parts.append("<tr>" + "".join(row_cells) + "</tr>")
        parts.append("</tbody></table>")
    parts.append("</section>")

    # ----- Headline by TTO -----
    parts.append('<section class="card">')
    parts.append("<h2>Projection by times through the order</h2>")
    if tto_proj.empty:
        parts.append('<p class="note">no TTO data available</p>')
    else:
        parts.append("<table><thead><tr>"
                     "<th>TTO</th><th>Proj xwOBA</th><th>Proj xBA</th>"
                     "<th>Proj xSLG</th><th>Proj K %</th>"
                     "<th>Proj Whiff %</th><th>Proj Hard Hit %</th><th>Sample (eff PA)</th>"
                     "</tr></thead><tbody>")
        for _, r in tto_proj.iterrows():
            x = float(r["Projected xwOBA"])
            a = r.get("Projected xBA")
            s = r.get("Projected xSLG")
            cls_x = edge_class(x, LG_XWOBA, 0.030, batter_favors_high=True)
            cls_a = edge_class(a, LG_XBA, 0.025, batter_favors_high=True)
            cls_s = edge_class(s, LG_XSLG, 0.040, batter_favors_high=True)
            parts.append(
                "<tr>"
                f"<td>{int(r['TTO'])}</td>"
                f"{_td(f'{x:.3f}', cls_x)}"
                f"{_td(fmt3(a), cls_a)}"
                f"{_td(fmt3(s), cls_s)}"
                f"<td>{r['Projected K %']:.1f}%</td>"
                f"<td>{r['Projected Whiff %']:.1f}%</td>"
                f"<td>{r['Projected Hard Hit %']:.1f}%</td>"
                f"<td>{r['Sample (eff PA)']:.1f}</td>"
                "</tr>"
            )
        parts.append("</tbody></table>")
    parts.append("</section>")

    # ----- Discipline panel -----
    parts.append('<section class="card">')
    parts.append("<h2>Side-by-side profile (platoon-filtered)</h2>")
    parts.append("<table><thead><tr>"
                 "<th>Metric</th><th>Batter</th><th>Pitcher (allowed)</th>"
                 "</tr></thead><tbody>")
    for _, r in panel.iterrows():
        metric = r["Metric"]
        b_val = r["Batter (vs same hand)"]
        p_val = r["Pitcher (vs same hand)"]
        # Color cells by whether higher is better for the hitter or pitcher.
        # For metrics like K%/Whiff%/GB%: higher in pitcher column = pitcher edge,
        # higher in batter column = bad for batter (color it negative for batter).
        # Both columns use the same direction: high Whiff% is bad for the batter
        # whether the cell shows the batter's own rate or what the pitcher generates.
        # High HardHit% likewise is good for the batter on either side of the panel.
        if metric in ("xwOBA", "xBA", "xSLG"):
            ref, scale = {
                "xwOBA": (LG_XWOBA, 0.030),
                "xBA":   (LG_XBA,   0.025),
                "xSLG":  (LG_XSLG,  0.040),
            }[metric]
            cls_b = edge_class(b_val, ref, scale, batter_favors_high=True)
            cls_p = edge_class(p_val, ref, scale, batter_favors_high=True)
            parts.append(
                "<tr>"
                f"<td>{_h(metric)}</td>"
                f"{_td(fmt3(b_val), cls_b)}"
                f"{_td(fmt3(p_val), cls_p)}"
                "</tr>"
            )
            continue
        rules = {
            "Chase %":    (False, 4),
            "Whiff %":    (False, 4),
            "K %":        (False, 4),
            "BB %":       (True,  2.5),
            "Hard Hit %": (True,  5),
            "Barrel %":   (True,  2),
            "GB %":       (False, 6),
            "Air %":      (True,  6),
        }
        favor_high, scale = rules.get(metric, (True, 5))
        cls_b = edge_class(b_val, _ref_lg_pct(metric), scale, batter_favors_high=favor_high)
        cls_p = edge_class(p_val, _ref_lg_pct(metric), scale, batter_favors_high=favor_high)
        parts.append(
            "<tr>"
            f"<td>{_h(metric)}</td>"
            f"{_td(f'{b_val:.1f}%', cls_b)}"
            f"{_td(f'{p_val:.1f}%', cls_p)}"
            "</tr>"
        )
    parts.append("</tbody></table>")
    if panel_notes:
        parts.append("<ul class='note-list'>")
        for n in panel_notes:
            parts.append(f"<li>{_h(n)}</li>")
        parts.append("</ul>")
    parts.append("</section>")

    # ----- Recent form -----
    if recent_form:
        windows = recent_form.get("windows", RECENT_FORM_WINDOWS)
        platoon_labels = recent_form.get("platoon_labels", {})
        parts.append('<section class="card">')
        parts.append("<h2>Recent form (rolling)</h2>")

        def _emit_row(side_label: str, window_label: str, snap: dict | None) -> None:
            if snap is None:
                parts.append(
                    "<tr>"
                    f"<td>{_h(side_label)}</td>"
                    f"<td>{_h(window_label)}</td>"
                    "<td>&mdash;</td><td>&mdash;</td><td>&mdash;</td><td>&mdash;</td>"
                    "<td>&mdash;</td><td>&mdash;</td><td>&mdash;</td><td>&mdash;</td>"
                    "</tr>"
                )
                return
            xwoba = snap.get("xwOBA")
            xba = snap.get("xBA")
            xslg = snap.get("xSLG")
            cls_x = edge_class(xwoba, LG_XWOBA, 0.030, True)
            cls_ba = edge_class(xba, LG_XBA, 0.025, True)
            cls_slg = edge_class(xslg, LG_XSLG, 0.040, True)
            n_pa = snap.get("n_pa", 0)
            parts.append(
                "<tr>"
                f"<td>{_h(side_label)}</td>"
                f"<td>{_h(window_label)}</td>"
                f"<td>{n_pa:.0f}</td>"
                f"{_td(fmt3(xwoba), cls_x)}"
                f"{_td(fmt3(xba), cls_ba)}"
                f"{_td(fmt3(xslg), cls_slg)}"
                f"<td>{snap['K_pct']*100:.1f}</td>"
                f"<td>{snap['BB_pct']*100:.1f}</td>"
                f"<td>{snap['Whiff_pct']*100:.1f}</td>"
                f"<td>{snap['HardHit_pct']*100:.1f}</td>"
                "</tr>"
            )

        def _emit_view(view: dict, sub_heading: str) -> None:
            parts.append(f"<h3>{_h(sub_heading)}</h3>")
            parts.append("<table><thead><tr>"
                         "<th>Side</th><th>Window</th><th>n PA</th>"
                         "<th>xwOBA</th><th>xBA</th><th>xSLG</th>"
                         "<th>K %</th><th>BB %</th>"
                         "<th>Whiff %</th><th>Hard Hit %</th>"
                         "</tr></thead><tbody>")
            for side_key, side_label in (("batter", "Batter"), ("pitcher", "Pitcher")):
                side = view.get(side_key, {}) or {}
                _emit_row(side_label, "season", side.get("season"))
                for w in windows:
                    snap = side.get(f"d{int(w)}")
                    _emit_row(side_label, f"last {int(w)}d", snap)
            parts.append("</tbody></table>")

        overall_view = recent_form.get("overall") or {
            "batter": {k: recent_form.get("batter", {}).get(k) for k in ("season",) + tuple(f"d{int(w)}" for w in windows)},
            "pitcher": {k: recent_form.get("pitcher", {}).get(k) for k in ("season",) + tuple(f"d{int(w)}" for w in windows)},
        }
        _emit_view(overall_view, "Overall (no platoon filter)")

        platoon_view = recent_form.get("platoon")
        if platoon_view:
            bat_pl = platoon_labels.get("batter") or "vs same hand"
            pit_pl = platoon_labels.get("pitcher") or "vs same hand"
            _emit_view(platoon_view,
                       f"Platoon \u2014 Batter {bat_pl}, Pitcher {pit_pl}")

        parts.append(
            "<p class='note'>Both tables use raw current-season PA (no season-blend, "
            "no recency decay). <strong>Overall</strong> is un-platoon-filtered and "
            "matches the player's year-to-date totals on BR / Savant &mdash; the "
            "natural \"how is this player going right now\" read. <strong>Platoon"
            "</strong> restricts to tonight's matchup hand, so it's apples-to-apples "
            "with the rest of the report (which is also platoon-filtered). Rolling "
            f"rows are shown only when n PA &ge; {RECENT_FORM_MIN_PA:.0f}; the season "
            "row is suppressed if there's no qualifying data. The verdict's hot/cold "
            "tail is anchored on the overall view.</p>"
        )
        parts.append("</section>")

    # ----- Pitch-mix projection -----
    parts.append('<section class="card">')
    parts.append("<h2>Pitch-mix projection</h2>")
    if pitch_table.empty:
        parts.append('<p class="note">no arsenal data</p>')
    else:
        parts.append("<table><thead><tr>"
                     "<th>Pitch</th><th>Marginal Usage %</th>"
                     "<th>Batter xwOBA</th><th>Pitcher xwOBA allowed</th>"
                     "<th style='text-align:left'>Pitcher BB-mix</th>"
                     "<th>Adj batter xwOBA</th>"
                     "<th>Adj &Delta; (pts)</th>"
                     "<th>Projected xwOBA</th><th>Projected xBA</th>"
                     "<th>Projected xSLG</th><th>Projected Whiff %</th>"
                     "</tr></thead><tbody>")
        for _, r in pitch_table.iterrows():
            b_x = float(r["Batter xwOBA"])
            p_x = float(r["Pitcher xwOBA allowed"])
            proj_x = float(r["Projected xwOBA"])
            proj_a = r.get("Projected xBA")
            proj_s = r.get("Projected xSLG")
            b_x_adj = float(r.get("Adj batter xwOBA", b_x))
            adj_pts = float(r.get("Adj delta (pts)", 0.0))
            cls_b = edge_class(b_x, LG_XWOBA, 0.030, True)
            cls_p = edge_class(p_x, LG_XWOBA, 0.030, True)
            cls_proj = edge_class(proj_x, LG_XWOBA, 0.030, True)
            cls_proj_a = edge_class(proj_a, LG_XBA, 0.025, True)
            cls_proj_s = edge_class(proj_s, LG_XSLG, 0.040, True)
            cls_b_adj = edge_class(b_x_adj, LG_XWOBA, 0.030, True)
            # Treat the adj delta the same as a per-pitch xwOBA delta:
            # negative = pitcher edge (GB-mix discount on the batter), positive = hitter edge.
            cls_adj = (
                "pit-edge-strong" if adj_pts <= -25
                else "pit-edge-mild" if adj_pts <= -10
                else "bat-edge-strong" if adj_pts >= 25
                else "bat-edge-mild" if adj_pts >= 10
                else ""
            )
            mix_str = str(r.get("Pitcher BB-mix", ""))
            parts.append(
                "<tr>"
                f"<td>{_h(r['Pitch'])}</td>"
                f"<td>{r['Marginal Usage %']:.1f}</td>"
                f"{_td(f'{b_x:.3f}', cls_b)}"
                f"{_td(f'{p_x:.3f}', cls_p)}"
                f"<td style='text-align:left;font-size:12px;color:var(--muted)'>{_h(mix_str)}</td>"
                f"{_td(f'{b_x_adj:.3f}', cls_b_adj)}"
                f"{_td(f'{adj_pts:+.0f}', cls_adj)}"
                f"{_td(f'{proj_x:.3f}', cls_proj)}"
                f"{_td(fmt3(proj_a), cls_proj_a)}"
                f"{_td(fmt3(proj_s), cls_proj_s)}"
                f"<td>{r['Projected Whiff %']:.1f}</td>"
                "</tr>"
            )
        parts.append("</tbody></table>")
        parts.append('<p class="note">'
                     'Adj &Delta; (pts) is the bb_type mix discount applied to the batter\'s '
                     'per-pitch xwOBA: a 100 mph 0&deg; grounder produces a much lower xwOBA '
                     'than a 100 mph 25&deg; line drive, so when the pitcher\'s induced '
                     'bb_type mix is more grounder-heavy than the batter\'s career mix on '
                     'this pitch, the projection is shifted down by the equivalent per-PA '
                     'amount (and vice versa for air-ball-heavy pitchers).'
                     '</p>')
    parts.append("</section>")

    # ----- Contact-quality projection -----
    if contact_quality and contact_quality.get("available"):
        cq = contact_quality
        parts.append('<section class="card">')
        parts.append("<h2>Contact-quality projection</h2>")
        n_bbe = cq.get("n_career_bbe", 0.0)
        mev_c = cq.get("mean_ev_career", float("nan"))
        mev_p = cq.get("mean_ev_proj", float("nan"))
        mla_c = cq.get("mean_la_career", float("nan"))
        mla_p = cq.get("mean_la_proj", float("nan"))
        d_ev = (mev_p - mev_c) if (not math.isnan(mev_p) and not math.isnan(mev_c)) else float("nan")
        d_la = (mla_p - mla_c) if (not math.isnan(mla_p) and not math.isnan(mla_c)) else float("nan")

        sub_bits = [f"weighted from <b>{n_bbe:.0f}</b> career BBE"]
        if not math.isnan(mev_c):
            sub_bits.append(
                f"mean EV career <b>{mev_c:.1f}</b> &rarr; proj <b>{mev_p:.1f}</b> "
                f"({d_ev:+.1f} mph)"
            )
        if not math.isnan(mla_c):
            sub_bits.append(
                f"mean LA career <b>{mla_c:.1f}&deg;</b> &rarr; proj <b>{mla_p:.1f}&deg;</b> "
                f"({d_la:+.1f}&deg;)"
            )
        parts.append(f'<p class="subtitle">{" &middot; ".join(sub_bits)}</p>')

        def _cq_table(rows: list[tuple[str, float, float, bool]]) -> str:
            """Render a Career % / Projected % / Δ (pp) table.

            Each row is (label, career_share_0to1, proj_share_0to1, is_sweet)."""
            buf = ['<table><thead><tr>'
                   '<th>Bin</th><th>Career %</th><th>Projected %</th>'
                   '<th>&Delta; (pp)</th></tr></thead><tbody>']
            for lab, c, p, sweet in rows:
                c100 = c * 100
                p100 = p * 100
                d = p100 - c100
                tr_cls = ' class="sweet"' if sweet else ''
                buf.append(
                    f'<tr{tr_cls}><td>{_h(lab)}</td>'
                    f'<td>{c100:.1f}</td><td>{p100:.1f}</td>'
                    f'<td>{d:+.1f}</td></tr>'
                )
            buf.append('</tbody></table>')
            return ''.join(buf)

        qoc_rows = [
            (QOC_LABELS[q], cq["qoc_career"].get(q, 0.0),
             cq["qoc_proj"].get(q, 0.0), False)
            for q in sorted(QOC_LABELS)
        ]
        ev_rows = [
            (lab, c, p, False)
            for lab, c, p in zip(EV_LABELS, cq["ev_career"], cq["ev_proj"])
        ]
        la_rows = [
            (lab, c, p, i in LA_SWEET_SPOT_BINS)
            for i, (lab, c, p) in enumerate(zip(LA_LABELS, cq["la_career"], cq["la_proj"]))
        ]

        parts.append('<div class="cq-grid">')
        for heading, table_html in (
            ("Quality of contact (Statcast 6-bin)", _cq_table(qoc_rows)),
            ("Exit velocity (mph)", _cq_table(ev_rows)),
            ("Launch angle (deg)", _cq_table(la_rows)),
        ):
            parts.append('<div class="cq-panel">')
            parts.append(f'<h3>{_h(heading)}</h3>')
            parts.append(table_html)
            parts.append('</div>')
        parts.append('</div>')

        parts.append('<p class="note">'
                     'Career = batter\'s full platoon-filtered BBE distribution. '
                     'Projected re-weights each pitch-specific BBE distribution by the '
                     'pitcher\'s marginal pitch usage; pitches with &lt;5 weighted BBE '
                     'fall back to the batter\'s overall mix. &Delta; (pp) is the '
                     'projected share minus career share, in percentage points. '
                     '&#9733; rows mark the highlighted 10&ndash;26&deg; barrel zone '
                     '(overlaps Statcast\'s broader 8&ndash;32&deg; sweet-spot range).'
                     '</p>')
        parts.append("</section>")

    # ----- Count-state mix -----
    parts.append('<section class="card">')
    parts.append("<h2>Count-state pitch mix (pitcher, vs same hand)</h2>")
    if count_mix_summary.empty:
        parts.append('<p class="note">no count-state data</p>')
    else:
        cols = list(count_mix_summary.columns)
        parts.append("<table><thead><tr>")
        for c in cols:
            parts.append(f"<th>{_h(c)}</th>")
        parts.append("</tr></thead><tbody>")
        for _, r in count_mix_summary.iterrows():
            parts.append("<tr>")
            for c in cols:
                v_cell = r[c]
                if isinstance(v_cell, (int, float)) and not (isinstance(v_cell, float) and math.isnan(v_cell)):
                    parts.append(f"<td>{v_cell:.1f}</td>")
                else:
                    parts.append(f"<td>{_h(v_cell)}</td>")
            parts.append("</tr>")
        parts.append("</tbody></table>")
    parts.append("</section>")

    # ----- Shape comps -----
    parts.append('<section class="card">')
    parts.append("<h2>Shape-aware comps from batter history</h2>")
    if comps.empty:
        parts.append('<p class="note">no comps available</p>')
    else:
        parts.append("<table><thead><tr>"
                     "<th>Pitch</th><th>Shape (eff velo / IVB / HB-in)</th>"
                     "<th>n comps (eff)</th><th>Whiff %</th><th>xwOBA</th>"
                     "<th>xBA</th><th>xSLG</th>"
                     "<th>Hard Hit %</th><th>Confidence</th>"
                     "</tr></thead><tbody>")
        for _, r in comps.iterrows():
            xw = r["xwOBA"]
            xa = r.get("xBA")
            xs = r.get("xSLG")
            cls_x = edge_class(xw, LG_XWOBA, 0.030, True)
            cls_a = edge_class(xa, LG_XBA, 0.025, True)
            cls_s = edge_class(xs, LG_XSLG, 0.040, True)
            whiff = r["Whiff %"]
            cls_w = edge_class(whiff / 100.0 if not pd.isna(whiff) else float("nan"),
                               LG_WHIFF, 0.04, batter_favors_high=False)
            conf_pill = f'<span class="pill">{_h(r["Confidence"])}</span>'
            parts.append(
                "<tr>"
                f"<td>{_h(r['Pitch'])}</td>"
                f"<td>{_h(r['Shape (eff velo / IVB / HB-in)'])}</td>"
                f"<td>{r['n comps (eff)']:.1f}</td>"
                f"{_td(_pct_or_dash(whiff), cls_w)}"
                f"{_td(fmt3(xw), cls_x)}"
                f"{_td(fmt3(xa), cls_a)}"
                f"{_td(fmt3(xs), cls_s)}"
                f"<td>{_pct_or_dash(r['Hard Hit %'])}</td>"
                f"<td>{conf_pill}</td>"
                "</tr>"
            )
        parts.append("</tbody></table>")
    parts.append("</section>")

    # ----- Zone overlay -----
    parts.append('<section class="card">')
    parts.append("<h2>Zone overlay</h2>")
    if zone_overlay_df.empty:
        parts.append('<p class="note">no zone data</p>')
    else:
        parts.append("<table><thead><tr>"
                     "<th>Pitch</th><th>In-zone %</th>"
                     "<th>Top zones (attack share)</th><th>Intersection xwOBA</th>"
                     "<th>Intersection xBA</th><th>Intersection xSLG</th>"
                     "<th>Coverage %</th>"
                     "</tr></thead><tbody>")
        for _, r in zone_overlay_df.iterrows():
            ix = r["Intersection xwOBA"]
            ia = r.get("Intersection xBA")
            iss = r.get("Intersection xSLG")
            cls_ix = edge_class(ix if not pd.isna(ix) else float("nan"),
                                LG_XWOBA, 0.030, True)
            cls_ia = edge_class(ia if (ia is not None and not pd.isna(ia)) else float("nan"),
                                LG_XBA, 0.025, True)
            cls_is = edge_class(iss if (iss is not None and not pd.isna(iss)) else float("nan"),
                                LG_XSLG, 0.040, True)
            cov = r.get("Coverage %")
            cov_str = (f"{cov:.0f}"
                       if cov is not None and not (isinstance(cov, float) and math.isnan(cov))
                       else "—")
            parts.append(
                "<tr>"
                f"<td>{_h(r['Pitch'])}</td>"
                f"<td>{r['In-zone %']:.1f}</td>"
                f"<td>{_h(r['Top zones'])}</td>"
                f"{_td(fmt3(ix), cls_ix)}"
                f"{_td(fmt3(ia), cls_ia)}"
                f"{_td(fmt3(iss), cls_is)}"
                f"<td>{cov_str}</td>"
                "</tr>"
            )
        parts.append("</tbody></table>")
        parts.append("<p class='note'>Coverage % = share of the pitcher's attack on this pitch landing in zones with batter data; remainder imputed at league baselines (xwOBA 0.315, xBA 0.245, xSLG 0.405) so two batters with different coverage maps stay comparable.</p>")
    parts.append("</section>")

    # ----- Bat tracking -----
    parts.append('<section class="card">')
    parts.append("<h2>Bat-tracking interaction</h2>")
    if not bat_track_overall:
        parts.append('<p class="note">no bat-tracking data in this window</p>')
    else:
        if bat_track_overall.get("attack_angle") is not None:
            parts.append('<div class="kv">')
            parts.append(f'<div>bat speed <b>{bat_track_overall["bat_speed"]:.1f}</b> mph</div>')
            parts.append(f'<div>swing length <b>{bat_track_overall["swing_length"]:.1f}</b> ft</div>')
            parts.append(f'<div>attack angle <b>{bat_track_overall["attack_angle"]:.1f}&deg;</b></div>')
            parts.append('</div>')
        if not bat_track_pitch.empty:
            parts.append("<table><thead><tr>"
                         "<th>Pitch</th><th>VAA (deg)</th>"
                         "<th>Bat attack (deg)</th><th>Swings (n)</th>"
                         "<th>Match gap (deg)</th><th>Note</th>"
                         "</tr></thead><tbody>")
            for _, r in bat_track_pitch.iterrows():
                gap = r["Match gap (deg)"]
                if isinstance(gap, (int, float)) and not math.isnan(gap):
                    if abs(gap) > MATCH_THRESHOLD_DEG * 2:
                        cls_gap = "pit-edge-strong"
                    elif abs(gap) > MATCH_THRESHOLD_DEG:
                        cls_gap = "pit-edge-mild"
                    else:
                        cls_gap = "bat-edge-mild"
                    gap_str = f"{gap:+.1f}"
                else:
                    cls_gap = ""
                    gap_str = "&mdash;"
                attack = r["Bat attack (deg)"]
                if isinstance(attack, (int, float)) and not math.isnan(attack):
                    attack_str = f"{attack:+.1f}"
                    if r["Fallback"]:
                        attack_str += "*"
                else:
                    attack_str = "&mdash;"
                parts.append(
                    "<tr>"
                    f"<td>{_h(r['Pitch'])}</td>"
                    f"<td>{r['VAA (deg)']:+.1f}</td>"
                    f"<td>{attack_str}</td>"
                    f"<td>{r['Swings (n)']:.0f}</td>"
                    f"{_td(gap_str, cls_gap)}"
                    f"<td>{_h(r['Note'])}</td>"
                    "</tr>"
                )
            parts.append("</tbody></table>")
            if bat_track_pitch["Fallback"].any():
                parts.append(
                    f'<p class="note">* fewer than {MIN_SWINGS_PER_PITCH:.0f} swings '
                    "on this pitch type; showing batter's overall attack angle as a fallback.</p>"
                )
            parts.append(
                '<p class="note">Match gap = bat attack angle + pitch VAA. '
                f'|gap| &le; {MATCH_THRESHOLD_DEG:.0f}&deg; = on plane; '
                'positive = swing steeper than pitch (under risk); '
                'negative = swing flatter than pitch (top risk).</p>'
            )
    parts.append("</section>")

    # ----- Sub-profiles -----
    fp = sub.get("first_pitch")
    ts = sub.get("two_strike")
    parts.append('<section class="card">')
    parts.append("<h2>First-pitch &amp; two-strike sub-profiles</h2>")
    if fp:
        parts.append('<div class="kv">')
        parts.append(f'<div>pitcher first-pitch strike% <b>{fp["pitcher_strike_pct"]*100:.1f}%</b></div>')
        parts.append(f'<div>batter first-pitch swing% <b>{fp["batter_swing_pct"]*100:.1f}%</b></div>')
        parts.append(f'<div>batter xwOBA on first-pitch swings <b>{fmt3(fp["batter_xwoba_on_swing"])}</b></div>')
        parts.append('</div>')
    if ts:
        parts.append('<div class="kv">')
        parts.append(f'<div>pitcher putaway% <b>{ts["pitcher_putaway_pct"]*100:.1f}%</b></div>')
        parts.append(f'<div>batter K% in 2-strike counts <b>{ts["batter_K_pct_2s"]*100:.1f}%</b></div>')
        parts.append(f'<div>batter xwOBA in 2-strike counts <b>{fmt3(ts["batter_xwoba_2s"])}</b></div>')
        parts.append('</div>')
        if ts.get("two_strike_mix"):
            mix_pills = " ".join(f'<span class="pill">{_h(k)} {v:.1f}%</span>'
                                 for k, v in ts["two_strike_mix"].items())
            parts.append(f'<p class="subtitle">Pitcher\'s two-strike mix: {mix_pills}</p>')
    parts.append("</section>")

    # ----- Edge analysis -----
    def _edge_row(r) -> str:
        b_x = float(r["Batter xwOBA"])
        p_x = float(r["Pitcher xwOBA allowed"])
        proj_x = float(r["Projected xwOBA"])
        proj_a = r.get("Projected xBA")
        proj_s = r.get("Projected xSLG")
        edge_pts = float(r["Edge (pts)"])
        cls_b = edge_class(b_x, LG_XWOBA, 0.030, True)
        cls_p = edge_class(p_x, LG_XWOBA, 0.030, True)
        cls_proj = edge_class(proj_x, LG_XWOBA, 0.030, True)
        cls_proj_a = edge_class(proj_a, LG_XBA, 0.025, True)
        cls_proj_s = edge_class(proj_s, LG_XSLG, 0.040, True)
        cls_edge = (
            "pit-edge-strong" if edge_pts <= -25
            else "pit-edge-mild" if edge_pts <= -10
            else "bat-edge-strong" if edge_pts >= 25
            else "bat-edge-mild" if edge_pts >= 10
            else ""
        )
        return (
            "<tr>"
            f"<td>{_h(r['Pitch'])}</td>"
            f"<td>{r['Usage %']:.1f}</td>"
            f"{_td(f'{b_x:.3f}', cls_b)}"
            f"{_td(f'{p_x:.3f}', cls_p)}"
            f"{_td(f'{proj_x:.3f}', cls_proj)}"
            f"{_td(fmt3(proj_a), cls_proj_a)}"
            f"{_td(fmt3(proj_s), cls_proj_s)}"
            f"{_td(f'{edge_pts:+.0f}', cls_edge)}"
            "</tr>"
        )

    edge_table_head = (
        "<table><thead><tr>"
        "<th>Pitch</th><th>Usage %</th><th>Batter xwOBA</th>"
        "<th>Pitcher xwOBA allowed</th><th>Projected xwOBA</th>"
        "<th>Projected xBA</th><th>Projected xSLG</th>"
        "<th>Edge (pts)</th>"
        "</tr></thead><tbody>"
    )

    parts.append('<section class="card">')
    parts.append("<h2>Edge analysis</h2>")
    parts.append('<p class="subtitle">Per-pitch matchup result vs league (additive '
                 'batter + pitcher projection minus league baseline). <b>Edge (pts)</b> '
                 'is the xwOBA projection delta in wOBA points; positive favors the '
                 'hitter, negative favors the pitcher. Tables only list pitches whose '
                 'net xwOBA projection actually leans in that direction; xBA / xSLG are '
                 'shown for context (each colored vs its own league baseline).</p>')
    parts.append('<p class="subtitle" style="margin-top:10px"><b>Pitches favoring the hitter</b></p>')
    if bat_fav.empty:
        parts.append('<p class="note">no pitch in the arsenal projects above league for this hitter</p>')
    else:
        parts.append(edge_table_head)
        for _, r in bat_fav.iterrows():
            parts.append(_edge_row(r))
        parts.append("</tbody></table>")
    parts.append('<p class="subtitle" style="margin-top:14px"><b>Pitches favoring the pitcher</b></p>')
    if pit_fav.empty:
        parts.append('<p class="note">no pitch in the arsenal projects below league for this hitter</p>')
    else:
        parts.append(edge_table_head)
        for _, r in pit_fav.iterrows():
            parts.append(_edge_row(r))
        parts.append("</tbody></table>")
    parts.append("</section>")

    # ----- Deception -----
    parts.append('<section class="card">')
    parts.append("<h2>Deception &amp; shape signature</h2>")
    if decep:
        parts.append('<div class="kv">')
        parts.append(f'<div>release-point cluster <b>{decep["cluster_stdev_in"]:.2f}</b> in '
                     f'<span class="pill">{_h(decep["deception_label"])}</span></div>')
        parts.append('</div>')
        if not decep["per_pitch"].empty:
            parts.append("<table><thead><tr>"
                         "<th>Pitch</th><th>&Delta; from release centroid (in)</th>"
                         "<th>Spin axis (deg)</th>"
                         "</tr></thead><tbody>")
            for _, r in decep["per_pitch"].iterrows():
                axis_v = "&mdash;" if pd.isna(r["Spin axis (deg)"]) else f"{r['Spin axis (deg)']:.0f}"
                parts.append(
                    "<tr>"
                    f"<td>{_h(r['Pitch'])}</td>"
                    f"<td>{r['Δ from centroid (in)']:.2f}</td>"
                    f"<td>{axis_v}</td>"
                    "</tr>"
                )
            parts.append("</tbody></table>")
    if handed_note:
        parts.append(f'<p class="note">{_h(handed_note)}</p>')
    parts.append("</section>")

    # ----- Defensive alignment -----
    parts.append('<section class="card">')
    parts.append("<h2>Defensive alignment</h2>")
    if not align:
        parts.append('<p class="note">no alignment data</p>')
    else:
        ba = align.get("batter_gb_babip")
        if ba is not None and not ba.empty:
            parts.append('<p class="subtitle">Batter ground-ball BABIP by infield alignment</p>')
            parts.append("<table><thead><tr>"
                         "<th>Alignment</th><th>GB BABIP</th><th>Sample (eff GB)</th>"
                         "</tr></thead><tbody>")
            for _, r in ba.iterrows():
                v_b = r["GB BABIP"]
                cls = edge_class(v_b, 0.240, 0.040, True)   # league GB BABIP ~ .240
                parts.append(
                    "<tr>"
                    f"<td>{_h(r['Alignment'])}</td>"
                    f"{_td(f'{v_b:.3f}', cls)}"
                    f"<td>{r['Sample (eff GB)']:.1f}</td>"
                    "</tr>"
                )
            parts.append("</tbody></table>")
        pa_mix = align.get("pitcher_alignment_mix")
        if pa_mix is not None and len(pa_mix):
            mix_pills = " ".join(f'<span class="pill">{_h(idx)} {val:.1f}%</span>'
                                 for idx, val in pa_mix.items())
            parts.append(f'<p class="subtitle" style="margin-top:14px">'
                         f'Pitcher\'s typical infield alignment usage: {mix_pills}</p>')
    parts.append("</section>")

    # ----- Notes / footer -----
    if not body_only:
        parts.append(_html_footer_notes(batter_meta, pitcher_meta, season))
        parts.append("</main>")
        parts.append(sortable_html())
        parts.append("</body></html>")
    return "\n".join(parts)


def _html_footer_notes(batter_meta: dict, pitcher_meta: dict, season: int) -> str:
    parts: list[str] = []
    parts.append("<footer>")
    parts.append("<ul class='note-list'>")
    parts.append(
        "<li>Headline projection combines a count-conditional pitch mix from the pitcher with "
        "the batter's per-pitch-type xwOBA / xBA / xSLG (additive vs league) and Whiff% / Hard Hit% (log5).</li>"
    )
    parts.append(
        f"<li>All inputs are platoon-filtered: batter rows restricted to {pitcher_meta['p_throws']}HP, "
        f"pitcher rows to {batter_meta['stand']}HB.</li>"
    )
    parts.append(
        "<li>Window blends "
        + ", ".join(
            f"{season - off} (weight {w:g})" for off, w in enumerate(SEASON_WEIGHTS)
        )
        + ". Tune <code>SEASON_WEIGHTS</code> at the top of <code>matchup.py</code>.</li>"
    )
    parts.append(
        "<li>Per-PA outcome shares are derived from the projection plus the batter's hit-type mix; "
        "cross-checked against projected xwOBA and reconciled if drift exceeds 5 pts.</li>"
    )
    parts.append(
        "<li>Cell coloring: <span class='bat-edge-strong' style='padding:1px 6px;border-radius:4px'>"
        "strong batter edge</span>, <span class='bat-edge-mild' style='padding:1px 6px;border-radius:4px'>"
        "mild batter edge</span>, <span class='pit-edge-mild' style='padding:1px 6px;border-radius:4px'>"
        "mild pitcher edge</span>, <span class='pit-edge-strong' style='padding:1px 6px;border-radius:4px'>"
        "strong pitcher edge</span> &mdash; relative to league baseline.</li>"
    )
    parts.append(
        "<li>Not modeled: park / weather, catcher framing, umpire zone, fatigue beyond TTO. "
        "Recent form is modeled (current-season row weights decay with a 30d half-life, "
        "and the Recent form panel above shows 14d / 30d rolling rates) but rolling samples "
        "remain noisy. Treat single-season cells as noisy.</li>"
    )
    parts.append("</ul>")
    parts.append("</footer>")
    return "\n".join(parts)


def _pct_or_dash(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "&mdash;"
    return f"{v:.1f}%"


# League-rate references used for highlighting the discipline panel.
def _ref_lg_pct(metric: str) -> float:
    return {
        "Chase %":     28.5,
        "Whiff %":     LG_WHIFF * 100,
        "K %":         LG_K_PCT * 100,
        "BB %":        LG_BB_PCT * 100,
        "Hard Hit %":  LG_HARD_HIT * 100,
        "Barrel %":    8.0,
        "GB %":        43.0,
        "Air %":       57.0,
    }.get(metric, 0.0)


# ---------- count-mix summary helper -------------------------------------

def count_mix_summary(pit_vs_bat: pd.DataFrame, top_n: int = 4) -> pd.DataFrame:
    """Compact pitch x count matrix: top counts as columns, top pitches as rows."""
    if pit_vs_bat.empty or "pitch_name" not in pit_vs_bat.columns:
        return pd.DataFrame()
    g = pit_vs_bat.dropna(subset=["pitch_name", "count_state"])
    if g.empty:
        return pd.DataFrame()

    pivot = (g.groupby(["pitch_name", "count_state"])["weight"]
              .sum()
              .unstack(fill_value=0.0))
    col_totals = pivot.sum(axis=0)
    pivot = pivot.div(col_totals.where(col_totals > 0, 1.0), axis=1) * 100  # pct of pitches in that count

    # Pick top N counts by raw frequency, plus 0-0, 0-2, 3-2 always.
    must = [c for c in ["0-0", "0-2", "3-2"] if c in pivot.columns]
    top_counts = list(col_totals.sort_values(ascending=False).head(top_n).index)
    cols = list(dict.fromkeys(must + top_counts))
    cols = [c for c in cols if c in pivot.columns]

    out = pivot[cols].copy()
    # Pick top pitch types by overall usage
    pitch_totals = pit_vs_bat.dropna(subset=["pitch_name"]).groupby("pitch_name")["weight"].sum()
    top_pitches = list(pitch_totals.sort_values(ascending=False).head(6).index)
    out = out.loc[[p for p in top_pitches if p in out.index]]

    out = out.round(1).reset_index()
    out = out.rename(columns={"pitch_name": "Pitch"})
    return out


# ---------- pipeline ------------------------------------------------------

def compute_matchup_pieces(batter_id: int, pitcher_id: int,
                           season: int = DEFAULT_SEASON) -> dict:
    """Run the analysis pipeline and return all computed pieces in a dict.

    The returned dict has every kwarg needed by `to_markdown` / `to_html`,
    plus a `season` key. Useful for the lineup driver, which renders many
    matchups with shared chrome.
    """
    bat_blended = load_blended_batter(batter_id, season)
    pit_blended = load_blended_pitcher(pitcher_id, season)

    if bat_blended.empty:
        raise SystemExit(f"No batter rows for id {batter_id}")
    if pit_blended.empty:
        raise SystemExit(f"No pitcher rows for id {pitcher_id}")

    batter_meta = _player_meta(bat_blended, "batter")
    pitcher_meta = _player_meta(pit_blended, "pitcher")

    # Capture pristine (pre-recency-decay) frames so the Recent form panel can
    # snapshot raw rolling-window rates without being skewed by the engine's
    # row-weight multiplier. _prepare() decays in place; we hand it a copy so
    # the originals stay flat-weighted.
    bat_pristine = bat_blended.copy()
    pit_pristine = pit_blended.copy()

    bat_blended = _prepare(bat_blended)
    pit_blended = _prepare(pit_blended)

    # Layer 0: platoon filters.
    if pitcher_meta["p_throws"]:
        bat_vs_pit = bat_blended[bat_blended["p_throws"] == pitcher_meta["p_throws"]].copy()
    else:
        bat_vs_pit = bat_blended.copy()
    if batter_meta["stand"]:
        pit_vs_bat = pit_blended[pit_blended["stand"] == batter_meta["stand"]].copy()
    else:
        pit_vs_bat = pit_blended.copy()

    if bat_vs_pit.empty:
        bat_vs_pit = bat_blended.copy()
    if pit_vs_bat.empty:
        pit_vs_bat = pit_blended.copy()

    batter_pt = per_pitch_type_table(bat_vs_pit)
    pitcher_pt = per_pitch_type_table(pit_vs_bat)
    batter_overall = overall_rates(bat_vs_pit)
    pitcher_overall = overall_rates(pit_vs_bat)

    marginal = count_conditional_marginal(pit_vs_bat, bat_vs_pit)
    bat_count_xwoba = batter_xwoba_by_count(bat_vs_pit)
    pit_count_dist = pitcher_count_state_distribution(pit_vs_bat)
    proj = project(batter_pt, pitcher_pt, batter_overall, pitcher_overall,
                    marginal,
                    batter_count_xwoba=bat_count_xwoba,
                    pitcher_count_dist=pit_count_dist)
    comps = shape_comps(bat_vs_pit, pitcher_pt)
    zone_df = zone_overlay(pit_vs_bat, bat_vs_pit, pitcher_pt)
    tto = tto_curve(pit_vs_bat)
    tto_proj = tto_projections(proj, tto, pitcher_overall)
    bat_track_overall, bat_track_pitch = bat_tracking(bat_vs_pit, pit_vs_bat, pitcher_pt)
    sub = count_subprofiles(bat_vs_pit, pit_vs_bat)
    panel = discipline_panel(batter_overall, pitcher_overall)
    panel_notes = discipline_notes(batter_overall, pitcher_overall)
    recent_form = recent_form_panel(bat_pristine, pit_pristine,
                                     batter_meta=batter_meta,
                                     pitcher_meta=pitcher_meta)
    recent_summary = recent_form_summary(recent_form)
    bat_fav, pit_fav = edge_analysis(batter_pt, pitcher_pt, marginal)
    decep = deception(pitcher_pt)
    handed_note = handedness_verdict(batter_meta["stand"], pitcher_meta["p_throws"])
    outcomes = outcome_distribution(proj, batter_pt, marginal)
    multi_pa = multi_pa_outlook(outcomes, ns=(2, 3, 4))

    v = verdict(proj["xwOBA"], LG_XWOBA, pitcher_overall["xwOBA"], batter_overall["xwOBA"])
    narrative_text = narrative(batter_meta, pitcher_meta,
                                proj, v, bat_fav, pit_fav, handed_note,
                                recent_summary=recent_summary)
    align = alignment_split(bat_vs_pit, pit_vs_bat)
    cm = count_mix_summary(pit_vs_bat)
    contact_quality = contact_quality_projection(bat_vs_pit, marginal)

    return {
        "batter_meta": batter_meta, "pitcher_meta": pitcher_meta,
        "season": season,
        "bat_blended": bat_blended, "pit_blended": pit_blended,
        "proj": proj,
        "v": v, "narrative_text": narrative_text,
        "outcomes": outcomes, "multi_pa": multi_pa, "tto_proj": tto_proj,
        "panel": panel, "panel_notes": panel_notes,
        "pitch_table": proj["pitch_table"], "count_mix_summary": cm,
        "comps": comps, "zone_overlay_df": zone_df,
        "bat_track_overall": bat_track_overall, "bat_track_pitch": bat_track_pitch,
        "sub": sub,
        "bat_fav": bat_fav, "pit_fav": pit_fav,
        "decep": decep, "handed_note": handed_note,
        "align": align,
        "contact_quality": contact_quality,
        "recent_form": recent_form,
        "batter_overall": batter_overall, "pitcher_overall": pitcher_overall,
    }


def _render_md_from_pieces(p: dict, body_only: bool = False) -> str:
    return to_markdown(
        p["batter_meta"], p["pitcher_meta"], p["season"],
        p["bat_blended"], p["pit_blended"],
        p["proj"], p["v"], p["narrative_text"],
        p["outcomes"], p["multi_pa"], p["tto_proj"],
        p["panel"], p["panel_notes"],
        p["pitch_table"], p["count_mix_summary"],
        p["comps"], p["zone_overlay_df"],
        p["bat_track_overall"], p["bat_track_pitch"],
        p["sub"], p["bat_fav"], p["pit_fav"],
        p["decep"], p["handed_note"], p["align"],
        contact_quality=p.get("contact_quality"),
        recent_form=p.get("recent_form"),
        body_only=body_only,
    )


def _render_html_from_pieces(p: dict, body_only: bool = False) -> str:
    return to_html(
        p["batter_meta"], p["pitcher_meta"], p["season"],
        p["bat_blended"], p["pit_blended"],
        p["proj"], p["v"], p["narrative_text"],
        p["outcomes"], p["multi_pa"], p["tto_proj"],
        p["panel"], p["panel_notes"],
        p["pitch_table"], p["count_mix_summary"],
        p["comps"], p["zone_overlay_df"],
        p["bat_track_overall"], p["bat_track_pitch"],
        p["sub"], p["bat_fav"], p["pit_fav"],
        p["decep"], p["handed_note"], p["align"],
        contact_quality=p.get("contact_quality"),
        recent_form=p.get("recent_form"),
        body_only=body_only,
    )


def analyze_matchup(batter_id: int, pitcher_id: int, season: int = DEFAULT_SEASON) -> tuple[str, str, str]:
    """Run the full matchup pipeline.

    Returns (output stem, markdown text, html text). The caller decides which
    files to write.
    """
    p = compute_matchup_pieces(batter_id, pitcher_id, season)
    md = _render_md_from_pieces(p)
    html_doc = _render_html_from_pieces(p)

    bat_slug = p["batter_meta"]["last"].lower().replace(" ", "_")
    pit_slug = p["pitcher_meta"]["last"].lower().replace(" ", "_")
    out_stem = f"{bat_slug}_vs_{pit_slug}_{season}_matchup"
    return out_stem, md, html_doc


# ---------- lineup pipeline -----------------------------------------------

DEFAULT_PA_PER_BATTER = 3


def _verdict_tag(delta_pts: float) -> tuple[str, str]:
    """(label, css class) for the lineup grid verdict cell.

    delta_pts is projected xwOBA minus league baseline, expressed in wOBA points.
    """
    a = abs(delta_pts)
    if a < 25:
        return "Even", "neutral"
    side = "Hitter" if delta_pts > 0 else "Pitcher"
    cls = "bat" if delta_pts > 0 else "pit"
    if a < 50:
        return f"Slight {side}", f"{cls}-edge-mild"
    if a < 100:
        return f"Edge {side}", f"{cls}-edge-mild"
    return f"Strong {side}", f"{cls}-edge-strong"


def _outcome_prob(outcomes: pd.DataFrame, label: str) -> float:
    if outcomes.empty:
        return 0.0
    row = outcomes[outcomes["Outcome"] == label]
    return float(row.iloc[0]["Prob"]) if len(row) else 0.0


def _lineup_summary_row(p: dict, spot: int) -> dict:
    bm = p["batter_meta"]
    proj = p["proj"]
    out = p["outcomes"]
    delta = (proj["xwOBA"] - LG_XWOBA) * 1000
    label, css = _verdict_tag(delta)

    bf, pf = p["bat_fav"], p["pit_fav"]
    best_pitch = (
        f"{bf.iloc[0]['Pitch']} ({bf.iloc[0]['Batter xwOBA']:.3f})"
        if len(bf) else "—"
    )
    worst_pitch = (
        f"{pf.iloc[0]['Pitch']} ({pf.iloc[0]['Batter xwOBA']:.3f})"
        if len(pf) else "—"
    )

    def _f(v, default=float("nan")) -> float:
        try:
            x = float(v)
            return x if not math.isnan(x) else default
        except (TypeError, ValueError):
            return default

    return {
        "spot": spot,
        "name": bm["name"],
        "stand": bm["stand"] or "?",
        "proj_xwoba": proj["xwOBA"],
        "proj_xba": float(proj.get("xBA", float("nan"))) if proj.get("xBA") is not None else float("nan"),
        "proj_xslg": float(proj.get("xSLG", float("nan"))) if proj.get("xSLG") is not None else float("nan"),
        "proj_xwoba_raw": float(proj.get("xwOBA_raw", proj["xwOBA"]) or proj["xwOBA"]),
        "bbtype_adj_pts": float(proj.get("xwOBA_adj_pts", 0.0) or 0.0),
        "delta_pts": delta,
        "k_pct": _outcome_prob(out, "Strikeout"),
        "bb_pct": _outcome_prob(out, "Walk"),
        "hr_pct": _outcome_prob(out, "Home Run"),
        "hit_pct": _outcome_prob(out, "Hit (any)"),
        "ob_pct": _outcome_prob(out, "On-base"),
        # Full 8-bucket outcome distribution for PA-level scoring in postgame.
        "proj_dist": {
            "K":  _outcome_prob(out, "Strikeout"),
            "BB": _outcome_prob(out, "Walk"),
            "HBP": _outcome_prob(out, "HBP"),
            "1B": _outcome_prob(out, "Single"),
            "2B": _outcome_prob(out, "Double"),
            "3B": _outcome_prob(out, "Triple"),
            "HR": _outcome_prob(out, "Home Run"),
            "Out": _outcome_prob(out, "In-play out"),
        },
        # Contact-quality + discipline projections (BABIP-independent eval).
        "proj_hardhit_pct": _f(proj.get("HardHit_pct")),
        "proj_whiff_pct": _f(proj.get("Whiff_pct")),
        "proj_xwoba_on_contact": _f(proj.get("xwOBA_bbtype", proj.get("xwOBA"))),
        "best_pitch": best_pitch,
        "worst_pitch": worst_pitch,
        "verdict_label": label,
        "verdict_css": css,
        "anchor": f"batter-{spot}",
    }


def _pitches_per_pa(df: pd.DataFrame | None, cap: int = 10) -> float | None:
    """Mean pitches per PA, with each PA winsorized at `cap` pitches.
    Returns None if data is missing/insufficient."""
    if df is None or df.empty:
        return None
    if "game_pk" not in df.columns or "at_bat_number" not in df.columns:
        return None
    counts = df.groupby(["game_pk", "at_bat_number"]).size()
    if counts.empty:
        return None
    return float(counts.clip(upper=cap).mean())


def project_pitcher_range(per_batter_pieces: list[dict],
                          summary_rows: list[dict],
                          bf_min: int = 20, bf_max: int = 30,
                          default_pp: float = 3.9) -> dict | None:
    """Project the pitcher's statline across a range of batters-faced totals.

    For each BF in [bf_min, bf_max], cycles through the lineup that many times,
    summing per-PA outcome probabilities and pitches. Each PA's pitch cost is
    the average of the pitcher's P/PA and the batter's P/PA (both winsorized at
    10). Returns one row per BF with Pitches/K/BB/Hits/HR/Outs.
    """
    if not per_batter_pieces or not summary_rows:
        return None

    pit_df = per_batter_pieces[0].get("pit_blended")
    starter_pit = pit_df
    if pit_df is not None and not pit_df.empty and "inning" in pit_df.columns and "game_pk" in pit_df.columns:
        starter_games = pit_df.loc[pit_df["inning"] == 1, "game_pk"].unique()
        if len(starter_games):
            starter_pit = pit_df[pit_df["game_pk"].isin(starter_games)]
    pitcher_pp = _pitches_per_pa(starter_pit) or default_pp

    batter_pp: list[float] = []
    for p in per_batter_pieces:
        bp = _pitches_per_pa(p.get("bat_blended")) or default_pp
        batter_pp.append(bp)

    n = len(summary_rows)
    rows = []
    cum_pitches = 0.0
    cum_k = cum_bb = cum_hits = cum_hr = cum_outs = 0.0
    for pa_idx in range(bf_max):
        i = pa_idx % n
        r = summary_rows[i]
        blended = (pitcher_pp + batter_pp[i]) / 2.0
        cum_pitches += blended
        cum_k += r["k_pct"]
        cum_bb += r["bb_pct"]
        cum_hits += r["hit_pct"]
        cum_hr += r["hr_pct"]
        cum_outs += (1.0 - r["ob_pct"])
        bf = pa_idx + 1
        if bf >= bf_min:
            rows.append({
                "bf": bf,
                "pitches": cum_pitches,
                "k": cum_k, "bb": cum_bb, "hits": cum_hits,
                "hr": cum_hr, "outs": cum_outs,
            })

    return {
        "pitcher_pp": pitcher_pp,
        "lineup_avg_pp": sum(batter_pp) / len(batter_pp) if batter_pp else default_pp,
        "rows": rows,
        "bf_min": bf_min, "bf_max": bf_max,
    }


def _lineup_rollup(rows: list[dict], pa_per_batter: int) -> dict:
    """Aggregate expected outcomes across the lineup over `pa_per_batter` PAs each."""
    n = len(rows)
    total_pa = n * pa_per_batter
    e_k = sum(r["k_pct"] for r in rows) * pa_per_batter
    e_bb = sum(r["bb_pct"] for r in rows) * pa_per_batter
    e_hr = sum(r["hr_pct"] for r in rows) * pa_per_batter
    e_hit = sum(r["hit_pct"] for r in rows) * pa_per_batter
    e_ob = sum(r["ob_pct"] for r in rows) * pa_per_batter

    # Probability of at least one HR somewhere in the lineup, assuming independence
    # of every PA (PAs * batters). For a batter with HR rate p over k PAs,
    # P(>=1 HR) = 1 - (1-p)^k. Combine across batters with product.
    p_no_hr = 1.0
    for r in rows:
        p = max(0.0, min(1.0, r["hr_pct"]))
        p_no_hr *= (1 - p) ** pa_per_batter
    p_at_least_one_hr = 1 - p_no_hr

    # Lineup-weighted xwOBA (simple average; each batter assumed equal PA weight).
    lineup_xwoba = sum(r["proj_xwoba"] for r in rows) / n if n else float("nan")

    return {
        "n": n,
        "total_pa": total_pa,
        "pa_per_batter": pa_per_batter,
        "lineup_xwoba": lineup_xwoba,
        "delta_pts": (lineup_xwoba - LG_XWOBA) * 1000 if n else 0.0,
        "e_k": e_k, "e_bb": e_bb, "e_hr": e_hr, "e_hit": e_hit, "e_ob": e_ob,
        "p_at_least_one_hr": p_at_least_one_hr,
    }


# ----- Lineup markdown -----

def to_lineup_markdown(pitcher_meta: dict, season: int,
                       summary_rows: list[dict], rollup: dict,
                       per_batter_pieces: list[dict],
                       projected: bool = False,
                       pitcher_range: dict | None = None) -> str:
    lines: list[str] = []
    pname = pitcher_meta["name"]
    lines.append(f"# Lineup vs {pname} — matchup report")
    lines.append("")
    if projected:
        lines.append("> **Projected lineup** — actual lineup not yet posted; "
                     "this is the team's most recent batting order vs a same-handed starter.")
        lines.append("")
    lines.append(
        f"_pitcher: {pname} ({pitcher_meta['p_throws'] or '?'}HP) · "
        f"{rollup['n']} batters × {rollup['pa_per_batter']} PA each = "
        f"{rollup['total_pa']} projected PAs · "
        f"window blends "
        + ", ".join(f"{season - off} (×{w:g})" for off, w in enumerate(SEASON_WEIGHTS))
        + "_"
    )
    lines.append("")

    # Lineup-level rollup card
    lines.append("## Lineup outlook")
    lines.append("")
    lines.append(
        f"- **Lineup avg projected xwOBA**: {rollup['lineup_xwoba']:.3f} "
        f"({rollup['delta_pts']:+.0f} pts vs league avg {LG_XWOBA:.3f})"
    )
    lines.append(f"- **Expected K** across lineup: {rollup['e_k']:.1f}")
    lines.append(f"- **Expected BB**: {rollup['e_bb']:.1f}")
    lines.append(f"- **Expected hits**: {rollup['e_hit']:.1f}")
    lines.append(f"- **Expected HR**: {rollup['e_hr']:.2f}")
    lines.append(f"- **Expected times on base**: {rollup['e_ob']:.1f}")
    lines.append(f"- **P(>=1 HR somewhere in lineup)**: {rollup['p_at_least_one_hr']*100:.1f}%")
    lines.append("")

    # Projected pitcher statline by batters faced
    if pitcher_range and pitcher_range.get("rows"):
        pr = pitcher_range
        lines.append(f"## Projected pitcher line by BF ({pr['bf_min']}–{pr['bf_max']})")
        lines.append("")
        lines.append(
            f"_Cycles through the order, blending {pname}'s {pr['pitcher_pp']:.2f} P/PA "
            f"with the lineup's {pr['lineup_avg_pp']:.2f} P/PA average (long PAs "
            f"winsorized at 10). Each row shows the cumulative line at that BF._"
        )
        lines.append("")
        lines.append("| BF | Pitches | K | BB | Hits | HR | Outs |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|")
        for r in pr["rows"]:
            lines.append(
                f"| {r['bf']} | {r['pitches']:.0f} | "
                f"{r['k']:.1f} | {r['bb']:.1f} | "
                f"{r['hits']:.1f} | {r['hr']:.2f} | {r['outs']:.1f} |"
            )
        lines.append("")

    # Lineup grid
    lines.append("## Lineup grid")
    lines.append("")
    lines.append(
        "| # | Batter | Hand | Proj xwOBA | Proj xBA | Proj xSLG | Δ (pts) | K% | BB% | HR% | Hit% | OB% | Best pitch (xwOBA) | Worst pitch (xwOBA) | Verdict |"
    )
    lines.append("|---:|---|:-:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|:-:|")
    for r in summary_rows:
        lines.append(
            f"| {r['spot']} | {r['name']} | {r['stand']}HB | "
            f"{r['proj_xwoba']:.3f} | {fmt3(r.get('proj_xba'))} | {fmt3(r.get('proj_xslg'))} | "
            f"{r['delta_pts']:+.0f} | "
            f"{r['k_pct']*100:.1f}% | {r['bb_pct']*100:.1f}% | "
            f"{r['hr_pct']*100:.1f}% | {r['hit_pct']*100:.1f}% | "
            f"{r['ob_pct']*100:.1f}% | {r['best_pitch']} | {r['worst_pitch']} | "
            f"{r['verdict_label']} |"
        )
    lines.append("")

    # Per-batter detail sections (full report content for each batter)
    lines.append("## Per-batter detail")
    lines.append("")
    for r, p in zip(summary_rows, per_batter_pieces):
        lines.append(f"### {r['spot']}. {r['name']} vs {pname}")
        lines.append("")
        body = _render_md_from_pieces(p, body_only=True)
        # Drop the per-batter `# title` line that body_only still emits, since
        # we already have a `### N. Name` header above it.
        body_lines = body.split("\n")
        if body_lines and body_lines[0].startswith("# "):
            body_lines = body_lines[1:]
            while body_lines and body_lines[0].strip() == "":
                body_lines = body_lines[1:]
        # Demote inner `## ` headings to `#### ` so they nest under the batter header.
        demoted = []
        for ln in body_lines:
            if ln.startswith("## "):
                demoted.append("#### " + ln[3:])
            else:
                demoted.append(ln)
        lines.extend(demoted)
        lines.append("")

    # Document-level footer notes (single copy)
    lines.append("## Notes & caveats")
    lines.append("")
    lines.append(
        "- Each batter section uses the same per-matchup methodology; see any single-matchup "
        "report or `README.md` for the full layer-by-layer breakdown."
    )
    lines.append(
        "- Lineup rollups assume each batter gets exactly "
        f"{rollup['pa_per_batter']} PAs vs this pitcher and that PAs are independent "
        "(no fatigue / TTO interaction across the rollup)."
    )
    lines.append(
        f"- Best/worst pitch columns show the top entry from the per-batter edge analysis; "
        f"the number in parens is the batter's xwOBA on that pitch type."
    )
    return "\n".join(lines)


# ----- Lineup HTML -----

def _summary_for_block(r: dict, pname: str) -> str:
    """Inline summary line for a per-batter <details><summary>."""
    delta_str = f"{r['delta_pts']:+.0f} pts"
    xba = r.get("proj_xba")
    xslg = r.get("proj_xslg")
    slash_bits = [f"xwOBA <b>{r['proj_xwoba']:.3f}</b>"]
    if xba is not None and not (isinstance(xba, float) and math.isnan(xba)):
        slash_bits.append(f"xBA <b>{xba:.3f}</b>")
    if xslg is not None and not (isinstance(xslg, float) and math.isnan(xslg)):
        slash_bits.append(f"xSLG <b>{xslg:.3f}</b>")
    slash_str = " / ".join(slash_bits)
    return (
        f'<span class="spot">{r["spot"]}.</span>'
        f'<span class="name">{_h(r["name"])}</span>'
        f'<span class="badge">{_h(r["stand"])}HB</span>'
        f'<span class="summary-stat">proj {slash_str} ({delta_str})</span>'
        f'<span class="summary-stat">K <b>{r["k_pct"]*100:.1f}%</b></span>'
        f'<span class="summary-stat">BB <b>{r["bb_pct"]*100:.1f}%</b></span>'
        f'<span class="summary-stat">HR <b>{r["hr_pct"]*100:.1f}%</b></span>'
        f'<span class="summary-stat">Hit <b>{r["hit_pct"]*100:.1f}%</b></span>'
        f'<span class="verdict-pill {r["verdict_css"]}">{_h(r["verdict_label"])}</span>'
    )


def to_lineup_html(pitcher_meta: dict, season: int,
                   summary_rows: list[dict], rollup: dict,
                   per_batter_pieces: list[dict],
                   projected: bool = False,
                   pitcher_range: dict | None = None) -> str:
    parts: list[str] = []
    pname = pitcher_meta["name"]
    title = f"Lineup vs {pname} - matchup report"

    parts.append("<!doctype html>")
    parts.append('<html lang="en">')
    parts.append("<head>")
    parts.append('<meta charset="utf-8">')
    parts.append(f"<title>{_h(title)}</title>")
    parts.append(f"<style>{_HTML_CSS}</style>")
    parts.append("</head>")
    parts.append("<body>")
    parts.append('<main class="container">')

    # ----- Header -----
    parts.append('<header class="page-head">')
    parts.append(
        f'<h1>Lineup vs {_h(pname)} '
        f'<span class="badge">{_h(pitcher_meta["p_throws"] or "?")}HP</span></h1>'
    )
    if projected:
        parts.append(
            '<div style="margin:8px 0;padding:8px 12px;background:#fff8dc;'
            'border-left:4px solid #d4a017;color:#5a4500;font-weight:600;">'
            'Projected lineup &mdash; actual lineup not yet posted; '
            'this is the team\'s most recent batting order vs a same-handed starter.'
            '</div>'
        )
    parts.append(
        f'<div class="meta">{rollup["n"]} batters &times; {rollup["pa_per_batter"]} PA each '
        f'= {rollup["total_pa"]} projected PAs &middot; '
        + "window blends "
        + ", ".join(
            f"{season - off} &times;{w:g}" for off, w in enumerate(SEASON_WEIGHTS)
        )
        + '</div>'
    )
    parts.append("</header>")

    # ----- Hero rollup -----
    delta_cls = (
        "bat-edge-strong" if rollup["delta_pts"] >= 50
        else "bat-edge-mild" if rollup["delta_pts"] >= 25
        else "pit-edge-strong" if rollup["delta_pts"] <= -50
        else "pit-edge-mild" if rollup["delta_pts"] <= -25
        else ""
    )
    parts.append('<section class="card">')
    parts.append("<h2>Lineup outlook</h2>")
    parts.append('<div class="lineup-hero">')
    parts.append(
        f'<div class="stat">Lineup avg xwOBA<b class="{delta_cls}">'
        f'{rollup["lineup_xwoba"]:.3f}</b></div>'
    )
    parts.append(
        f'<div class="stat">vs league<b class="{delta_cls}">'
        f'{rollup["delta_pts"]:+.0f} pts</b></div>'
    )
    parts.append(f'<div class="stat">Expected K<b>{rollup["e_k"]:.1f}</b></div>')
    parts.append(f'<div class="stat">Expected BB<b>{rollup["e_bb"]:.1f}</b></div>')
    parts.append(f'<div class="stat">Expected hits<b>{rollup["e_hit"]:.1f}</b></div>')
    parts.append(f'<div class="stat">Expected HR<b>{rollup["e_hr"]:.2f}</b></div>')
    parts.append(f'<div class="stat">Expected on-base<b>{rollup["e_ob"]:.1f}</b></div>')
    parts.append(
        f'<div class="stat">P(&ge;1 HR in lineup)<b>{rollup["p_at_least_one_hr"]*100:.1f}%</b></div>'
    )
    parts.append('</div>')
    parts.append("</section>")

    # ----- Projected pitcher line by BF -----
    if pitcher_range and pitcher_range.get("rows"):
        pr = pitcher_range
        pname = pitcher_meta["name"]
        parts.append('<section class="card">')
        parts.append(
            f"<h2>Projected pitcher line by BF "
            f"({pr['bf_min']}&ndash;{pr['bf_max']})</h2>"
        )
        parts.append(
            f'<p class="subtitle">Cycles through the order, blending {_h(pname)}\'s '
            f'<b>{pr["pitcher_pp"]:.2f}</b> P/PA with the lineup\'s '
            f'<b>{pr["lineup_avg_pp"]:.2f}</b> P/PA average (long PAs winsorized at 10). '
            f'Each row shows the cumulative line at that BF.</p>'
        )
        parts.append('<table class="pitcher-bf"><thead><tr>'
                     '<th>BF</th><th>Pitches</th><th>K</th><th>BB</th>'
                     '<th>Hits</th><th>HR</th><th>Outs</th>'
                     '</tr></thead><tbody>')
        for r in pr["rows"]:
            parts.append(
                "<tr>"
                f"<td><b>{r['bf']}</b></td>"
                f"<td>{r['pitches']:.0f}</td>"
                f"<td>{r['k']:.1f}</td>"
                f"<td>{r['bb']:.1f}</td>"
                f"<td>{r['hits']:.1f}</td>"
                f"<td>{r['hr']:.2f}</td>"
                f"<td>{r['outs']:.1f}</td>"
                "</tr>"
            )
        parts.append("</tbody></table>")
        parts.append("</section>")

    # ----- Lineup grid -----
    parts.append('<section class="card">')
    parts.append("<h2>Lineup grid</h2>")
    parts.append('<table class="lineup-grid"><thead><tr>'
                 '<th>#</th><th style="text-align:left">Batter</th><th>Hand</th>'
                 '<th>Proj xwOBA</th><th>Proj xBA</th><th>Proj xSLG</th>'
                 '<th>&Delta; (pts)</th>'
                 '<th>K%</th><th>BB%</th><th>HR%</th><th>Hit%</th><th>OB%</th>'
                 '<th style="text-align:left">Best pitch</th>'
                 '<th style="text-align:left">Worst pitch</th>'
                 '<th>Verdict</th>'
                 '</tr></thead><tbody>')
    for r in summary_rows:
        # Color cells: xwOBA + delta against league; K/BB/HR/Hit against league outcome rates.
        proj_cls = edge_class(r["proj_xwoba"], LG_XWOBA, 0.025, batter_favors_high=True)
        xba = r.get("proj_xba")
        xslg = r.get("proj_xslg")
        xba_cls = edge_class(xba, LG_XBA, 0.025, batter_favors_high=True)
        xslg_cls = edge_class(xslg, LG_XSLG, 0.040, batter_favors_high=True)
        delta_cls_row = (
            "bat-edge-strong" if r["delta_pts"] >= 50
            else "bat-edge-mild" if r["delta_pts"] >= 25
            else "pit-edge-strong" if r["delta_pts"] <= -50
            else "pit-edge-mild" if r["delta_pts"] <= -25
            else ""
        )
        k_cls = edge_class(r["k_pct"]*100, LG_K_PCT*100, 3.0, batter_favors_high=False)
        bb_cls = edge_class(r["bb_pct"]*100, LG_BB_PCT*100, 2.0, batter_favors_high=True)
        hr_cls = edge_class(r["hr_pct"]*100, LG_OUTCOMES["HR"]*100, 1.0, batter_favors_high=True)
        hit_cls = edge_class(
            r["hit_pct"]*100,
            sum(LG_OUTCOMES[k] for k in ("1B", "2B", "3B", "HR"))*100,
            2.0,
            batter_favors_high=True,
        )
        ob_cls = edge_class(
            r["ob_pct"]*100,
            sum(LG_OUTCOMES[k] for k in ("1B", "2B", "3B", "HR", "BB", "HBP"))*100,
            2.5,
            batter_favors_high=True,
        )
        parts.append(
            "<tr>"
            f'<td class="spot">{r["spot"]}</td>'
            f'<td class="name"><a href="#{_h(r["anchor"])}">{_h(r["name"])}</a></td>'
            f'<td class="handpill"><span class="pill">{_h(r["stand"])}HB</span></td>'
            f'{_td(f"{r['proj_xwoba']:.3f}", proj_cls)}'
            f'{_td(fmt3(xba), xba_cls)}'
            f'{_td(fmt3(xslg), xslg_cls)}'
            f'{_td(f"{r['delta_pts']:+.0f}", delta_cls_row)}'
            f'{_td(f"{r['k_pct']*100:.1f}%", k_cls)}'
            f'{_td(f"{r['bb_pct']*100:.1f}%", bb_cls)}'
            f'{_td(f"{r['hr_pct']*100:.1f}%", hr_cls)}'
            f'{_td(f"{r['hit_pct']*100:.1f}%", hit_cls)}'
            f'{_td(f"{r['ob_pct']*100:.1f}%", ob_cls)}'
            f'<td class="pitch-cell">{_h(r["best_pitch"])}</td>'
            f'<td class="pitch-cell">{_h(r["worst_pitch"])}</td>'
            f'<td class="verdict {r["verdict_css"]}">{_h(r["verdict_label"])}</td>'
            "</tr>"
        )
    parts.append("</tbody></table>")
    parts.append(
        '<p class="note">Click a batter\'s name (or the row below) to expand the full '
        'per-batter matchup report. Cells are colored vs the league baseline for that metric.</p>'
    )
    parts.append("</section>")

    # ----- Per-batter detail blocks -----
    parts.append('<section class="card">')
    parts.append("<h2>Per-batter detail</h2>")
    for r, p in zip(summary_rows, per_batter_pieces):
        parts.append(f'<details class="batter-block" id="{_h(r["anchor"])}">')
        parts.append(f'<summary>{_summary_for_block(r, pname)}</summary>')
        parts.append('<div class="batter-body">')
        parts.append(_render_html_from_pieces(p, body_only=True))
        parts.append("</div>")
        parts.append("</details>")
    parts.append("</section>")

    # ----- Footer notes (document-level, single copy) -----
    parts.append("<footer>")
    parts.append("<ul class='note-list'>")
    parts.append(
        "<li>Each batter block uses the same per-matchup methodology; see any single-matchup "
        "report or <code>README.md</code> for the full layer-by-layer breakdown.</li>"
    )
    parts.append(
        f"<li>Lineup rollups assume each batter gets exactly {rollup['pa_per_batter']} PAs "
        "vs this pitcher and that PAs are independent (no fatigue / TTO interaction across "
        "the rollup).</li>"
    )
    parts.append(
        "<li>Verdict pill is based on projected xwOBA vs league baseline: "
        "&lt;25 pts = Even, 25-50 = Slight, 50-100 = Edge, &gt;100 = Strong.</li>"
    )
    parts.append(
        "<li>Cell coloring in the grid: <span class='bat-edge-strong' "
        "style='padding:1px 6px;border-radius:4px'>strong batter edge</span>, "
        "<span class='bat-edge-mild' style='padding:1px 6px;border-radius:4px'>mild batter edge</span>, "
        "<span class='pit-edge-mild' style='padding:1px 6px;border-radius:4px'>mild pitcher edge</span>, "
        "<span class='pit-edge-strong' style='padding:1px 6px;border-radius:4px'>strong pitcher edge</span> "
        "&mdash; relative to league baseline.</li>"
    )
    parts.append("</ul>")
    parts.append("</footer>")

    parts.append("</main>")
    parts.append(sortable_html())
    parts.append("</body></html>")
    return "\n".join(parts)


def analyze_lineup(batter_ids: list[int], pitcher_id: int,
                   season: int = DEFAULT_SEASON,
                   pa_per_batter: int = DEFAULT_PA_PER_BATTER,
                   projected: bool = False) -> tuple[str, str, str, dict]:
    """Run analyze_matchup for each batter against the same pitcher and assemble
    a lineup-level report.

    Returns (output stem, markdown, html, roundup_data) where roundup_data is a
    dict the day-level roundup script can consume without re-running the
    expensive matchup compute. It contains the pitcher meta, the summary rows
    (one per batter), and the pre-rendered per-batter HTML body for each batter
    so they can be lifted directly into top/bottom-50 reports.
    """
    if not batter_ids:
        raise SystemExit("No batters in lineup.")

    pieces_per_batter: list[dict] = []
    summary_rows: list[dict] = []
    pitcher_meta: dict | None = None
    skipped: list[tuple[int, str]] = []
    spot_counter = 0
    for raw_spot, bid in enumerate(batter_ids, start=1):
        print(f"  [{raw_spot}/{len(batter_ids)}] computing batter id {bid} ...")
        try:
            p = compute_matchup_pieces(bid, pitcher_id, season)
        except SystemExit as exc:
            print(f"    skipping batter id {bid}: {exc}")
            skipped.append((bid, str(exc)))
            continue
        spot_counter += 1
        pieces_per_batter.append(p)
        summary_rows.append(_lineup_summary_row(p, spot_counter))
        if pitcher_meta is None:
            pitcher_meta = p["pitcher_meta"]

    if not summary_rows:
        raise SystemExit("No batters in lineup produced usable data; aborting.")
    if skipped:
        print(f"  warning: skipped {len(skipped)} batter(s) with no usable data: {skipped}")

    rollup = _lineup_rollup(summary_rows, pa_per_batter)
    pitcher_range = project_pitcher_range(pieces_per_batter, summary_rows)
    md = to_lineup_markdown(pitcher_meta, season, summary_rows, rollup, pieces_per_batter,
                            projected=projected, pitcher_range=pitcher_range)
    html_doc = to_lineup_html(pitcher_meta, season, summary_rows, rollup, pieces_per_batter,
                              projected=projected, pitcher_range=pitcher_range)

    pit_slug = pitcher_meta["last"].lower().replace(" ", "_")
    out_stem = f"lineup_vs_{pit_slug}_{season}"

    roundup_data = {
        "pitcher_meta": {
            "name": pitcher_meta.get("name"),
            "first": pitcher_meta.get("first"),
            "last": pitcher_meta.get("last"),
            "p_throws": pitcher_meta.get("p_throws"),
            "id": pitcher_meta.get("id"),
        },
        "season": season,
        "projected": projected,
        "pa_per_batter": pa_per_batter,
        "summary_rows": summary_rows,
        "per_batter_html": [
            _render_html_from_pieces(p, body_only=True) for p in pieces_per_batter
        ],
    }
    return out_stem, md, html_doc, roundup_data


# ---------- batch + CLI ---------------------------------------------------

def _resolve_inputs(batter_arg: str | None, pitcher_arg: str | None,
                     batter_id_arg: int | None, pitcher_id_arg: int | None) -> tuple[int, int]:
    if batter_id_arg is not None:
        bid = int(batter_id_arg)
    elif batter_arg:
        bid = _resolve_player(batter_arg, fallback_id=DEFAULT_BATTER_ID
                                if batter_arg.lower() == DEFAULT_BATTER_NAME.lower() else None)
    else:
        bid = DEFAULT_BATTER_ID

    if pitcher_id_arg is not None:
        pid = int(pitcher_id_arg)
    elif pitcher_arg:
        pid = _resolve_player(pitcher_arg, fallback_id=DEFAULT_PITCHER_ID
                                if pitcher_arg.lower() == DEFAULT_PITCHER_NAME.lower() else None)
    else:
        pid = DEFAULT_PITCHER_ID

    return bid, pid


# Mutable report-date override. Defaults to today; main() can rewrite this
# from a --date flag or by parsing the batch CSV filename.
_REPORT_DATE: date = date.today()


def _set_report_date(d: date) -> None:
    global _REPORT_DATE
    _REPORT_DATE = d


def _report_dir() -> Path:
    """Return (and create) the report output directory: reports/<_REPORT_DATE>/.

    Defaults to today, but is overridable when generating reports for a past
    or future slate (e.g. `--batch matchups_actual_2026-05-14.csv` should
    land under `reports/2026-05-14/`, not under today's folder).
    """
    d = ROOT / "reports" / _REPORT_DATE.isoformat()
    d.mkdir(parents=True, exist_ok=True)
    return d


_CSV_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def _derive_report_date(batch_path: Path | None, override: str | None) -> date:
    """Pick the right slate date for the report output directory.

    Priority: explicit --date override -> date parsed from batch CSV
    filename (e.g. matchups_2026-05-14.csv) -> today.
    """
    if override:
        return date.fromisoformat(override)
    if batch_path is not None:
        m = _CSV_DATE_RE.search(batch_path.name)
        if m:
            try:
                return date.fromisoformat(m.group(1))
            except ValueError:
                pass
    return date.today()


def _roundup_data_dir() -> Path:
    """Return (and create) today's sidecar data directory: reports/<date>/_data/.

    Each lineup batch run drops one JSON per (matchup, pitcher) into this folder
    so the day-level roundup script (`roundup.py`) can build the top-50 /
    bottom-50 reports without re-running the matchup compute.
    """
    d = _report_dir() / "_data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_roundup_sidecar(out_stem: str, matchup_key: str, hitter_team: str,
                           pitcher_name: str, data: dict) -> None:
    """Persist one lineup's summary rows + per-batter HTML body to JSON."""
    payload = {
        "out_stem": out_stem,
        "matchup_key": matchup_key,
        "hitter_team": hitter_team,
        "pitcher_name": pitcher_name,
        "pitcher_meta": data.get("pitcher_meta", {}),
        "season": data.get("season"),
        "projected": bool(data.get("projected")),
        "pa_per_batter": data.get("pa_per_batter"),
        "summary_rows": data.get("summary_rows", []),
        "per_batter_html": data.get("per_batter_html", []),
    }
    path = _roundup_data_dir() / f"{out_stem}.json"
    path.write_text(json.dumps(payload, default=str), encoding="utf-8")


def _slugify(text: str) -> str:
    out = []
    for ch in text.lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in (" ", "-", "_"):
            out.append("_")
    slug = "".join(out)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def _resolve_workers(workers: int | None) -> int:
    """Pick a sensible thread-pool size for parallel report generation.

    Defaults to min(8, cpu_count). Capped because each worker holds a chunk
    of pandas/HTML state in memory; going wider rarely speeds things up
    once the network preload is done.
    """
    if workers is not None:
        return max(1, int(workers))
    return min(8, os.cpu_count() or 1)


def run_batch(csv_path: Path, season: int, workers: int | None = None) -> None:
    legacy_rows: list[tuple[str, str]] = []
    # Lineup format groups:
    #   (matchup_key, pitcher_name) -> list of (position, hitter_name, status, hitter_team, hitter_id, pitcher_id)
    lineup_groups: dict[tuple[str, str], list[tuple[int, str, str, str, int | None, int | None]]] = {}

    # Older CSVs (pre-fetch_lineups encoding fix) were written using the
    # platform default (cp1252 on Windows), so non-ASCII names like "García"
    # would explode under strict utf-8. Try utf-8 with a BOM-tolerant codec
    # first, fall back to cp1252.
    try:
        f = csv_path.open("r", encoding="utf-8-sig", newline="")
        f.read(1)
        f.seek(0)
    except UnicodeDecodeError:
        f.close()
        f = csv_path.open("r", encoding="cp1252", newline="")
    with f:
        reader = csv.reader(f)
        for row in reader:
            row = [c.strip() for c in row if c.strip()]
            if len(row) < 2:
                continue

            if len(row) >= 4 and "@" in row[0]:
                # Lineup format: away@home,hitter_name,pitcher_name,lineup_position[,status[,hitter_team[,hitter_id[,pitcher_id]]]]
                matchup_key, hitter_name, pitcher_name = row[0], row[1], row[2]
                try:
                    pos = int(row[3])
                except ValueError:
                    pos = len(lineup_groups.get((matchup_key, pitcher_name), [])) + 1
                status = row[4].lower() if len(row) >= 5 else "confirmed"
                hitter_team = row[5].strip().upper() if len(row) >= 6 and row[5].strip() else ""
                hitter_id: int | None = int(row[6]) if len(row) >= 7 and row[6].strip().isdigit() else None
                pitcher_id: int | None = int(row[7]) if len(row) >= 8 and row[7].strip().isdigit() else None
                lineup_groups.setdefault((matchup_key, pitcher_name), []).append(
                    (pos, hitter_name, status, hitter_team, hitter_id, pitcher_id)
                )
            else:
                # Legacy format: batter,pitcher (any extra columns ignored)
                legacy_rows.append((row[0], row[1]))

    # ----- lineup batch mode --------------------------------------------------
    if lineup_groups:
        print(f"Batch (lineup mode): {len(lineup_groups)} team lineups from {csv_path.name}")

        # Pre-resolve unique players once.
        all_hitters: set[str] = set()
        all_pitchers: set[str] = set()
        # Track known MLBAM IDs from CSV (col 7 = hitter, col 8 = pitcher)
        csv_hitter_ids: dict[str, int] = {}   # name -> mlbam_id
        csv_pitcher_ids: dict[str, int] = {}  # name -> mlbam_id
        for (_mk, pname), batters in lineup_groups.items():
            all_pitchers.add(pname)
            for _pos, h, _status, _team, h_id, p_id in batters:
                all_hitters.add(h)
                if h_id is not None:
                    csv_hitter_ids[h] = h_id
                if p_id is not None:
                    csv_pitcher_ids[pname] = p_id

        hitter_id_map = {h: csv_hitter_ids[h] if h in csv_hitter_ids
                         else _resolve_player(h) for h in all_hitters}
        pitcher_id_map = {p: csv_pitcher_ids[p] if p in csv_pitcher_ids
                          else _resolve_player(p) for p in all_pitchers}

        unique_b = set(hitter_id_map.values())
        unique_p = set(pitcher_id_map.values())
        print(f"  unique batters: {len(unique_b)}, unique pitchers: {len(unique_p)}")
        preload_player_data(unique_b, unique_p, season)

        n_workers = _resolve_workers(workers)
        print(f"  generating reports across {n_workers} worker thread(s)")

        def _process_lineup(item) -> None:
            (matchup_key, pitcher_name), batters = item
            
            # Skip if pitcher is TBD (to be determined)
            if "TBD" in pitcher_name.upper():
                print(f"\n--- {matchup_key} vs {pitcher_name} ---")
                print(f"  skipped {matchup_key} vs {pitcher_name}: pitcher TBD")
                return
            
            batters_sorted = sorted(batters, key=lambda x: x[0])
            names_sorted = [name for _pos, name, _s, _t, _hid, _pid in batters_sorted]
            teams_sorted = [team for _pos, _n, _s, team, _hid, _pid in batters_sorted]
            is_projected = any(s == "projected" for _p, _n, s, _t, _hid, _pid in batters)
            # The hitters' team is the same for the whole group; pick the first
            # non-empty value as the canonical team code.
            hitter_team = next((t for t in teams_sorted if t), "")
            pid = pitcher_id_map[pitcher_name]
            batter_ids = [hitter_id_map[name] for name in names_sorted]

            tag = f"{matchup_key} vs {pitcher_name}" + (" (projected)" if is_projected else "")
            print(f"\n--- {tag} ---")
            try:
                _, md, html_doc, roundup_data = analyze_lineup(
                    batter_ids, pid, season, projected=is_projected,
                )
            except SystemExit as exc:
                print(f"  skipped {tag}: {exc}")
                return

            match_slug = matchup_key.replace("@", "_at_").lower()
            pit_slug = _slugify(pitcher_name)
            out_stem = f"{match_slug}_vs_{pit_slug}_{season}"

            html_path = _report_dir() / f"{out_stem}.html"
            html_path.write_text(html_doc, encoding="utf-8")
            print(f"  wrote {html_path.relative_to(ROOT)}")

            _write_roundup_sidecar(out_stem, matchup_key, hitter_team,
                                   pitcher_name, roundup_data)

        items = list(lineup_groups.items())
        if n_workers <= 1 or len(items) <= 1:
            for item in items:
                _process_lineup(item)
        else:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=n_workers, thread_name_prefix="report",
            ) as ex:
                futures = {ex.submit(_process_lineup, it): it[0] for it in items}
                for fut in concurrent.futures.as_completed(futures):
                    key = futures[fut]
                    try:
                        fut.result()
                    except Exception as exc:  # noqa: BLE001
                        print(f"  worker for {key} crashed: "
                              f"{exc.__class__.__name__}: {exc}", file=sys.stderr)
        return

    # ----- legacy per-row mode ------------------------------------------------
    rows = legacy_rows
    print(f"Batch: {len(rows)} matchups from {csv_path.name}")
    unique_b: set[int] = set()
    unique_p: set[int] = set()
    resolved: list[tuple[int, int]] = []
    for b_arg, p_arg in rows:
        bid, pid = _resolve_inputs(b_arg, p_arg, None, None)
        unique_b.add(bid)
        unique_p.add(pid)
        resolved.append((bid, pid))

    print(f"  unique batters: {len(unique_b)}, unique pitchers: {len(unique_p)}")
    preload_player_data(unique_b, unique_p, season)

    n_workers = _resolve_workers(workers)
    print(f"  generating reports across {n_workers} worker thread(s)")

    def _process_row(item) -> None:
        (b_arg, p_arg), (bid, pid) = item
        
        # Skip if pitcher is TBD (to be determined)
        if "TBD" in p_arg.upper():
            print(f"\n--- {b_arg} vs {p_arg} ---")
            print(f"  skipped {b_arg} vs {p_arg}: pitcher TBD")
            return
        
        print(f"\n--- {b_arg} vs {p_arg} ---")
        try:
            out_stem, md, html_doc = analyze_matchup(bid, pid, season)
        except SystemExit as exc:
            print(f"  skipped {b_arg} vs {p_arg}: {exc}")
            return
        html_path = _report_dir() / f"{out_stem}.html"
        html_path.write_text(html_doc, encoding="utf-8")
        print(f"  wrote {html_path.relative_to(ROOT)}")

    pairs = list(zip(rows, resolved))
    if n_workers <= 1 or len(pairs) <= 1:
        for item in pairs:
            _process_row(item)
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=n_workers, thread_name_prefix="report",
        ) as ex:
            futures = {ex.submit(_process_row, it): it[0] for it in pairs}
            for fut in concurrent.futures.as_completed(futures):
                key = futures[fut]
                try:
                    fut.result()
                except Exception as exc:  # noqa: BLE001
                    print(f"  worker for {key} crashed: "
                          f"{exc.__class__.__name__}: {exc}", file=sys.stderr)


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    )


def _normalize_data_layout() -> None:
    """Normalize cached parquet files into data/<start_year>/ folders."""
    data_dir = ROOT / "data"
    if not data_dir.exists():
        print("[normalize-data-layout] data/ directory does not exist; nothing to do")
        return

    moved = 0
    pattern = re.compile(r"statcast(?:_pitcher)?_\d+_(\d{4})-\d{2}-\d{2}_\d{4}-\d{2}-\d{2}\.parquet$")
    for path in sorted(data_dir.rglob("*.parquet")):
        if not path.is_file():
            continue
        match = pattern.search(path.name)
        if not match:
            continue

        target_year = match.group(1)
        target_dir = data_dir / target_year
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / path.name

        if path.parent == target_dir:
            continue
        if target_path.exists():
            print(f"[normalize-data-layout] skipping {path.relative_to(ROOT)}; target already exists: {target_path.relative_to(ROOT)}")
            continue

        print(f"[normalize-data-layout] moving {path.relative_to(ROOT)} -> {target_path.relative_to(ROOT)}")
        shutil.move(str(path), str(target_path))
        moved += 1

    if moved == 0:
        print("[normalize-data-layout] no misplaced parquet files found")
    else:
        print(f"[normalize-data-layout] moved {moved} file{'s' if moved != 1 else ''}")


def commit_prior_season_cache(current_season: int, push: bool = True) -> None:
    """Stage, commit, and (optionally) push any new prior-season parquets.

    Skips silently if:
      - we're not in a git repo,
      - data/ has no prior-season subfolders,
      - there are no new/modified prior-season parquets to add.
    """
    data_dir = ROOT / "data"
    if not data_dir.exists():
        return

    # Year subfolders strictly less than the current season.
    prior_dirs: list[Path] = []
    for sub in sorted(data_dir.iterdir()):
        if not sub.is_dir():
            continue
        try:
            year = int(sub.name)
        except ValueError:
            continue
        if year < current_season:
            prior_dirs.append(sub)

    if not prior_dirs:
        print("[commit-cache] no prior-season folders under data/, nothing to commit")
        return

    # Confirm we're in a git repo.
    rev = _git("rev-parse", "--git-dir")
    if rev.returncode != 0:
        print("[commit-cache] not a git repo, skipping")
        return

    rel_paths = [str(p.relative_to(ROOT)).replace("\\", "/") for p in prior_dirs]

    # Check what's actually new/modified under the prior-season folders. We
    # use --intent-to-add-free `git status --porcelain -- <paths>` so we don't
    # touch anything that isn't already a candidate to commit.
    status = _git("status", "--porcelain", "--", *rel_paths)
    if status.returncode != 0:
        print(f"[commit-cache] git status failed: {status.stderr.strip()}")
        return
    if not status.stdout.strip():
        print("[commit-cache] no new prior-season parquets to commit")
        return

    add = _git("add", "--", *rel_paths)
    if add.returncode != 0:
        print(f"[commit-cache] git add failed: {add.stderr.strip()}")
        return

    # Check that we actually staged something under those paths.
    staged = _git("diff", "--cached", "--name-only", "--", *rel_paths)
    if not staged.stdout.strip():
        print("[commit-cache] nothing staged after git add; skipping")
        return

    n_files = len(staged.stdout.strip().splitlines())
    today = date.today().isoformat()
    msg = f"cache: add prior-season parquets ({n_files} files) from {today} run"
    commit = _git("commit", "-m", msg, "--", *rel_paths)
    if commit.returncode != 0:
        print(f"[commit-cache] git commit failed: {commit.stderr.strip()}")
        return
    print(f"[commit-cache] committed {n_files} prior-season parquet(s)")

    if push:
        push_res = _git("push")
        if push_res.returncode != 0:
            print(
                f"[commit-cache] git push failed (commit is local): "
                f"{push_res.stderr.strip()}"
            )
        else:
            print("[commit-cache] pushed to origin")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    log_path = setup_logging("matchup")
    print(f"[matchup] logging to {log_path}")

    ap = argparse.ArgumentParser(description="batter-vs-pitcher matchup analysis")
    ap.add_argument("--batter", type=str, default=None,
                    help="batter name 'First Last' (default: Yordan Alvarez)")
    ap.add_argument("--pitcher", type=str, default=None,
                    help="pitcher name 'First Last' (default: Paul Skenes)")
    ap.add_argument("--batter-id", type=int, default=None, help="batter MLBAM id")
    ap.add_argument("--pitcher-id", type=int, default=None, help="pitcher MLBAM id")
    ap.add_argument("--season", type=int, default=DEFAULT_SEASON,
                    help=f"current season (default {DEFAULT_SEASON})")
    ap.add_argument("--batch", type=str, default=None,
                    help="path to a CSV file (batter,pitcher OR away@home,hitter,pitcher,position)")
    ap.add_argument("--fix-data-layout", action="store_true",
                    help="normalize cached parquet files into data/<year>/ folders")
    ap.add_argument("--lineup", type=str, default=None,
                    help="comma-separated batter names (or MLBAM ids) in lineup order, "
                         "vs the --pitcher flag. e.g. \"Yordan Alvarez,Aaron Judge,...\"")
    ap.add_argument("--lineup-csv", type=str, default=None,
                    help="path to a text file with one batter name (or MLBAM id) per line, "
                         "in lineup order, vs the --pitcher flag")
    ap.add_argument("--pa-per-batter", type=int, default=DEFAULT_PA_PER_BATTER,
                    help=f"projected PAs per batter for lineup rollups "
                         f"(default {DEFAULT_PA_PER_BATTER})")
    ap.add_argument("--commit-cache", action="store_true",
                    help="after the run completes, git add+commit+push any new "
                         "prior-season parquets under data/<year>/ (skips if "
                         "nothing new). Useful from codespaces/CI to keep the "
                         "shared cache warm.")
    ap.add_argument("--no-push", action="store_true",
                    help="with --commit-cache, commit locally but do not push")
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel workers for report generation in --batch mode "
                         "(default min(8, cpu_count)). Set 1 to disable threading.")
    ap.add_argument("--date", type=str, default=None,
                    help="report output date (YYYY-MM-DD). Defaults to the "
                         "date parsed from the --batch CSV filename "
                         "(matchups_YYYY-MM-DD.csv) or today.")
    args = ap.parse_args()

    if args.fix_data_layout:
        _normalize_data_layout()
        return

    batch_path = Path(args.batch) if args.batch else None
    _set_report_date(_derive_report_date(batch_path, args.date))
    if _REPORT_DATE != date.today():
        print(f"[matchup] output dir: reports/{_REPORT_DATE.isoformat()}/")

    if args.batch:
        run_batch(batch_path, args.season, workers=args.workers)
        if args.commit_cache:
            commit_prior_season_cache(args.season, push=not args.no_push)
        return

    if args.lineup or args.lineup_csv:
        if args.lineup_csv:
            lineup_path = Path(args.lineup_csv)
            raw = [ln.strip() for ln in lineup_path.read_text(encoding="utf-8").splitlines()
                   if ln.strip() and not ln.strip().startswith("#")]
        else:
            raw = [s.strip() for s in args.lineup.split(",") if s.strip()]
        if not raw:
            raise SystemExit("Lineup is empty.")
        if not (args.pitcher or args.pitcher_id):
            raise SystemExit("--lineup requires --pitcher (or --pitcher-id).")

        # Resolve all batter inputs (each may be a name or a numeric id).
        batter_ids: list[int] = []
        for entry in raw:
            if entry.isdigit():
                batter_ids.append(int(entry))
            else:
                batter_ids.append(_resolve_player(entry))
        if args.pitcher_id is not None:
            pid = int(args.pitcher_id)
        else:
            pid = _resolve_player(args.pitcher)

        print(f"Lineup ({len(batter_ids)}) vs pitcher id {pid}")
        out_stem, md, html_doc, _data = analyze_lineup(
            batter_ids, pid, args.season, args.pa_per_batter,
        )
        html_path = _report_dir() / f"{out_stem}.html"
        html_path.write_text(html_doc, encoding="utf-8")
        print()
        print(f"Lineup report written: {html_path.relative_to(ROOT)}")
        if args.commit_cache:
            commit_prior_season_cache(args.season, push=not args.no_push)
        return

    bid, pid = _resolve_inputs(args.batter, args.pitcher, args.batter_id, args.pitcher_id)
    out_stem, md, html_doc = analyze_matchup(bid, pid, args.season)
    html_path = _report_dir() / f"{out_stem}.html"
    html_path.write_text(html_doc, encoding="utf-8")

    print()
    print(md)
    print()
    print(f"Report written: {html_path.relative_to(ROOT)}")
    if args.commit_cache:
        commit_prior_season_cache(args.season, push=not args.no_push)


if __name__ == "__main__":
    main()
