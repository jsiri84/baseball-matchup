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
    """Map weight -> row count for a blended frame. Missing weights -> 0."""
    out: dict[float, int] = {float(w): 0 for w in SEASON_WEIGHTS}
    if df.empty or "weight" not in df.columns:
        return out
    grouped = df.groupby("weight").size()
    for w, n in grouped.items():
        out[float(w)] = int(n)
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
    """
    pa_rows = group[group["events"].notna()]
    if pa_rows.empty:
        return 0.0
    n_pa = float(pa_rows["weight"].sum())
    if not n_pa:
        return 0.0

    bb_w = float(pa_rows.loc[pa_rows["events"] == "walk", "weight"].sum())
    hbp_w = float(pa_rows.loc[pa_rows["events"] == "hit_by_pitch", "weight"].sum())

    bbe_x = group[(group["type"] == "X") & group["estimated_woba_using_speedangle"].notna()]
    contact = float((bbe_x["estimated_woba_using_speedangle"] * bbe_x["weight"]).sum())

    return (contact + WOBA_BB * bb_w + WOBA_HBP * hbp_w) / n_pa


def w_xba_xslg(group: pd.DataFrame) -> tuple[float, float]:
    """Return (xBA, xSLG) computed Savant-style: per-AB, K's count as 0."""
    pa_rows = group[group["events"].notna()]
    if pa_rows.empty:
        return 0.0, 0.0

    so_w = float(pa_rows.loc[pa_rows["events"].isin(K_EVENTS), "weight"].sum())
    bbe_x = group[(group["type"] == "X") & group["estimated_ba_using_speedangle"].notna()]
    bbe_w = float(bbe_x["weight"].sum())
    denom = bbe_w + so_w
    if not denom:
        return 0.0, 0.0
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

def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Add count_state, in_zone, tto_bucket columns. Idempotent on the cached frame."""
    if df.empty:
        return df
    df = df.copy()
    df["count_state"] = df["balls"].astype("Int64").astype(str) + "-" + df["strikes"].astype("Int64").astype(str)
    df["in_zone"] = df["zone"].between(1, 9)
    df["tto_bucket"] = df["n_thruorder_pitcher"].clip(upper=3).fillna(1).astype(int)
    return df


# ---------- per-pitch-type aggregation ------------------------------------

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

    return {
        "n_pa_w": n_pa_w, "n_pitches_w": float(df["weight"].sum()),
        "K_pct": k, "BB_pct": bb, "HBP_pct": hbp,
        "Whiff_pct": whiff, "Chase_pct": chase,
        "HardHit_pct": hh, "Barrel_pct": barrel,
        "GB_pct": gb, "Air_pct": air,
        "xwOBA": w_xwoba(df), "xBA": w_xba_xslg(df)[0], "xSLG": w_xba_xslg(df)[1],
    }


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

    # P(c) from batter: weighted share of pitches per count-state.
    bat_count_w = bat.groupby("count_state")["weight"].sum()
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


def project(batter_pt: pd.DataFrame, pitcher_pt: pd.DataFrame,
            batter_overall: dict, pitcher_overall: dict,
            marginal: pd.Series) -> dict:
    """Headline projection: log5 for rates, additive for xwOBA/xBA/xSLG."""
    # Per-PA rates: log5 on overall numbers, not per-pitch.
    proj_k = log5(batter_overall["K_pct"], pitcher_overall["K_pct"], LG_K_PCT)
    proj_bb = log5(batter_overall["BB_pct"], pitcher_overall["BB_pct"], LG_BB_PCT)
    proj_hbp = log5(batter_overall["HBP_pct"], pitcher_overall["HBP_pct"], LG_HBP_PCT)

    # Per-pitch-type contributions, weighted by marginal pitcher usage.
    bat_idx = batter_pt.set_index("pitch_name") if not batter_pt.empty else pd.DataFrame()
    pit_idx = pitcher_pt.set_index("pitch_name") if not pitcher_pt.empty else pd.DataFrame()

    proj_xwoba = 0.0
    proj_xba = 0.0
    proj_xslg = 0.0
    proj_whiff = 0.0
    proj_hh = 0.0
    rows = []
    for pitch_name, p in marginal.items():
        b_x = float(bat_idx.loc[pitch_name, "xwOBA"]) if pitch_name in bat_idx.index else batter_overall["xwOBA"]
        p_x = float(pit_idx.loc[pitch_name, "xwOBA"]) if pitch_name in pit_idx.index else pitcher_overall["xwOBA"]
        b_a = float(bat_idx.loc[pitch_name, "xBA"]) if pitch_name in bat_idx.index else batter_overall["xBA"]
        p_a = float(pit_idx.loc[pitch_name, "xBA"]) if pitch_name in pit_idx.index else pitcher_overall["xBA"]
        b_s = float(bat_idx.loc[pitch_name, "xSLG"]) if pitch_name in bat_idx.index else batter_overall["xSLG"]
        p_s = float(pit_idx.loc[pitch_name, "xSLG"]) if pitch_name in pit_idx.index else pitcher_overall["xSLG"]
        b_w = float(bat_idx.loc[pitch_name, "Whiff_pct"]) if pitch_name in bat_idx.index else batter_overall["Whiff_pct"]
        p_w = float(pit_idx.loc[pitch_name, "Whiff_pct"]) if pitch_name in pit_idx.index else pitcher_overall["Whiff_pct"]
        b_h = float(bat_idx.loc[pitch_name, "HardHit_pct"]) if pitch_name in bat_idx.index else batter_overall["HardHit_pct"]
        p_h = float(pit_idx.loc[pitch_name, "HardHit_pct"]) if pitch_name in pit_idx.index else pitcher_overall["HardHit_pct"]

        m_xwoba = additive(b_x, p_x, LG_XWOBA)
        m_xba = additive(b_a, p_a, LG_XBA)
        m_xslg = additive(b_s, p_s, LG_XSLG)
        m_whiff = log5(b_w, p_w, LG_WHIFF)
        m_hh = log5(b_h, p_h, LG_HARD_HIT)

        proj_xwoba += p * m_xwoba
        proj_xba += p * m_xba
        proj_xslg += p * m_xslg
        proj_whiff += p * m_whiff
        proj_hh += p * m_hh

        rows.append({
            "Pitch": pitch_name,
            "Marginal Usage %": p * 100,
            "Batter xwOBA": b_x,
            "Pitcher xwOBA allowed": p_x,
            "Projected xwOBA": m_xwoba,
            "Projected Whiff %": m_whiff * 100,
        })

    pitch_table = pd.DataFrame(rows).sort_values("Marginal Usage %", ascending=False).reset_index(drop=True)

    return {
        "K_pct": proj_k, "BB_pct": proj_bb, "HBP_pct": proj_hbp,
        "xwOBA": proj_xwoba, "xBA": proj_xba, "xSLG": proj_xslg,
        "Whiff_pct": proj_whiff, "HardHit_pct": proj_hh,
        "pitch_table": pitch_table,
        "marginal": marginal,
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

        rows.append({
            "Pitch": ar["pitch_name"],
            "Shape (eff velo / IVB / HB-in)": f"{ar['velo']:.1f} / {ar['ivb']:+.1f} / {ar['hb_in']:+.1f}",
            "n comps (eff)": eff_n,
            "Whiff %": whiff_pct * 100,
            "xwOBA": w_xwoba(comps),
            "Hard Hit %": hh * 100,
            "Confidence": confidence,
        })

    return pd.DataFrame(rows)


# ---------- Layer 3: zone overlay -----------------------------------------

def zone_overlay(pit_vs_bat: pd.DataFrame, bat_vs_pit: pd.DataFrame,
                 arsenal: pd.DataFrame) -> pd.DataFrame:
    """Per arsenal pitch: weighted intersection of attack-share x batter xwOBA per zone."""
    if pit_vs_bat.empty or arsenal.empty:
        return pd.DataFrame()

    bat_zone_xwoba = {}
    for z in ALL_ZONES:
        cell = bat_vs_pit[bat_vs_pit["zone"] == z]
        if cell.empty:
            bat_zone_xwoba[z] = float("nan")
            continue
        bat_zone_xwoba[z] = w_xwoba(cell) if cell["events"].notna().any() else float("nan")

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

        # Intersection metric: sum_z (attack_z * batter_xwoba_z), skipping NaN cells.
        intersect_num = 0.0
        intersect_w = 0.0
        for z in ALL_ZONES:
            xw = bat_zone_xwoba[z]
            if math.isnan(xw):
                continue
            intersect_num += attack_share[z] * xw
            intersect_w += attack_share[z]
        intersect = intersect_num / intersect_w if intersect_w else float("nan")

        in_zone_share = float(attack_share.loc[IN_ZONE].sum())
        # Top 3 zones by attack share for this pitch
        top_zones = attack_share.sort_values(ascending=False).head(3)
        top_zone_str = ", ".join(f"z{int(z)}={s*100:.0f}%" for z, s in top_zones.items() if s > 0)

        rows.append({
            "Pitch": ar["pitch_name"],
            "In-zone %": in_zone_share * 100,
            "Top zones": top_zone_str,
            "Intersection xwOBA": intersect,
        })

    return pd.DataFrame(rows)


# ---------- Layer 4: TTO curve --------------------------------------------

def tto_curve(pit_vs_bat: pd.DataFrame) -> pd.DataFrame:
    """xwOBA / Whiff% / HardHit% allowed by times through the order."""
    if pit_vs_bat.empty:
        return pd.DataFrame()

    rows = []
    for tto, group in pit_vs_bat.groupby("tto_bucket"):
        n_pa_w = float(group[group["events"].notna()]["weight"].sum())
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
        rows.append({
            "TTO": int(tto),
            "PA (eff)": n_pa_w,
            "xwOBA allowed": w_xwoba(group),
            "Whiff %": whiff * 100,
            "Hard Hit %": hh * 100,
        })

    return pd.DataFrame(rows).sort_values("TTO").reset_index(drop=True)


def tto_projections(base_proj: dict, tto: pd.DataFrame, pitcher_overall: dict) -> pd.DataFrame:
    """Blend the headline projection with per-TTO pitcher xwOBA delta."""
    if tto.empty:
        return pd.DataFrame()
    base_pit = pitcher_overall["xwOBA"]
    rows = []
    for _, r in tto.iterrows():
        delta = r["xwOBA allowed"] - base_pit
        rows.append({
            "TTO": int(r["TTO"]),
            "Projected xwOBA": float(np.clip(base_proj["xwOBA"] + delta, 0.0, 1.0)),
            "Projected Whiff %": base_proj["Whiff_pct"] * 100,
            "Projected K %": base_proj["K_pct"] * 100,
            "Projected Hard Hit %": base_proj["HardHit_pct"] * 100,
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
        ts_pa = ts_bat[ts_bat["events"].notna()]
        n_pa_w = float(ts_pa["weight"].sum())
        k_w = float((ts_pa["events"].isin(K_EVENTS).astype(float) * ts_pa["weight"]).sum())
        # Pitcher's two-strike pitch mix
        ts_mix = (ts_pit.dropna(subset=["pitch_name"])
                        .groupby("pitch_name")["weight"].sum())
        ts_mix = (ts_mix / ts_mix.sum() * 100).round(1).sort_values(ascending=False).head(5)
        out["two_strike"] = {
            "batter_K_pct_2s": k_w / n_pa_w if n_pa_w else 0.0,
            "batter_xwoba_2s": w_xwoba(ts_bat),
            "batter_chase_2s": w_rate(ts_bat["description"].isin(SWING_DESCRIPTIONS) & ts_bat["zone"].isin(OOZ_ZONE),
                                       ts_bat["weight"]) /
                                max(w_rate(ts_bat["zone"].isin(OOZ_ZONE), ts_bat["weight"]), 1e-9)
                                if w_rate(ts_bat["zone"].isin(OOZ_ZONE), ts_bat["weight"]) else 0.0,
            "pitcher_xwoba_2s": w_xwoba(ts_pit),
            "pitcher_putaway_pct": k_w / float(ts_pit[ts_pit["events"].notna()]["weight"].sum())
                                   if float(ts_pit[ts_pit["events"].notna()]["weight"].sum()) else 0.0,
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
    ]
    return pd.DataFrame(rows, columns=["Metric", "Batter (vs same hand)", "Pitcher (vs same hand)"])


def discipline_notes(b: dict, p: dict) -> list[str]:
    notes = []
    if p["GB_pct"] > 0.50 and b["Air_pct"] > 0.60:
        notes.append("GB-heavy pitcher vs air-ball hitter — pitcher edge in BABIP suppression on contact.")
    if p["Whiff_pct"] > 0.27 and b["Whiff_pct"] > 0.27:
        notes.append("High-whiff pitcher meets a swing-and-miss prone hitter — strikeout odds elevated.")
    if p["Whiff_pct"] < 0.20 and b["Chase_pct"] < 0.25:
        notes.append("Contact-pitch-to-contact hitter — likely BIP-heavy AB.")
    if b["HardHit_pct"] > 0.50 and p["HardHit_pct"] < 0.32:
        notes.append("Elite hard-hit batter vs contact-suppressing pitcher — wash on quality of contact.")
    return notes


# ---------- Layer 8: edge analysis ----------------------------------------

def edge_analysis(batter_pt: pd.DataFrame, pitcher_pt: pd.DataFrame,
                  marginal: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    if batter_pt.empty or pitcher_pt.empty or marginal.empty:
        return pd.DataFrame(), pd.DataFrame()

    bat_idx = batter_pt.set_index("pitch_name")
    pit_idx = pitcher_pt.set_index("pitch_name")

    rows = []
    for pitch_name, p in marginal.items():
        b_x = float(bat_idx.loc[pitch_name, "xwOBA"]) if pitch_name in bat_idx.index else float("nan")
        pi_x = float(pit_idx.loc[pitch_name, "xwOBA"]) if pitch_name in pit_idx.index else float("nan")
        if math.isnan(b_x):
            continue
        edge = (b_x - LG_XWOBA)        # positive = batter does well vs this pitch
        rows.append({
            "Pitch": pitch_name,
            "Usage %": p * 100,
            "Batter xwOBA": b_x,
            "Pitcher xwOBA allowed": pi_x,
            "edge_score": p * edge,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df, df

    bat_fav = df.sort_values("edge_score", ascending=False).head(3).reset_index(drop=True)
    pit_fav = df.sort_values("edge_score", ascending=True).head(3).reset_index(drop=True)
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
    if abs(woba0 - target) > 0.005 and (H1 + H2 + H3 + HR) > 0:
        contact_woba = WOBA_1B * H1 + WOBA_2B * H2 + WOBA_3B * H3 + WOBA_HR * HR
        residual = target - (WOBA_BB * BB + WOBA_HBP * HBP)
        if contact_woba > 0 and residual > 0:
            scale = residual / contact_woba
            scale = float(np.clip(scale, 0.5, 2.0))
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
              handed: str) -> str:
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
        lines.append("| TTO | Proj xwOBA | Proj K % | Proj Whiff % | Proj Hard Hit % | Sample (eff PA) |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for _, r in tto_proj.iterrows():
            lines.append(
                f"| {int(r['TTO'])} | {fmt3(r['Projected xwOBA'])} | "
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
        if r["Metric"] == "xwOBA":
            lines.append(f"| {r['Metric']} | {fmt3(b_val)} | {fmt3(p_val)} |")
        else:
            lines.append(f"| {r['Metric']} | {b_val:.1f}% | {p_val:.1f}% |")
    if panel_notes:
        lines.append("")
        for n in panel_notes:
            lines.append(f"- {n}")
    lines.append("")

    # ----- Pitch-mix projection -----
    lines.append("## Pitch-mix projection")
    lines.append("")
    if pitch_table.empty:
        lines.append("_no arsenal data_")
    else:
        lines.append("| Pitch | Marginal Usage % | Batter xwOBA | Pitcher xwOBA allowed | Projected xwOBA | Projected Whiff % |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for _, r in pitch_table.iterrows():
            lines.append(
                f"| {r['Pitch']} | {r['Marginal Usage %']:.1f} | "
                f"{fmt3(r['Batter xwOBA'])} | {fmt3(r['Pitcher xwOBA allowed'])} | "
                f"{fmt3(r['Projected xwOBA'])} | {r['Projected Whiff %']:.1f} |"
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
        lines.append("| Pitch | Shape (eff velo / IVB / HB-in) | n comps (eff) | Whiff % | xwOBA | Hard Hit % | Confidence |")
        lines.append("|---|---|---:|---:|---:|---:|---|")
        for _, r in comps.iterrows():
            lines.append(
                f"| {r['Pitch']} | {r['Shape (eff velo / IVB / HB-in)']} | "
                f"{r['n comps (eff)']:.1f} | {fmt_pct(r['Whiff %'])} | "
                f"{fmt3(r['xwOBA'])} | {fmt_pct(r['Hard Hit %'])} | {r['Confidence']} |"
            )
    lines.append("")

    # ----- Zone overlay -----
    lines.append("## Zone overlay")
    lines.append("")
    if zone_overlay_df.empty:
        lines.append("_no zone data_")
    else:
        lines.append("| Pitch | In-zone % | Top zones (attack share) | Intersection xwOBA |")
        lines.append("|---|---:|---|---:|")
        for _, r in zone_overlay_df.iterrows():
            lines.append(
                f"| {r['Pitch']} | {r['In-zone %']:.1f} | "
                f"{r['Top zones']} | {fmt3(r['Intersection xwOBA'])} |"
            )
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
    lines.append("**Pitches favoring the hitter**")
    lines.append("")
    if bat_fav.empty:
        lines.append("_no data_")
    else:
        lines.append("| Pitch | Usage % | Batter xwOBA | Pitcher xwOBA allowed |")
        lines.append("|---|---:|---:|---:|")
        for _, r in bat_fav.iterrows():
            lines.append(
                f"| {r['Pitch']} | {r['Usage %']:.1f} | "
                f"{fmt3(r['Batter xwOBA'])} | {fmt3(r['Pitcher xwOBA allowed'])} |"
            )
    lines.append("")
    lines.append("**Pitches favoring the pitcher**")
    lines.append("")
    if pit_fav.empty:
        lines.append("_no data_")
    else:
        lines.append("| Pitch | Usage % | Batter xwOBA | Pitcher xwOBA allowed |")
        lines.append("|---|---:|---:|---:|")
        for _, r in pit_fav.iterrows():
            lines.append(
                f"| {r['Pitch']} | {r['Usage %']:.1f} | "
                f"{fmt3(r['Batter xwOBA'])} | {fmt3(r['Pitcher xwOBA allowed'])} |"
            )
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
            "- Not modeled: park / weather, catcher framing, umpire zone, fatigue beyond TTO, "
            "and rolling 14-day form. Treat single-season cells as noisy."
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
                     "<th>TTO</th><th>Proj xwOBA</th><th>Proj K %</th>"
                     "<th>Proj Whiff %</th><th>Proj Hard Hit %</th><th>Sample (eff PA)</th>"
                     "</tr></thead><tbody>")
        for _, r in tto_proj.iterrows():
            x = float(r["Projected xwOBA"])
            cls_x = edge_class(x, LG_XWOBA, 0.030, batter_favors_high=True)
            parts.append(
                "<tr>"
                f"<td>{int(r['TTO'])}</td>"
                f"{_td(f'{x:.3f}', cls_x)}"
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
        if metric == "xwOBA":
            cls_b = edge_class(b_val, LG_XWOBA, 0.030, batter_favors_high=True)
            cls_p = edge_class(p_val, LG_XWOBA, 0.030, batter_favors_high=True)
            parts.append(
                "<tr>"
                f"<td>{_h(metric)}</td>"
                f"{_td(f'{b_val:.3f}', cls_b)}"
                f"{_td(f'{p_val:.3f}', cls_p)}"
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

    # ----- Pitch-mix projection -----
    parts.append('<section class="card">')
    parts.append("<h2>Pitch-mix projection</h2>")
    if pitch_table.empty:
        parts.append('<p class="note">no arsenal data</p>')
    else:
        parts.append("<table><thead><tr>"
                     "<th>Pitch</th><th>Marginal Usage %</th>"
                     "<th>Batter xwOBA</th><th>Pitcher xwOBA allowed</th>"
                     "<th>Projected xwOBA</th><th>Projected Whiff %</th>"
                     "</tr></thead><tbody>")
        for _, r in pitch_table.iterrows():
            b_x = float(r["Batter xwOBA"])
            p_x = float(r["Pitcher xwOBA allowed"])
            proj_x = float(r["Projected xwOBA"])
            cls_b = edge_class(b_x, LG_XWOBA, 0.030, True)
            cls_p = edge_class(p_x, LG_XWOBA, 0.030, True)
            cls_proj = edge_class(proj_x, LG_XWOBA, 0.030, True)
            parts.append(
                "<tr>"
                f"<td>{_h(r['Pitch'])}</td>"
                f"<td>{r['Marginal Usage %']:.1f}</td>"
                f"{_td(f'{b_x:.3f}', cls_b)}"
                f"{_td(f'{p_x:.3f}', cls_p)}"
                f"{_td(f'{proj_x:.3f}', cls_proj)}"
                f"<td>{r['Projected Whiff %']:.1f}</td>"
                "</tr>"
            )
        parts.append("</tbody></table>")
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
                     "<th>Hard Hit %</th><th>Confidence</th>"
                     "</tr></thead><tbody>")
        for _, r in comps.iterrows():
            xw = r["xwOBA"]
            cls_x = edge_class(xw, LG_XWOBA, 0.030, True) if not (isinstance(xw, float) and math.isnan(xw)) else ""
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
                     "</tr></thead><tbody>")
        for _, r in zone_overlay_df.iterrows():
            ix = r["Intersection xwOBA"]
            cls_ix = edge_class(ix if not pd.isna(ix) else float("nan"),
                                LG_XWOBA, 0.030, True)
            parts.append(
                "<tr>"
                f"<td>{_h(r['Pitch'])}</td>"
                f"<td>{r['In-zone %']:.1f}</td>"
                f"<td>{_h(r['Top zones'])}</td>"
                f"{_td(fmt3(ix), cls_ix)}"
                "</tr>"
            )
        parts.append("</tbody></table>")
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
        parts.append(f'<div>batter xwOBA on first-pitch swings <b>{fp["batter_xwoba_on_swing"]:.3f}</b></div>')
        parts.append('</div>')
    if ts:
        parts.append('<div class="kv">')
        parts.append(f'<div>pitcher putaway% <b>{ts["pitcher_putaway_pct"]*100:.1f}%</b></div>')
        parts.append(f'<div>batter K% in 2-strike counts <b>{ts["batter_K_pct_2s"]*100:.1f}%</b></div>')
        parts.append(f'<div>batter xwOBA in 2-strike counts <b>{ts["batter_xwoba_2s"]:.3f}</b></div>')
        parts.append('</div>')
        if ts.get("two_strike_mix"):
            mix_pills = " ".join(f'<span class="pill">{_h(k)} {v:.1f}%</span>'
                                 for k, v in ts["two_strike_mix"].items())
            parts.append(f'<p class="subtitle">Pitcher\'s two-strike mix: {mix_pills}</p>')
    parts.append("</section>")

    # ----- Edge analysis -----
    parts.append('<section class="card">')
    parts.append("<h2>Edge analysis</h2>")
    parts.append('<p class="subtitle"><b>Pitches favoring the hitter</b></p>')
    if bat_fav.empty:
        parts.append('<p class="note">no data</p>')
    else:
        parts.append("<table><thead><tr>"
                     "<th>Pitch</th><th>Usage %</th><th>Batter xwOBA</th>"
                     "<th>Pitcher xwOBA allowed</th>"
                     "</tr></thead><tbody>")
        for _, r in bat_fav.iterrows():
            parts.append(
                "<tr>"
                f"<td>{_h(r['Pitch'])}</td>"
                f"<td>{r['Usage %']:.1f}</td>"
                f"{_td(f'{r['Batter xwOBA']:.3f}', 'bat-edge-strong')}"
                f"<td>{r['Pitcher xwOBA allowed']:.3f}</td>"
                "</tr>"
            )
        parts.append("</tbody></table>")
    parts.append('<p class="subtitle" style="margin-top:14px"><b>Pitches favoring the pitcher</b></p>')
    if pit_fav.empty:
        parts.append('<p class="note">no data</p>')
    else:
        parts.append("<table><thead><tr>"
                     "<th>Pitch</th><th>Usage %</th><th>Batter xwOBA</th>"
                     "<th>Pitcher xwOBA allowed</th>"
                     "</tr></thead><tbody>")
        for _, r in pit_fav.iterrows():
            parts.append(
                "<tr>"
                f"<td>{_h(r['Pitch'])}</td>"
                f"<td>{r['Usage %']:.1f}</td>"
                f"<td>{r['Batter xwOBA']:.3f}</td>"
                f"{_td(f'{r['Pitcher xwOBA allowed']:.3f}', 'pit-edge-strong')}"
                "</tr>"
            )
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
        parts.append("</main></body></html>")
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
        "<li>Not modeled: park / weather, catcher framing, umpire zone, fatigue beyond TTO, rolling form. "
        "Treat single-season cells as noisy.</li>"
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
    proj = project(batter_pt, pitcher_pt, batter_overall, pitcher_overall, marginal)
    comps = shape_comps(bat_vs_pit, pitcher_pt)
    zone_df = zone_overlay(pit_vs_bat, bat_vs_pit, pitcher_pt)
    tto = tto_curve(pit_vs_bat)
    tto_proj = tto_projections(proj, tto, pitcher_overall)
    bat_track_overall, bat_track_pitch = bat_tracking(bat_vs_pit, pit_vs_bat, pitcher_pt)
    sub = count_subprofiles(bat_vs_pit, pit_vs_bat)
    panel = discipline_panel(batter_overall, pitcher_overall)
    panel_notes = discipline_notes(batter_overall, pitcher_overall)
    bat_fav, pit_fav = edge_analysis(batter_pt, pitcher_pt, marginal)
    decep = deception(pitcher_pt)
    handed_note = handedness_verdict(batter_meta["stand"], pitcher_meta["p_throws"])
    outcomes = outcome_distribution(proj, batter_pt, marginal)
    multi_pa = multi_pa_outlook(outcomes, ns=(2, 3, 4))

    v = verdict(proj["xwOBA"], LG_XWOBA, pitcher_overall["xwOBA"], batter_overall["xwOBA"])
    narrative_text = narrative(batter_meta, pitcher_meta,
                                proj, v, bat_fav, pit_fav, handed_note)
    align = alignment_split(bat_vs_pit, pit_vs_bat)
    cm = count_mix_summary(pit_vs_bat)

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

    return {
        "spot": spot,
        "name": bm["name"],
        "stand": bm["stand"] or "?",
        "proj_xwoba": proj["xwOBA"],
        "delta_pts": delta,
        "k_pct": _outcome_prob(out, "Strikeout"),
        "bb_pct": _outcome_prob(out, "Walk"),
        "hr_pct": _outcome_prob(out, "Home Run"),
        "hit_pct": _outcome_prob(out, "Hit (any)"),
        "ob_pct": _outcome_prob(out, "On-base"),
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
        "| # | Batter | Hand | Proj xwOBA | Δ (pts) | K% | BB% | HR% | Hit% | OB% | Best pitch (xwOBA) | Worst pitch (xwOBA) | Verdict |"
    )
    lines.append("|---:|---|:-:|---:|---:|---:|---:|---:|---:|---:|---|---|:-:|")
    for r in summary_rows:
        lines.append(
            f"| {r['spot']} | {r['name']} | {r['stand']}HB | "
            f"{r['proj_xwoba']:.3f} | {r['delta_pts']:+.0f} | "
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
    return (
        f'<span class="spot">{r["spot"]}.</span>'
        f'<span class="name">{_h(r["name"])}</span>'
        f'<span class="badge">{_h(r["stand"])}HB</span>'
        f'<span class="summary-stat">proj xwOBA <b>{r["proj_xwoba"]:.3f}</b> ({delta_str})</span>'
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
                 '<th>Proj xwOBA</th><th>&Delta; (pts)</th>'
                 '<th>K%</th><th>BB%</th><th>HR%</th><th>Hit%</th><th>OB%</th>'
                 '<th style="text-align:left">Best pitch</th>'
                 '<th style="text-align:left">Worst pitch</th>'
                 '<th>Verdict</th>'
                 '</tr></thead><tbody>')
    for r in summary_rows:
        # Color cells: xwOBA + delta against league; K/BB/HR/Hit against league outcome rates.
        proj_cls = edge_class(r["proj_xwoba"], LG_XWOBA, 0.025, batter_favors_high=True)
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

    parts.append("</main></body></html>")
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


def _report_dir() -> Path:
    """Return (and create) today's report output directory: reports/<YYYY-MM-DD>/."""
    d = ROOT / "reports" / date.today().isoformat()
    d.mkdir(parents=True, exist_ok=True)
    return d


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
    #   (matchup_key, pitcher_name) -> list of (position, hitter_name, status, hitter_team)
    lineup_groups: dict[tuple[str, str], list[tuple[int, str, str, str]]] = {}

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            row = [c.strip() for c in row if c.strip()]
            if len(row) < 2:
                continue

            if len(row) >= 4 and "@" in row[0]:
                # Lineup format: away@home,hitter_name,pitcher_name,lineup_position[,status]
                matchup_key, hitter_name, pitcher_name = row[0], row[1], row[2]
                try:
                    pos = int(row[3])
                except ValueError:
                    pos = len(lineup_groups.get((matchup_key, pitcher_name), [])) + 1
                status = row[4].lower() if len(row) >= 5 else "confirmed"
                hitter_team = row[5].strip().upper() if len(row) >= 6 and row[5].strip() else ""
                lineup_groups.setdefault((matchup_key, pitcher_name), []).append(
                    (pos, hitter_name, status, hitter_team)
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
        for (_mk, pname), batters in lineup_groups.items():
            all_pitchers.add(pname)
            for _pos, h, _status, _team in batters:
                all_hitters.add(h)

        hitter_id_map = {h: _resolve_player(h) for h in all_hitters}
        pitcher_id_map = {p: _resolve_player(p) for p in all_pitchers}

        unique_b = set(hitter_id_map.values())
        unique_p = set(pitcher_id_map.values())
        print(f"  unique batters: {len(unique_b)}, unique pitchers: {len(unique_p)}")
        preload_player_data(unique_b, unique_p, season)

        n_workers = _resolve_workers(workers)
        print(f"  generating reports across {n_workers} worker thread(s)")

        def _process_lineup(item) -> None:
            (matchup_key, pitcher_name), batters = item
            batters_sorted = sorted(batters, key=lambda x: x[0])
            names_sorted = [name for _pos, name, _s, _t in batters_sorted]
            teams_sorted = [team for _pos, _n, _s, team in batters_sorted]
            is_projected = any(s == "projected" for _p, _n, s, _t in batters)
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
    args = ap.parse_args()

    if args.batch:
        run_batch(Path(args.batch), args.season, workers=args.workers)
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
