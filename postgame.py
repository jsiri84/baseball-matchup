#!/usr/bin/env python3
"""Post-game summary report: pregame projections vs actual hitter results.

For a given slate date this script:

1. Loads every pregame projection sidecar in ``reports/<date>/_data/*.json``
   (written by ``matchup.py`` during the pregame run).
2. Pulls the box score for each game from MLB StatsAPI to resolve hitter
   MLBAM IDs and grab vanilla counting stats (PA/AB/H/BB/K/HR/RBI/R).
3. Pulls per-batter Statcast through ``batter.pull`` (reusing the
   ``data/<year>/`` parquet cache layout) and filters to that day's
   plate appearances to compute actual xwOBA and xBA Savant-style.
4. Joins projection vs actual per hitter, emits one HTML per matchup and
   a slate-wide HTML, plus a sidecar JSON for downstream weekly accuracy
   tracking.

Usage::

    python postgame.py --date 2026-05-14
    python postgame.py --date 2026-05-14 --no-refresh  # don't re-pull Statcast
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date as date_cls
from pathlib import Path

import pandas as pd
import requests
from pybaseball import cache as pyb_cache, statcast as pyb_statcast

from matchup import (
    LG_BB_PCT,
    LG_K_PCT,
    LG_OUTCOMES,
    LG_XBA,
    LG_XWOBA,
    _HTML_CSS,
    _h,
    _td,
    edge_class,
    fmt3,
    report_timestamp_html,
)
from log_setup import setup_logging
from sortable import sortable_html

pyb_cache.enable()

ROOT = Path(__file__).parent
STATSAPI = "https://statsapi.mlb.com/api/v1"

LG_HIT_PCT = sum(LG_OUTCOMES[k] for k in ("1B", "2B", "3B", "HR"))
LG_OB_PCT = sum(LG_OUTCOMES[k] for k in ("1B", "2B", "3B", "HR", "BB", "HBP"))

# Pass/fail + verdict-axis thresholds, in points of xwOBA (1 pt = 0.001).
# A hitter "passes" if the headline OR contact-axis delta is within tolerance,
# OR if proj/actual fall in the same xwOBA tier.  Thresholds widened from
# 50 -> 65 after empirical-Bayes shrinkage compressed projections toward the
# league mean, since most projections now sit near .315 and single-game actuals
# still have full variance -- the old 50-pt window was too strict.
PASS_FAIL_HEADLINE_THRESH = 65.0    # |actual_xwoba - proj_xwoba| in points
PASS_FAIL_CONTACT_THRESH = 65.0     # |actual_on_contact - proj_on_contact|
CONTACT_AXIS_MILD = 65.0            # verdict pill: "Beat/Under contact"
CONTACT_AXIS_STRONG = 130.0         # verdict pill: "Beat/Under contact (+/-)"
OUTCOME_AXIS_MILD = 30.0            # verdict pill: "Beat/Under outcome"
OUTCOME_AXIS_STRONG = 65.0          # verdict pill: "Beat/Under outcome (+/-)"

# Savant wOBA non-contact constants (FanGraphs 2025 guts; matches batter.py).
WOBA_BB = 0.696
WOBA_HBP = 0.722

EVENT_BB = {"walk"}
EVENT_IBB = {"intent_walk"}
EVENT_HBP = {"hit_by_pitch"}
EVENT_K = {"strikeout", "strikeout_double_play"}
EVENT_HIT = {"single", "double", "triple", "home_run"}
EVENT_SAC = {"sac_fly", "sac_fly_double_play", "sac_bunt", "sac_bunt_double_play"}
EVENT_NO_AB = EVENT_BB | EVENT_IBB | EVENT_HBP | EVENT_SAC | {
    "catcher_interf", "batter_interference"
}

# ---------- PA outcome bucketing for proper-scoring -----------------------

OUTCOME_CLASSES = ("K", "BB", "HBP", "1B", "2B", "3B", "HR", "Out")

# Single -> class map. Anything not listed (and not in EVENTS_TO_SKIP) is "Out".
EVENT_TO_CLASS = {
    "strikeout": "K", "strikeout_double_play": "K",
    "walk": "BB",
    "hit_by_pitch": "HBP",
    "single": "1B", "double": "2B", "triple": "3B", "home_run": "HR",
}
# Treat sac fly / sac fly DP / errors / fielders_choice / regular outs as "Out".

# Events excluded from scoring entirely (matches xwOBA exclusions).
EVENTS_TO_SKIP = (EVENT_IBB | {"sac_bunt", "sac_bunt_double_play",
                                "catcher_interf", "batter_interference"})

# League prior over the 8 classes (re-bucketed from matchup.LG_OUTCOMES).
LG_PRIOR_DIST = {
    "K":   LG_OUTCOMES["K"],
    "BB":  LG_OUTCOMES["BB"],
    "HBP": LG_OUTCOMES["HBP"],
    "1B":  LG_OUTCOMES["1B"],
    "2B":  LG_OUTCOMES["2B"],
    "3B":  LG_OUTCOMES["3B"],
    "HR":  LG_OUTCOMES["HR"],
    "Out": LG_OUTCOMES["BIP_out"],
}


def _pa_actual_class(events: object) -> str | None:
    """Map a Statcast `events` string to one of OUTCOME_CLASSES, or None to skip."""
    if events is None or (isinstance(events, float) and math.isnan(events)):
        return None
    ev = str(events)
    if ev in EVENTS_TO_SKIP:
        return None
    return EVENT_TO_CLASS.get(ev, "Out")


def _score_pa(actual_class: str, dist: dict) -> tuple[float, float]:
    """Multinomial log-loss and Brier for a single PA against `dist`.

    `dist` must be a dict keyed by OUTCOME_CLASSES summing to ~1.0. Missing keys
    are treated as 0 (clamped to a small epsilon for log).
    """
    eps = 1e-6
    p_actual = max(float(dist.get(actual_class, 0.0)), eps)
    logloss = -math.log(p_actual)
    brier = 0.0
    for cls in OUTCOME_CLASSES:
        p = float(dist.get(cls, 0.0))
        target = 1.0 if cls == actual_class else 0.0
        brier += (p - target) ** 2
    return logloss, brier


# ---------- sidecar loading -----------------------------------------------

def _load_sidecars(report_dir: Path) -> list[dict]:
    """Load all pregame game-entries for a date.

    Prefers the consolidated ``_data/slate.json`` written by matchup.py's batch
    mode (single canonical source of truth, regenerated on every run).  Falls
    back to legacy per-game JSONs for older report directories that predate
    the slate format.
    """
    data_dir = report_dir / "_data"
    if not data_dir.exists():
        return []

    slate_path = data_dir / "slate.json"
    if slate_path.exists():
        try:
            payload = json.loads(slate_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[postgame] failed to read {slate_path.name}: {exc}",
                  file=sys.stderr)
        else:
            games = payload.get("games", [])
            if games:
                return list(games)
            print(f"[postgame] {slate_path.name} contained no games; "
                  "falling back to legacy per-game JSONs.", file=sys.stderr)

    out: list[dict] = []
    seen_stems: set[str] = set()
    for p in sorted(data_dir.glob("*.json")):
        if p.name in ("slate.json",):
            continue
        if p.name.startswith("_postgame"):
            continue
        if p.name.startswith("\ufeff"):
            print(f"[postgame] skipping BOM-prefixed dupe {p.name!r}", file=sys.stderr)
            continue
        try:
            sc = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[postgame] skipping {p.name}: {exc}", file=sys.stderr)
            continue
        stem = sc.get("out_stem") or p.stem
        if stem in seen_stems:
            continue
        seen_stems.add(stem)
        out.append(sc)
    return out


# ---------- StatsAPI helpers ----------------------------------------------

def _fetch_schedule(date_str: str) -> list[dict]:
    """Return a list of game dicts for the given date (one per game_pk)."""
    r = requests.get(
        f"{STATSAPI}/schedule",
        params={"sportId": 1, "date": date_str, "hydrate": "team"},
        timeout=15,
    )
    r.raise_for_status()
    js = r.json()
    if not js.get("dates"):
        return []
    return js["dates"][0].get("games", [])


def _fetch_boxscore(game_pk: int) -> dict:
    r = requests.get(f"{STATSAPI}/game/{game_pk}/boxscore", timeout=15)
    r.raise_for_status()
    return r.json()


def _team_abbrev(team_block: dict) -> str:
    return (team_block.get("team", {}).get("abbreviation")
            or team_block.get("team", {}).get("triCode")
            or team_block.get("team", {}).get("teamCode")
            or "").upper()


def _norm_name(s: str) -> str:
    return "".join(c for c in s.lower() if c.isalnum())


def _build_box_index(schedule: list[dict]) -> dict:
    """Map matchup_key (AWAY@HOME) -> {team_abbr: {norm_name: (mlbam, stats)}}.

    `stats` is the StatsAPI batting block (PA/AB/H/BB/K/HR/RBI/R/...).
    """
    out: dict = {}
    for g in schedule:
        away_abbr = _team_abbrev(g["teams"]["away"])
        home_abbr = _team_abbrev(g["teams"]["home"])
        if not (away_abbr and home_abbr):
            continue
        key = f"{away_abbr}@{home_abbr}"
        try:
            box = _fetch_boxscore(g["gamePk"])
        except Exception as exc:
            print(f"[postgame] boxscore fetch failed for {key} ({g['gamePk']}): {exc}",
                  file=sys.stderr)
            continue
        out[key] = {"game_pk": g["gamePk"]}
        for side, side_abbr in (("away", away_abbr), ("home", home_abbr)):
            players = box["teams"][side].get("players", {})
            by_name: dict = {}
            for pkey, pdata in players.items():
                person = pdata.get("person", {})
                name = person.get("fullName") or ""
                mlbam = person.get("id")
                bat = (pdata.get("stats") or {}).get("batting") or {}
                if not name or not mlbam:
                    continue
                by_name[_norm_name(name)] = {
                    "mlbam": int(mlbam),
                    "name": name,
                    "stats": bat,
                }
            out[key][side_abbr] = by_name
    return out


# ---------- Statcast per-PA actuals ---------------------------------------

def _fetch_slate_statcast(date_str: str) -> pd.DataFrame:
    """One league-wide PA-level pull for the date. Returns one row per PA."""
    cache_path = ROOT / "data" / date_str.split("-")[0] / f"statcast_slate_{date_str}.parquet"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        print(f"[postgame] using cached slate Statcast: {cache_path.name}")
        df = pd.read_parquet(cache_path)
    else:
        print(f"[postgame] fetching league-wide Statcast for {date_str} ...")
        df = pyb_statcast(start_dt=date_str, end_dt=date_str)
        if df is None or df.empty:
            print(f"[postgame] WARN: no Statcast rows returned for {date_str}", file=sys.stderr)
            return pd.DataFrame()
        df.to_parquet(cache_path, index=False)
    if "events" not in df.columns:
        return pd.DataFrame()
    pa = df[df["events"].notna() & (df["events"] != "")].copy()
    if "batter" in pa.columns:
        pa["batter"] = pd.to_numeric(pa["batter"], errors="coerce").astype("Int64")
    return pa


def _pa_rows_for_batter(slate_pa: pd.DataFrame, mlbam: int) -> pd.DataFrame:
    if slate_pa.empty or "batter" not in slate_pa.columns:
        return pd.DataFrame()
    return slate_pa[slate_pa["batter"] == mlbam]


def _to_float(v, default=float("nan")) -> float:
    if v is None:
        return default
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if not math.isnan(f) else default


def _actual_xwoba_xba(pa_df: pd.DataFrame) -> tuple[float, float, int, int]:
    """Return (xwOBA, xBA, woba_denom_count, ab_denom_count).

    Conventions match ``matchup.w_xwoba`` / ``w_xba_xslg``:
      * K -> 0 in both numerators
      * uBB / HBP contribute WOBA_BB / WOBA_HBP; IBB excluded
      * Ball-in-play -> estimated_woba_using_speedangle /
        estimated_ba_using_speedangle (fallback to woba_value if NaN)
      * woba denom = PA - IBB - SH - catcher_interf
      * xBA denom = AB (excludes BB/HBP/SF/SH/catcher_interf)
    """
    if pa_df.empty:
        return float("nan"), float("nan"), 0, 0

    woba_num = 0.0
    woba_den = 0
    xba_num = 0.0
    ab_den = 0
    for _, row in pa_df.iterrows():
        ev = row.get("events")
        if ev in EVENT_IBB or ev in {"sac_bunt", "sac_bunt_double_play", "catcher_interf",
                                      "batter_interference"}:
            continue
        woba_den += 1
        if ev in EVENT_K:
            pass  # contributes 0 to both
        elif ev in EVENT_BB:
            woba_num += WOBA_BB
        elif ev in EVENT_HBP:
            woba_num += WOBA_HBP
        elif ev in EVENT_SAC:
            # SF/SH count in woba_den (as 0 numerator) but not in AB
            pass
        else:
            # Ball in play
            est_w = _to_float(row.get("estimated_woba_using_speedangle"))
            if math.isnan(est_w):
                est_w = _to_float(row.get("woba_value"), default=0.0)
            est_b = _to_float(row.get("estimated_ba_using_speedangle"))
            if math.isnan(est_b):
                est_b = 1.0 if ev in EVENT_HIT else 0.0
            woba_num += est_w
            ab_den += 1
            xba_num += est_b

        if ev in EVENT_K:
            ab_den += 1  # K counts as AB with xBA contribution 0

    xwoba = woba_num / woba_den if woba_den else float("nan")
    xba = xba_num / ab_den if ab_den else float("nan")
    return xwoba, xba, woba_den, ab_den


# ---------- join + compute ------------------------------------------------

def _stat_int(v) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _safe_div(a: float, b: float) -> float:
    return a / b if b else float("nan")


def _actual_contact_quality(pa_df: pd.DataFrame) -> dict:
    """Compute actual contact-quality stats from a hitter's PA rows.

    Returns dict with: n_bip, avg_ev, max_ev, hardhit_pct, sweetspot_pct,
    on_contact_xwoba, on_contact_xba. NaN where no BIP.
    """
    out = {
        "n_bip": 0,
        "avg_ev": float("nan"), "max_ev": float("nan"),
        "hardhit_pct": float("nan"), "sweetspot_pct": float("nan"),
        "on_contact_xwoba": float("nan"), "on_contact_xba": float("nan"),
    }
    if pa_df.empty or "events" not in pa_df.columns:
        return out
    # BIP = events that aren't K/BB/HBP/IBB/SAC and have a valid EV.
    contact_events = (~pa_df["events"].isin(EVENT_K | EVENT_BB | EVENT_IBB
                                            | EVENT_HBP | EVENT_SAC
                                            | {"catcher_interf",
                                               "batter_interference"})
                      & pa_df["events"].notna())
    bip = pa_df[contact_events].copy()
    if "launch_speed" in bip.columns:
        bip = bip[bip["launch_speed"].notna()]
    if bip.empty:
        return out

    n = len(bip)
    ev = bip["launch_speed"].astype(float)
    la = bip["launch_angle"].astype(float) if "launch_angle" in bip.columns else pd.Series([float("nan")] * n)

    out["n_bip"] = n
    out["avg_ev"] = float(ev.mean())
    out["max_ev"] = float(ev.max())
    out["hardhit_pct"] = float((ev >= 95).mean())
    out["sweetspot_pct"] = float(la.between(8, 32).mean()) if la.notna().any() else float("nan")

    # On-contact xwOBA / xBA: estimated_woba_using_speedangle / estimated_ba_using_speedangle.
    if "estimated_woba_using_speedangle" in bip.columns:
        est_w = bip["estimated_woba_using_speedangle"].astype(float)
        # Fallback to woba_value if NaN.
        if "woba_value" in bip.columns:
            est_w = est_w.where(est_w.notna(),
                                bip["woba_value"].astype(float))
        if est_w.notna().any():
            out["on_contact_xwoba"] = float(est_w.mean())
    if "estimated_ba_using_speedangle" in bip.columns:
        est_b = bip["estimated_ba_using_speedangle"].astype(float)
        if est_b.notna().any():
            out["on_contact_xba"] = float(est_b.mean())
    return out


def _split_pa_quality(pa_df: pd.DataFrame) -> tuple[list[dict], list[dict], int]:
    """Split each PA into hard-contact / weak-contact / K buckets.

    Hard contact: BIP with EV >= 95 mph. Weak contact: BIP with EV < 95 mph.
    K: strikeouts (not split further). Walks / HBP / IBB / sac bunts are
    excluded — they're surfaced separately in the discipline axis pill.

    Each list entry carries the EV and xBA so the row display can show e.g.
    "3 BIP: .520 / .080 / .720" — useful for spotting at-em-ball misfortune
    vs genuinely good contact.
    """
    hh: list[dict] = []
    weak: list[dict] = []
    n_k = 0
    if pa_df.empty or "events" not in pa_df.columns:
        return hh, weak, n_k
    for _, row in pa_df.iterrows():
        ev = row.get("events")
        if ev is None or (isinstance(ev, float) and math.isnan(ev)):
            continue
        if ev in EVENT_K:
            n_k += 1
            continue
        if ev in EVENT_BB or ev in EVENT_IBB or ev in EVENT_HBP or ev in EVENT_SAC:
            continue
        if ev in {"catcher_interf", "batter_interference"}:
            continue
        ls = _to_float(row.get("launch_speed"))
        if math.isnan(ls):
            continue
        xba = _to_float(row.get("estimated_ba_using_speedangle"))
        rec = {"ev": ls, "xba": xba}
        if ls >= 95:
            hh.append(rec)
        else:
            weak.append(rec)
    return hh, weak, n_k


def _score_hitter_pas(pa_df: pd.DataFrame, proj_dist: dict,
                      lg_dist: dict) -> dict:
    """Sum per-PA log-loss / Brier vs model and league prior.

    Returns dict with: n_scored, model_logloss, model_brier, lg_logloss,
    lg_brier, plus per_pa: list[(pa_idx, actual_class, logloss, brier)].
    """
    res = {
        "n_scored": 0,
        "model_logloss": float("nan"), "model_brier": float("nan"),
        "lg_logloss": float("nan"), "lg_brier": float("nan"),
        "per_pa": [],
    }
    if pa_df.empty:
        return res
    m_ll = m_b = lg_ll = lg_b = 0.0
    n = 0
    for idx, (_, row) in enumerate(pa_df.iterrows()):
        cls = _pa_actual_class(row.get("events"))
        if cls is None:
            continue
        ll_m, br_m = _score_pa(cls, proj_dist)
        ll_lg, br_lg = _score_pa(cls, lg_dist)
        m_ll += ll_m;   m_b += br_m
        lg_ll += ll_lg; lg_b += br_lg
        n += 1
        res["per_pa"].append({
            "pa_idx": idx,
            "actual_class": cls,
            "launch_speed": _to_float(row.get("launch_speed")),
            "launch_angle": _to_float(row.get("launch_angle")),
            "hit_distance_sc": _to_float(row.get("hit_distance_sc")),
            "est_ba": _to_float(row.get("estimated_ba_using_speedangle")),
            "est_woba": _to_float(row.get("estimated_woba_using_speedangle")),
            "model_logloss": ll_m,
            "model_brier": br_m,
            "lg_logloss": ll_lg,
            "lg_brier": br_lg,
        })
    if n:
        res["n_scored"] = n
        res["model_logloss"] = m_ll / n
        res["model_brier"] = m_b / n
        res["lg_logloss"] = lg_ll / n
        res["lg_brier"] = lg_b / n
    return res


def _three_axis_verdict(proj_xwoba_oc: float, actual_xwoba_oc: float,
                        proj_k: float, actual_k: float,
                        proj_bb: float, actual_bb: float,
                        proj_xwoba: float, actual_xwoba: float) -> dict:
    """Return three verdict pills: contact / discipline / outcome."""

    def _band(delta_pts: float, thresh_mild: float, thresh_strong: float,
              up_label: str, down_label: str,
              up_css_mild: str, up_css_strong: str,
              down_css_mild: str, down_css_strong: str,
              neutral_label: str = "Matched",
              neutral_css: str = "verdict-neutral") -> tuple[str, str]:
        if math.isnan(delta_pts):
            return "—", "verdict-neutral"
        if delta_pts >= thresh_strong:
            return up_label + " (+)", up_css_strong
        if delta_pts >= thresh_mild:
            return up_label, up_css_mild
        if delta_pts <= -thresh_strong:
            return down_label + " (-)", down_css_strong
        if delta_pts <= -thresh_mild:
            return down_label, down_css_mild
        return neutral_label, neutral_css

    # Contact axis: actual on-contact xwOBA vs projected.
    if proj_xwoba_oc is None or math.isnan(actual_xwoba_oc):
        c_delta = float("nan")
    else:
        c_delta = (actual_xwoba_oc - proj_xwoba_oc) * 1000.0
    c_label, c_css = _band(
        c_delta, CONTACT_AXIS_MILD, CONTACT_AXIS_STRONG,
        "Beat contact", "Under contact",
        "bat-edge-mild", "bat-edge-strong",
        "pit-edge-mild", "pit-edge-strong",
        neutral_label="Matched contact",
    )

    # Discipline axis: aggressive = more K + fewer BB than proj; disciplined = less K + more BB.
    if math.isnan(actual_k) or math.isnan(actual_bb):
        d_delta = float("nan")
    else:
        d_delta = ((actual_k - (proj_k or 0.0)) - (actual_bb - (proj_bb or 0.0))) * 100.0
    d_label, d_css = _band(
        d_delta, 8, 18,
        "More aggressive", "More disciplined",
        "pit-edge-mild", "pit-edge-strong",
        "bat-edge-mild", "bat-edge-strong",
        neutral_label="Matched approach",
    )

    # Outcome axis: actual xwOBA vs projected (the existing delta_pts).
    if proj_xwoba is None or math.isnan(actual_xwoba):
        o_delta = float("nan")
    else:
        o_delta = (actual_xwoba - proj_xwoba) * 1000.0
    o_label, o_css = _band(
        o_delta, OUTCOME_AXIS_MILD, OUTCOME_AXIS_STRONG,
        "Beat outcome", "Under outcome",
        "bat-edge-mild", "bat-edge-strong",
        "pit-edge-mild", "pit-edge-strong",
        neutral_label="Matched outcome",
    )

    return {
        "contact_label": c_label, "contact_css": c_css, "contact_delta": c_delta,
        "discipline_label": d_label, "discipline_css": d_css, "discipline_delta": d_delta,
        "outcome_label": o_label, "outcome_css": o_css, "outcome_delta": o_delta,
    }


def _compute_hitter_row(sr: dict, box_player: dict | None,
                        pa_df: pd.DataFrame) -> dict:
    """Merge one projection row with actuals.

    Adds three-axis verdict (contact/discipline/outcome), contact-quality
    actuals from Statcast (avg/max EV, hardhit%, sweetspot%, on-contact xwOBA),
    per-PA log-loss vs the model's outcome distribution and league prior, and
    a list of per-PA Statcast rows for the Savant-style grid.
    """
    proj_xwoba = sr.get("proj_xwoba")
    proj_k = sr.get("k_pct") or 0.0
    proj_bb = sr.get("bb_pct") or 0.0
    proj_hit = sr.get("hit_pct") or 0.0
    proj_ob = sr.get("ob_pct") or 0.0
    proj_hr = sr.get("hr_pct") or 0.0
    proj_hardhit = _to_float(sr.get("proj_hardhit_pct"))
    proj_whiff = _to_float(sr.get("proj_whiff_pct"))
    proj_xwoba_oc = _to_float(sr.get("proj_xwoba_on_contact"))
    proj_dist = sr.get("proj_dist") or {}

    played_flag = box_player is not None and bool(box_player.get("stats"))
    stats = (box_player or {}).get("stats", {}) if played_flag else {}

    pa = _stat_int(stats.get("plateAppearances"))
    ab = _stat_int(stats.get("atBats"))
    h = _stat_int(stats.get("hits"))
    bb = _stat_int(stats.get("baseOnBalls"))
    hbp = _stat_int(stats.get("hitByPitch"))
    so = _stat_int(stats.get("strikeOuts"))
    hr = _stat_int(stats.get("homeRuns"))
    rbi = _stat_int(stats.get("rbi"))
    r = _stat_int(stats.get("runs"))
    sf = _stat_int(stats.get("sacFlies"))
    tb = _stat_int(stats.get("totalBases"))

    avg = _safe_div(h, ab)
    obp_den = ab + bb + hbp + sf
    obp = _safe_div(h + bb + hbp, obp_den)
    slg = _safe_div(tb, ab)
    actual_k_pct = _safe_div(so, pa)
    actual_bb_pct = _safe_div(bb, pa)
    actual_hr_pct = _safe_div(hr, pa)
    actual_hbp_pct = _safe_div(hbp, pa)
    actual_hit_pct = _safe_div(h, ab)
    actual_ob_pct = obp

    actual_xwoba, actual_xba, _, _ = _actual_xwoba_xba(pa_df)

    delta_pts = ((actual_xwoba - proj_xwoba) * 1000.0
                 if proj_xwoba is not None and not math.isnan(actual_xwoba)
                 else float("nan"))

    # Contact-quality actuals (BIP-only stats).
    cq = _actual_contact_quality(pa_df)
    # Per-PA hard-contact / weak-contact / K split for the row display.
    hh_balls, weak_balls, n_k_pa = _split_pa_quality(pa_df)

    # Per-PA scoring vs model dist and league prior.
    score = _score_hitter_pas(pa_df, proj_dist or {}, LG_PRIOR_DIST)

    # Three-axis verdict.
    verdicts = _three_axis_verdict(
        proj_xwoba_oc=proj_xwoba_oc,
        actual_xwoba_oc=cq["on_contact_xwoba"],
        proj_k=proj_k, actual_k=actual_k_pct,
        proj_bb=proj_bb, actual_bb=actual_bb_pct,
        proj_xwoba=proj_xwoba, actual_xwoba=actual_xwoba,
    )

    # Pass/fail: did the projection roughly land?  Pass if ANY of:
    #   1. headline xwOBA within +/-PASS_FAIL_HEADLINE_THRESH pts (Marte case)
    #   2. proj and actual fall in the same xwOBA tier
    #      Good>=.380, Avg .270-.380, Bad<.270  (Jac case: both Bad)
    #   3. n_bip>=2 AND on-contact xwOBA within +/-PASS_FAIL_CONTACT_THRESH pts
    #      (unlucky-but-good-contact: 0-for-4 with three 105 mph outs).
    # NA only when no PA were played (DNP).
    def _tier(x):
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "?"
        if x >= 0.380:
            return "good"
        if x >= 0.270:
            return "avg"
        return "bad"

    h_delta = None
    if (proj_xwoba is not None and actual_xwoba is not None
            and not (isinstance(proj_xwoba, float) and math.isnan(proj_xwoba))
            and not (isinstance(actual_xwoba, float) and math.isnan(actual_xwoba))):
        h_delta = (actual_xwoba - proj_xwoba) * 1000.0
    c_delta = verdicts["contact_delta"]
    proj_tier = _tier(proj_xwoba)
    act_tier = _tier(actual_xwoba)

    if not played_flag or pa == 0:
        pass_fail = "na"
    elif h_delta is not None and abs(h_delta) < PASS_FAIL_HEADLINE_THRESH:
        pass_fail = "pass"
    elif proj_tier != "?" and act_tier != "?" and proj_tier == act_tier:
        pass_fail = "pass"
    elif (cq["n_bip"] >= 2 and c_delta is not None
          and not math.isnan(c_delta) and abs(c_delta) < PASS_FAIL_CONTACT_THRESH):
        pass_fail = "pass"
    else:
        pass_fail = "fail"

    # Backwards-compat: keep a single rolled-up verdict for any legacy renderer.
    if math.isnan(delta_pts):
        verdict = "DNP" if pa == 0 else "No xwOBA"
        verdict_css = "verdict-neutral"
    elif delta_pts >= 50:
        verdict, verdict_css = "Crushed proj", "bat-edge-strong"
    elif delta_pts >= 20:
        verdict, verdict_css = "Beat proj", "bat-edge-mild"
    elif delta_pts <= -50:
        verdict, verdict_css = "Way under", "pit-edge-strong"
    elif delta_pts <= -20:
        verdict, verdict_css = "Underperformed", "pit-edge-mild"
    else:
        verdict, verdict_css = "Met proj", "verdict-neutral"

    return {
        "name": sr.get("name"),
        "stand": sr.get("stand"),
        "spot": sr.get("spot"),
        "anchor": sr.get("anchor"),
        "played": played_flag and pa > 0,
        "pa": pa, "ab": ab, "h": h, "bb": bb, "k": so, "hr": hr,
        "rbi": rbi, "r": r, "hbp": hbp, "sf": sf,
        "avg": avg, "obp": obp, "slg": slg,
        "proj_xwoba": proj_xwoba,
        "actual_xwoba": actual_xwoba,
        "actual_xba": actual_xba,
        "delta_pts": delta_pts,
        "proj_k_pct": proj_k,   "actual_k_pct": actual_k_pct,
        "proj_bb_pct": proj_bb, "actual_bb_pct": actual_bb_pct,
        "proj_hr_pct": proj_hr, "actual_hr_pct": actual_hr_pct,
        "proj_hit_pct": proj_hit, "actual_hit_pct": actual_hit_pct,
        "proj_ob_pct": proj_ob, "actual_ob_pct": actual_ob_pct,
        # Contact-quality + discipline proj vs actual.
        "proj_xwoba_on_contact": proj_xwoba_oc,
        "actual_xwoba_on_contact": cq["on_contact_xwoba"],
        "actual_xba_on_contact": cq["on_contact_xba"],
        "proj_hardhit_pct": proj_hardhit,
        "actual_hardhit_pct": cq["hardhit_pct"],
        "actual_sweetspot_pct": cq["sweetspot_pct"],
        "actual_avg_ev": cq["avg_ev"],
        "actual_max_ev": cq["max_ev"],
        "actual_hbp_pct": actual_hbp_pct,
        "proj_whiff_pct": proj_whiff,
        "n_bip": cq["n_bip"],
        "hh_balls": hh_balls,
        "weak_balls": weak_balls,
        "n_k_pa": n_k_pa,
        # Per-PA scoring.
        "n_pa_scored": score["n_scored"],
        "model_logloss": score["model_logloss"],
        "model_brier": score["model_brier"],
        "lg_logloss": score["lg_logloss"],
        "lg_brier": score["lg_brier"],
        # Per-PA rows for the Savant grid.
        "per_pa_rows": _pa_grid_rows(pa_df),
        "pa_scoring": score["per_pa"],
        # Three-axis verdict.
        "verdict_contact": verdicts["contact_label"],
        "verdict_contact_css": verdicts["contact_css"],
        "verdict_contact_delta": verdicts["contact_delta"],
        "verdict_discipline": verdicts["discipline_label"],
        "verdict_discipline_css": verdicts["discipline_css"],
        "verdict_discipline_delta": verdicts["discipline_delta"],
        "verdict_outcome": verdicts["outcome_label"],
        "verdict_outcome_css": verdicts["outcome_css"],
        "verdict_outcome_delta": verdicts["outcome_delta"],
        "pass_fail": pass_fail,
        # Backwards-compat single-axis verdict.
        "verdict_label": verdict,
        "verdict_css": verdict_css,
    }


# ---------- per-PA Savant-style grid --------------------------------------

_RESULT_LABEL = {
    "single": "Single", "double": "Double", "triple": "Triple",
    "home_run": "Home Run",
    "walk": "Walk", "intent_walk": "IBB", "hit_by_pitch": "HBP",
    "strikeout": "Strikeout", "strikeout_double_play": "Strikeout (DP)",
    "field_out": "Out", "force_out": "Out",
    "grounded_into_double_play": "GIDP",
    "fielders_choice": "FC", "fielders_choice_out": "FC",
    "double_play": "Double Play", "triple_play": "Triple Play",
    "sac_fly": "Sac Fly", "sac_fly_double_play": "Sac Fly DP",
    "sac_bunt": "Sac Bunt", "sac_bunt_double_play": "Sac Bunt DP",
    "field_error": "Reached Error", "catcher_interf": "Catcher Int.",
    "batter_interference": "Batter Int.",
}


def _pa_grid_rows(pa_df: pd.DataFrame) -> list[dict]:
    """One row per PA for the Savant-style grid in the per-game HTML."""
    if pa_df.empty:
        return []
    rows: list[dict] = []
    for _, r in pa_df.iterrows():
        ev = r.get("events")
        if ev is None or (isinstance(ev, float) and math.isnan(ev)):
            continue
        rows.append({
            "inning": _stat_int(r.get("inning")) or None,
            "events": str(ev),
            "result": _RESULT_LABEL.get(str(ev), str(ev).replace("_", " ").title()),
            "launch_speed": _to_float(r.get("launch_speed")),
            "launch_angle": _to_float(r.get("launch_angle")),
            "hit_distance_sc": _to_float(r.get("hit_distance_sc")),
            "bat_speed": _to_float(r.get("bat_speed")),
            "pitch_velo": _to_float(r.get("release_speed")),
            "est_ba": _to_float(r.get("estimated_ba_using_speedangle")),
            "est_woba": _to_float(r.get("estimated_woba_using_speedangle")),
        })
    # Sort by inning ascending; PAs without inning at end.
    rows.sort(key=lambda x: (x["inning"] is None, x["inning"] or 0))
    return rows


# ---------- HTML rendering ------------------------------------------------

_POSTGAME_CSS_EXTRA = """
.verdict-neutral { color: var(--muted); }
.line { color: var(--muted); font-variant-numeric: tabular-nums; }
.delta-pos { color: #1c8c4e; font-weight: 600; }
.delta-neg { color: #b34141; font-weight: 600; }
.delta-zero { color: var(--muted); }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.verdict-pills { display: flex; flex-direction: column; gap: 2px; align-items: stretch; }
.verdict-pills .pill { font-size: 0.78em; padding: 1px 6px; text-align: center;
                      border-radius: 4px; white-space: nowrap; }
td.pass-fail { text-align: center; font-size: 1.3em; font-weight: 700; }
td.pass-fail.pf-pass { color: #1c8c4e; }
td.pass-fail.pf-fail { color: #b34141; }
td.pass-fail.pf-na   { color: var(--muted); font-size: 1.0em; font-weight: 400; }
.pa-grid-wrap { margin-top: 18px; }
.pa-grid-wrap h3 { margin: 6px 0 4px 0; font-size: 1.0em; }
.pa-grid-wrap .summary-line { font-size: 0.9em; color: var(--muted); margin-bottom: 6px;
                              font-variant-numeric: tabular-nums; }
.cq-table { border-collapse: collapse; font-size: 0.88em; margin-bottom: 8px;
            font-variant-numeric: tabular-nums; }
.cq-table th, .cq-table td { padding: 2px 8px; text-align: right; }
.cq-table th { font-weight: 500; color: var(--muted); }
.cq-table td.label { text-align: left; color: var(--muted); }
.pa-grid { border-collapse: collapse; font-size: 0.88em; font-variant-numeric: tabular-nums; }
.pa-grid th, .pa-grid td { padding: 2px 8px; }
.pa-grid th { font-weight: 500; color: var(--muted); border-bottom: 1px solid var(--border, #ccc); }
.pa-grid td { text-align: right; }
.pa-grid td.label { text-align: left; }
.pa-grid .hh   { background: rgba(179,65,65,0.18); font-weight: 600; }
.pa-grid .hh-strong { background: rgba(179,65,65,0.32); font-weight: 700; }
.pa-grid .ss   { background: rgba(28,140,78,0.14); }
.pa-grid .xba-hot   { background: rgba(28,140,78,0.20); font-weight: 600; }
.pa-grid .xba-cold  { background: rgba(179,65,65,0.18); }
.pa-grid .xwoba-hot { background: rgba(28,140,78,0.22); font-weight: 600; }
.pa-grid .xwoba-cold{ background: rgba(179,65,65,0.18); }
"""


def _fmt_pct(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v*100:.1f}%"


def _fmt_delta(v) -> tuple[str, str]:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—", "delta-zero"
    cls = "delta-pos" if v > 0 else "delta-neg" if v < 0 else "delta-zero"
    return f"{v:+.0f}", cls


def _fmt_num(v, ndigits: int = 1) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return ""
    return f"{v:.{ndigits}f}"


def _fmt_int(v) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return ""
    try:
        return f"{int(round(float(v)))}"
    except (TypeError, ValueError):
        return ""


_PASS_TOOLTIP = (
    f"PASS: headline xwOBA within \u00b1{PASS_FAIL_HEADLINE_THRESH:.0f} pts "
    "OR proj/actual in same xwOBA tier "
    "(Good\u2265.380, Avg .270-.380, Bad<.270) OR on-contact xwOBA "
    f"within \u00b1{PASS_FAIL_CONTACT_THRESH:.0f} pts with \u22652 BIP"
)
_FAIL_TOOLTIP = (
    f"FAIL: headline xwOBA off by >{PASS_FAIL_HEADLINE_THRESH:.0f} pts AND "
    "different xwOBA tier AND contact axis didn't save it"
)


def _pass_fail_cell(row: dict) -> str:
    pf = row.get("pass_fail") or "na"
    if pf == "pass":
        return f'<td class="pass-fail pf-pass" title="{_PASS_TOOLTIP}">&check;</td>'
    if pf == "fail":
        return f'<td class="pass-fail pf-fail" title="{_FAIL_TOOLTIP}">&times;</td>'
    return '<td class="pass-fail pf-na" title="DNP — no plate appearances">—</td>'


def _fmt_xba_list(balls: list[dict]) -> str:
    """Format a list of {ev, xba} dicts as 'N: .xxx/.xxx/.xxx'.

    xBA falls back to '?' if missing. Empty list renders as a single em-dash.
    """
    if not balls:
        return "—"
    parts = []
    for b in balls:
        x = b.get("xba")
        if x is None or (isinstance(x, float) and math.isnan(x)):
            parts.append("?")
        else:
            # Strip the leading zero to match Statcast style (.520 not 0.520).
            s = f"{x:.3f}"
            if s.startswith("0."):
                s = s[1:]
            elif s.startswith("-0."):
                s = "-" + s[2:]
            parts.append(s)
    return f"{len(balls)}: {' / '.join(parts)}"


def _verdict_pills_cell(row: dict) -> str:
    """Return a <td> with three stacked verdict pills (contact/discipline/outcome)."""
    parts = ['<td class="verdict"><div class="verdict-pills">']
    for label_key, css_key, title in (
        ("verdict_contact", "verdict_contact_css", "Contact"),
        ("verdict_discipline", "verdict_discipline_css", "Discipline"),
        ("verdict_outcome", "verdict_outcome_css", "Outcome"),
    ):
        css = row.get(css_key) or "verdict-neutral"
        label = row.get(label_key) or "—"
        parts.append(f'<span class="pill {_h(css)}" title="{_h(title)}">{_h(label)}</span>')
    parts.append("</div></td>")
    return "".join(parts)


def _line_text(row: dict) -> str:
    if not row["played"]:
        return "DNP"
    parts = [f"{row['h']}-for-{row['ab']}"]
    extras = []
    if row["hr"]: extras.append(f"{row['hr']} HR")
    if row["rbi"]: extras.append(f"{row['rbi']} RBI")
    if row["r"]:   extras.append(f"{row['r']} R")
    if row["bb"]:  extras.append(f"{row['bb']} BB")
    if row["k"]:   extras.append(f"{row['k']} K")
    if extras:
        parts.append(", ".join(extras))
    return " · ".join(parts)


def _grid_row(rank: int, row: dict, include_pitcher: bool = False,
              team: str | None = None, pitcher: str | None = None,
              p_throws: str | None = None) -> str:
    proj_x = row["proj_xwoba"]
    act_x = row["actual_xwoba"]
    delta_str, delta_cls = _fmt_delta(row["delta_pts"])
    proj_x_cls = edge_class(proj_x, LG_XWOBA, 0.025, batter_favors_high=True) if proj_x else ""
    act_x_cls = edge_class(act_x, LG_XWOBA, 0.025, batter_favors_high=True) if not math.isnan(act_x) else ""
    act_b_cls = edge_class(row["actual_xba"], LG_XBA, 0.025, batter_favors_high=True) if not math.isnan(row["actual_xba"]) else ""

    proj_oc = row.get("proj_xwoba_on_contact")
    act_oc = row.get("actual_xwoba_on_contact")
    contact_delta_str, contact_delta_cls = _fmt_delta(row.get("verdict_contact_delta"))

    hh_balls = row.get("hh_balls") or []
    weak_balls = row.get("weak_balls") or []
    n_k_pa = row.get("n_k_pa") or 0

    hh_str = _fmt_xba_list(hh_balls)
    weak_xba_str = _fmt_xba_list(weak_balls)
    if n_k_pa:
        if weak_balls:
            weak_cell = f"{weak_xba_str} · {n_k_pa}K"
        else:
            weak_cell = f"{n_k_pa}K"
    else:
        weak_cell = weak_xba_str

    pitcher_cell = ""
    team_cell = ""
    if include_pitcher:
        team_cell = f'<td class="handpill"><span class="pill">{_h(team or "?")}</span></td>'
        pitcher_cell = (f'<td class="name">{_h(pitcher or "?")} '
                        f'<span class="pill">{_h(p_throws or "?")}HP</span></td>')

    return (
        "<tr>"
        f'<td class="spot">{rank}</td>'
        + team_cell
        + f'<td class="name">{_h(row["name"] or "?")}</td>'
        f'<td class="handpill"><span class="pill">{_h(row.get("stand") or "?")}HB</span></td>'
        + pitcher_cell
        + f'{_td(f"{proj_x:.3f}" if proj_x is not None else "—", proj_x_cls)}'
        f'{_td(fmt3(act_x), act_x_cls)}'
        f'{_td(fmt3(row["actual_xba"]), act_b_cls)}'
        f'<td class="num {delta_cls}">{delta_str}</td>'
        f'<td class="num">{fmt3(proj_oc) if proj_oc is not None else "—"} → {fmt3(act_oc)}</td>'
        f'<td class="num {contact_delta_cls}">{contact_delta_str}</td>'
        f'<td class="num">{_h(hh_str)}</td>'
        f'<td class="num">{_h(weak_cell)}</td>'
        f'<td class="line">{_h(_line_text(row))}</td>'
        + _pass_fail_cell(row)
        + _verdict_pills_cell(row)
        + "</tr>"
    )


def _grid_header_html(include_pitcher: bool) -> str:
    pitcher_th = '<th style="text-align:left">Pitcher</th>' if include_pitcher else ""
    team_th = "<th>Team</th>" if include_pitcher else ""
    return (
        '<thead><tr>'
        '<th>#</th>'
        + team_th
        + '<th style="text-align:left">Batter</th><th>Hand</th>'
        + pitcher_th
        + '<th>Proj xwOBA</th><th>Actual xwOBA</th><th>Actual xBA</th>'
        '<th>&Delta; xwOBA pts</th>'
        '<th>On-contact xwOBA proj → act</th>'
        '<th>&Delta; contact pts</th>'
        '<th>Hard contact (xBA)</th>'
        '<th>Weak / K (xBA)</th>'
        '<th style="text-align:left">Line</th>'
        f'<th title="{_PASS_TOOLTIP}">Proj</th>'
        '<th>Verdict (contact / discipline / outcome)</th>'
        '</tr></thead>'
    )


def _render_pa_grid(row: dict) -> str:
    """Render the per-hitter Savant-style PA grid + contact-quality summary."""
    name = row.get("name") or "?"
    spot = row.get("spot")
    anchor = row.get("anchor") or ""
    rows = row.get("per_pa_rows") or []
    cq_table = _render_cq_summary(row)

    head = (f'<div class="pa-grid-wrap" id="{_h(anchor)}-pa">'
            f'<h3>#{spot} {_h(name)} '
            f'<span class="pill">{_h(row.get("stand") or "?")}HB</span> · '
            f'{_h(_line_text(row))}</h3>')

    if not rows:
        return head + '<div class="summary-line">No Statcast PA rows.</div></div>'

    body = [head, cq_table]
    body.append('<table class="pa-grid"><thead><tr>'
                '<th>Inn</th>'
                '<th class="label" style="text-align:left">Result</th>'
                '<th>EV</th><th>LA</th><th>Hit Dist</th>'
                '<th>Bat Speed</th><th>Pitch Velo</th>'
                '<th>xBA</th><th>xwOBA</th>'
                '</tr></thead><tbody>')

    for r in rows:
        ev = r["launch_speed"]
        la = r["launch_angle"]
        xba = r["est_ba"]
        xw = r["est_woba"]

        ev_cls = ""
        if not math.isnan(ev):
            if ev >= 100:
                ev_cls = "hh-strong"
            elif ev >= 95:
                ev_cls = "hh"

        la_cls = ""
        if not math.isnan(la) and 8 <= la <= 32:
            la_cls = "ss"

        xba_cls = ""
        if not math.isnan(xba):
            if xba >= 0.500:
                xba_cls = "xba-hot"
            elif xba <= 0.150:
                xba_cls = "xba-cold"

        xw_cls = ""
        if not math.isnan(xw):
            if xw >= 0.500:
                xw_cls = "xwoba-hot"
            elif xw <= 0.150:
                xw_cls = "xwoba-cold"

        body.append(
            "<tr>"
            f"<td>{_h(str(r['inning']) if r['inning'] is not None else '')}</td>"
            f'<td class="label">{_h(r["result"])}</td>'
            f'<td class="{ev_cls}">{_fmt_num(ev, 1)}</td>'
            f'<td class="{la_cls}">{_fmt_int(la)}</td>'
            f"<td>{_fmt_int(r['hit_distance_sc'])}</td>"
            f"<td>{_fmt_num(r['bat_speed'], 1)}</td>"
            f"<td>{_fmt_num(r['pitch_velo'], 1)}</td>"
            f'<td class="{xba_cls}">{fmt3(xba) if not math.isnan(xba) else ""}</td>'
            f'<td class="{xw_cls}">{fmt3(xw) if not math.isnan(xw) else ""}</td>'
            "</tr>"
        )

    body.append("</tbody></table>")
    body.append("</div>")
    return "\n".join(body)


def _render_cq_summary(row: dict) -> str:
    """Render the contact-quality + discipline proj-vs-actual mini-table."""
    n_bip = row.get("n_bip") or 0
    n_scored = row.get("n_pa_scored") or 0
    m_ll = row.get("model_logloss")
    lg_ll = row.get("lg_logloss")
    skill_line = ""
    if m_ll is not None and not math.isnan(m_ll) and lg_ll and not math.isnan(lg_ll):
        skill = 1.0 - (m_ll / lg_ll)
        skill_line = (f" · log-loss {m_ll:.3f} vs lg {lg_ll:.3f} "
                      f"(skill {skill*100:+.1f}% over {n_scored} PA)")

    def _row(label: str, proj, actual, fmt) -> str:
        return (f'<tr><td class="label">{_h(label)}</td>'
                f'<td>{fmt(proj)}</td><td>{fmt(actual)}</td></tr>')

    parts = [f'<div class="summary-line">BIP {n_bip}{skill_line}</div>',
             '<table class="cq-table"><thead><tr>'
             '<th class="label" style="text-align:left">Metric</th>'
             '<th>Projected</th><th>Actual</th></tr></thead><tbody>']
    parts.append(_row("On-contact xwOBA",
                      row.get("proj_xwoba_on_contact"),
                      row.get("actual_xwoba_on_contact"),
                      lambda v: fmt3(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else "—"))
    parts.append(_row("Hard-hit %", row.get("proj_hardhit_pct"),
                      row.get("actual_hardhit_pct"), _fmt_pct))
    parts.append(_row("Sweet-spot %", None,
                      row.get("actual_sweetspot_pct"), _fmt_pct))
    parts.append(_row("Avg EV", None, row.get("actual_avg_ev"),
                      lambda v: _fmt_num(v, 1) if v is not None else "—"))
    parts.append(_row("Max EV", None, row.get("actual_max_ev"),
                      lambda v: _fmt_num(v, 1) if v is not None else "—"))
    parts.append(_row("Whiff %", row.get("proj_whiff_pct"),
                      None, _fmt_pct))
    parts.append(_row("K %", row.get("proj_k_pct"),
                      row.get("actual_k_pct"), _fmt_pct))
    parts.append(_row("BB %", row.get("proj_bb_pct"),
                      row.get("actual_bb_pct"), _fmt_pct))
    parts.append(_row("HBP %", None,
                      row.get("actual_hbp_pct"), _fmt_pct))
    parts.append("</tbody></table>")
    return "\n".join(parts)


def _doc_open(title: str) -> list[str]:
    return [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        f"<title>{_h(title)}</title>",
        f"<style>{_HTML_CSS}{_POSTGAME_CSS_EXTRA}</style>",
        "</head><body>",
        '<main class="container">',
    ]


def _doc_close() -> list[str]:
    return [report_timestamp_html(), "</main>", sortable_html(), "</body></html>"]


def _render_game_html(sidecar: dict, rows: list[dict], date_str: str) -> str:
    title = (f"Post-game vs {sidecar.get('pitcher_name','?')} — "
             f"{sidecar.get('hitter_team','?')} ({date_str})")
    parts = _doc_open(title)
    parts.append('<header class="page-head">')
    parts.append(f"<h1>{_h(title)}</h1>")
    parts.append('<div class="meta">'
                 'Three-axis projection vs actual per hitter: <b>contact</b> '
                 '(on-contact xwOBA, hard-hit%), <b>discipline</b> (K%, BB%), '
                 'and <b>outcome</b> (rolled-up xwOBA). Contact-axis isolates '
                 'projection skill from BABIP noise — a 0-for-4 with three '
                 '105 mph at-em-balls can read "Matched contact / Matched approach '
                 '/ Under outcome" instead of looking like a model miss.</div>')
    parts.append("</header>")
    parts.append('<section class="card">')
    parts.append("<h2>Hitter results</h2>")
    parts.append('<table class="lineup-grid">')
    parts.append(_grid_header_html(include_pitcher=False))
    parts.append("<tbody>")
    for i, row in enumerate(rows, start=1):
        parts.append(_grid_row(i, row, include_pitcher=False))
    parts.append("</tbody></table>")
    parts.append('<p class="note">&Delta; xwOBA pts = (actual − proj) × 1000. '
                 '&Delta; contact pts = (actual on-contact xwOBA − proj on-contact xwOBA) × 1000. '
                 'On-contact xwOBA strips BABIP noise from results.</p>')
    parts.append("</section>")

    # Per-hitter Savant-style PA grids.
    parts.append('<section class="card">')
    parts.append("<h2>Per-PA contact quality (Savant view)</h2>")
    for row in rows:
        parts.append(_render_pa_grid(row))
    parts.append("</section>")

    parts.extend(_doc_close())
    return "\n".join(parts)


def _render_slate_html(rows_with_meta: list[dict], date_str: str) -> str:
    title = f"Post-game slate report — {date_str}"
    parts = _doc_open(title)
    parts.append('<header class="page-head">')
    parts.append(f"<h1>{_h(title)}</h1>")
    n_pass = sum(1 for m in rows_with_meta if (m["row"].get("pass_fail") == "pass"))
    n_fail = sum(1 for m in rows_with_meta if (m["row"].get("pass_fail") == "fail"))
    n_na = sum(1 for m in rows_with_meta if (m["row"].get("pass_fail") == "na"))
    n_evaluable = n_pass + n_fail
    pass_rate = (n_pass / n_evaluable * 100.0) if n_evaluable else 0.0
    parts.append(
        f'<div class="meta">{len(rows_with_meta)} projected hitters across the slate. '
        f'<b>Projection pass rate:</b> '
        f'<span style="color:#1c8c4e;font-weight:700">{n_pass} &check;</span> / '
        f'<span style="color:#b34141;font-weight:700">{n_fail} &times;</span> '
        f'({pass_rate:.0f}% of {n_evaluable} evaluable; {n_na} DNP/no BIP). '
        'Sorted by &Delta; contact xwOBA pts (descending) so projection-confirming '
        'unlucky games rise above BABIP-driven box-score beats. Verdict column '
        'shows three axes: contact / discipline / outcome.</div>')
    parts.append("</header>")
    parts.append('<section class="card">')
    parts.append("<h2>All hitters · projection vs actual</h2>")
    parts.append('<table class="lineup-grid">')
    parts.append(_grid_header_html(include_pitcher=True))
    parts.append("<tbody>")
    for i, m in enumerate(rows_with_meta, start=1):
        parts.append(_grid_row(i, m["row"], include_pitcher=True,
                               team=m["team"], pitcher=m["pitcher_name"],
                               p_throws=m["p_throws"]))
    parts.append("</tbody></table>")
    parts.append('<p class="note">Played hitters sorted by contact-axis '
                 '&Delta; (actual − projected on-contact xwOBA in points); DNPs at the bottom.</p>')
    parts.append("</section>")
    parts.extend(_doc_close())
    return "\n".join(parts)


# ---------- main ----------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", default=date_cls.today().isoformat(),
                    help="Slate date (YYYY-MM-DD). Defaults to today.")
    ap.add_argument("--reports-dir", default=None,
                    help="Override reports directory (default reports/<date>/). "
                         "Pregame sidecars are loaded from <reports-dir>/_data/*.json.")
    args = ap.parse_args()

    log_path = setup_logging("postgame")
    print(f"[postgame] logging to {log_path}")

    if args.reports_dir:
        report_dir = Path(args.reports_dir).resolve()
    else:
        report_dir = (ROOT / "reports" / args.date).resolve()

    if not report_dir.exists():
        sys.exit(f"[postgame] reports dir not found: {report_dir}")

    sidecars = _load_sidecars(report_dir)
    if not sidecars:
        sys.exit(f"[postgame] no pregame sidecar JSONs under {report_dir}/_data/")
    print(f"[postgame] loaded {len(sidecars)} pregame sidecar(s)")

    schedule = _fetch_schedule(args.date)
    if not schedule:
        sys.exit(f"[postgame] no games scheduled on {args.date}")
    print(f"[postgame] {len(schedule)} games on {args.date}; fetching box scores...")
    box_index = _build_box_index(schedule)
    print(f"[postgame] built box index for {len(box_index)} matchups")

    slate_pa = _fetch_slate_statcast(args.date)
    print(f"[postgame] slate Statcast: {len(slate_pa)} PA rows across "
          f"{slate_pa['batter'].nunique() if not slate_pa.empty else 0} batters")

    slate_rows: list[dict] = []
    game_outputs: list[Path] = []
    postgame_sidecar: list[dict] = []

    for sc in sidecars:
        matchup_key = sc.get("matchup_key") or ""
        team = sc.get("hitter_team") or ""
        pitcher_name = sc.get("pitcher_name") or ""
        p_throws = (sc.get("pitcher_meta") or {}).get("p_throws") or "?"
        out_stem = sc.get("out_stem") or "postgame"
        summary_rows = sc.get("summary_rows") or []

        game_block = box_index.get(matchup_key, {})
        team_block = game_block.get(team, {})

        rendered_rows: list[dict] = []
        for sr in summary_rows:
            name = sr.get("name") or ""
            box_player = team_block.get(_norm_name(name))
            mlbam = box_player.get("mlbam") if box_player else None
            pa_df = _pa_rows_for_batter(slate_pa, mlbam) if mlbam is not None else pd.DataFrame()

            row = _compute_hitter_row(sr, box_player, pa_df)
            row["mlbam"] = mlbam
            rendered_rows.append(row)
            slate_rows.append({
                "row": row,
                "team": team,
                "pitcher_name": pitcher_name,
                "p_throws": p_throws,
                "matchup_key": matchup_key,
            })

        # Order per-game rows by batting spot for the per-game file.
        rendered_rows.sort(key=lambda r: (r.get("spot") or 999))

        game_html = _render_game_html(sc, rendered_rows, args.date)
        postgame_dir = report_dir / "postgame"
        postgame_dir.mkdir(parents=True, exist_ok=True)
        out_path = postgame_dir / f"postgame_{out_stem}.html"
        out_path.write_text(game_html, encoding="utf-8")
        game_outputs.append(out_path)

        postgame_sidecar.append({
            "matchup_key": matchup_key,
            "hitter_team": team,
            "pitcher_name": pitcher_name,
            "out_stem": out_stem,
            "rows": rendered_rows,
        })

    # Slate: sort played hitters by contact-axis delta desc, DNPs at bottom.
    def _contact_sort_key(m: dict) -> float:
        d = m["row"].get("verdict_contact_delta")
        if d is None or (isinstance(d, float) and math.isnan(d)):
            return float("-inf")
        return float(d)

    played = [m for m in slate_rows if m["row"]["played"]
              and not math.isnan(_contact_sort_key(m))]
    dnps = [m for m in slate_rows if not m["row"]["played"]
            or math.isnan(_contact_sort_key(m))]
    played.sort(key=_contact_sort_key, reverse=True)
    slate_sorted = played + dnps

    slate_html = _render_slate_html(slate_sorted, args.date)
    postgame_dir = report_dir / "postgame"
    postgame_dir.mkdir(parents=True, exist_ok=True)
    slate_path = postgame_dir / f"postgame_{args.date}.html"
    slate_path.write_text(slate_html, encoding="utf-8")

    # Strip large per-PA payloads from the sidecar JSON to keep it readable.
    def _trim_for_sidecar(games: list[dict]) -> list[dict]:
        out = []
        for g in games:
            g_copy = dict(g)
            g_copy["rows"] = [
                {k: v for k, v in r.items()
                 if k not in {"per_pa_rows", "pa_scoring"}}
                for r in g.get("rows", [])
            ]
            out.append(g_copy)
        return out

    sidecar_path = report_dir / "_data" / f"_postgame_{args.date}.json"
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(
        json.dumps({"date": args.date, "games": _trim_for_sidecar(postgame_sidecar)},
                   indent=2, default=str),
        encoding="utf-8",
    )

    # Persist rolling accuracy store under data/accuracy/.
    n_hitter, n_pa = _persist_accuracy_store(args.date, slate_rows)

    def _rel(p: Path) -> str:
        try:
            return str(p.relative_to(ROOT))
        except ValueError:
            return str(p)

    print(f"[postgame] wrote {len(game_outputs)} per-game report(s)")
    print(f"[postgame] wrote slate report: {_rel(slate_path)}")
    print(f"[postgame] wrote sidecar:      {_rel(sidecar_path)}")
    print(f"[postgame] accuracy store:     {n_hitter} hitter-game rows, {n_pa} PA rows")

    # Quick text summary to stdout: by three-axis verdict.
    contact_w = contact_n = contact_l = 0
    outcome_w = outcome_n = outcome_l = 0
    dnp_count = 0
    for m in slate_rows:
        r = m["row"]
        if not r["played"]:
            dnp_count += 1
            continue
        cd = r.get("verdict_contact_delta")
        if cd is not None and not math.isnan(cd):
            if cd >= CONTACT_AXIS_MILD:
                contact_w += 1
            elif cd <= -CONTACT_AXIS_MILD:
                contact_l += 1
            else:
                contact_n += 1
        od = r.get("delta_pts")
        if od is not None and not math.isnan(od):
            if od >= OUTCOME_AXIS_MILD:
                outcome_w += 1
            elif od <= -OUTCOME_AXIS_MILD:
                outcome_l += 1
            else:
                outcome_n += 1
    print(f"[postgame] contact axis: {contact_w} beat / {contact_n} matched / "
          f"{contact_l} under ({CONTACT_AXIS_MILD:.0f}-pt threshold)")
    print(f"[postgame] outcome axis: {outcome_w} beat / {outcome_n} matched / "
          f"{outcome_l} under ({OUTCOME_AXIS_MILD:.0f}-pt threshold)")
    n_pass = sum(1 for m in slate_rows if m["row"].get("pass_fail") == "pass")
    n_fail = sum(1 for m in slate_rows if m["row"].get("pass_fail") == "fail")
    n_pf_eval = n_pass + n_fail
    pf_rate = (n_pass / n_pf_eval * 100.0) if n_pf_eval else 0.0
    print(f"[postgame] projection pass/fail: {n_pass} pass / {n_fail} fail "
          f"({pf_rate:.0f}% pass rate over {n_pf_eval} evaluable)")
    print(f"[postgame] DNP: {dnp_count}")
    return 0


# ---------- rolling accuracy store ----------------------------------------

ACCURACY_DIR = ROOT / "data" / "accuracy"
HITTER_PARQUET = ACCURACY_DIR / "hitter_results.parquet"
PA_PARQUET = ACCURACY_DIR / "pa_results.parquet"


def _persist_accuracy_store(date_str: str,
                            slate_rows: list[dict]) -> tuple[int, int]:
    """Append-only persistence keyed by date.

    On re-run for the same date, existing rows are dropped first to keep
    the store idempotent. Skips DNP hitters entirely.
    """
    ACCURACY_DIR.mkdir(parents=True, exist_ok=True)

    hitter_records: list[dict] = []
    pa_records: list[dict] = []

    for m in slate_rows:
        r = m["row"]
        if not r["played"]:
            continue
        mlbam = r.get("mlbam")
        if mlbam is None:
            continue
        base = {
            "date": date_str,
            "mlbam": int(mlbam),
            "name": r.get("name"),
            "team": m.get("team"),
            "opp_pitcher": m.get("pitcher_name"),
            "p_throws": m.get("p_throws"),
            "matchup_key": m.get("matchup_key"),
            "stand": r.get("stand"),
            "spot": r.get("spot"),
            "pa": r.get("pa"),
            "ab": r.get("ab"),
            "n_bip": r.get("n_bip"),
            "proj_xwoba": r.get("proj_xwoba"),
            "proj_xwoba_on_contact": r.get("proj_xwoba_on_contact"),
            "proj_hardhit_pct": r.get("proj_hardhit_pct"),
            "proj_whiff_pct": r.get("proj_whiff_pct"),
            "proj_k_pct": r.get("proj_k_pct"),
            "proj_bb_pct": r.get("proj_bb_pct"),
            "proj_hr_pct": r.get("proj_hr_pct"),
            "proj_hit_pct": r.get("proj_hit_pct"),
            "proj_ob_pct": r.get("proj_ob_pct"),
            "actual_xwoba": r.get("actual_xwoba"),
            "actual_xba": r.get("actual_xba"),
            "actual_xwoba_on_contact": r.get("actual_xwoba_on_contact"),
            "actual_xba_on_contact": r.get("actual_xba_on_contact"),
            "actual_hardhit_pct": r.get("actual_hardhit_pct"),
            "actual_sweetspot_pct": r.get("actual_sweetspot_pct"),
            "actual_avg_ev": r.get("actual_avg_ev"),
            "actual_max_ev": r.get("actual_max_ev"),
            "actual_k_pct": r.get("actual_k_pct"),
            "actual_bb_pct": r.get("actual_bb_pct"),
            "actual_hbp_pct": r.get("actual_hbp_pct"),
            "actual_hr_pct": r.get("actual_hr_pct"),
            "actual_hit_pct": r.get("actual_hit_pct"),
            "actual_ob_pct": r.get("actual_ob_pct"),
            "delta_pts": r.get("delta_pts"),
            "contact_delta_pts": r.get("verdict_contact_delta"),
            "discipline_delta_pts": r.get("verdict_discipline_delta"),
            "outcome_delta_pts": r.get("verdict_outcome_delta"),
            "n_pa_scored": r.get("n_pa_scored"),
            "model_logloss": r.get("model_logloss"),
            "model_brier": r.get("model_brier"),
            "lg_logloss": r.get("lg_logloss"),
            "lg_brier": r.get("lg_brier"),
        }
        hitter_records.append(base)

        # Per-PA rows.
        proj_dist = (m.get("row") or {}).get("proj_dist") or {}
        # proj_dist isn't on the row dict directly; pull from per-PA scoring entries
        # which already encode model log-loss/Brier — we additionally include the
        # projected dist columns by reading the original summary row off the slate.
        # For simplicity, store the projected dist as flat columns using whatever
        # is available from the hitter row.
        # Note: pa_scoring entries are written with model_logloss / model_brier and
        # the actual class; we attach proj dist by referencing the parent hitter.
        # Pull proj_dist from the sidecar via the hitter's row metadata.
        # Fallback: leave dist as NaN.
        for pa_entry in r.get("pa_scoring") or []:
            pa_records.append({
                "date": date_str,
                "mlbam": int(mlbam),
                "pa_idx": pa_entry["pa_idx"],
                "actual_class": pa_entry["actual_class"],
                "launch_speed": pa_entry["launch_speed"],
                "launch_angle": pa_entry["launch_angle"],
                "hit_distance_sc": pa_entry["hit_distance_sc"],
                "est_ba": pa_entry["est_ba"],
                "est_woba": pa_entry["est_woba"],
                "model_logloss": pa_entry["model_logloss"],
                "model_brier": pa_entry["model_brier"],
                "lg_logloss": pa_entry["lg_logloss"],
                "lg_brier": pa_entry["lg_brier"],
            })

    def _write(path: Path, new_records: list[dict]) -> None:
        if not new_records:
            return
        new_df = pd.DataFrame(new_records)
        if path.exists():
            try:
                old = pd.read_parquet(path)
                old = old[old["date"] != date_str]
                combined = pd.concat([old, new_df], ignore_index=True)
            except Exception as exc:
                print(f"[postgame] WARN: could not read {path.name} "
                      f"({exc}); overwriting.", file=sys.stderr)
                combined = new_df
        else:
            combined = new_df
        combined.to_parquet(path, index=False)

    _write(HITTER_PARQUET, hitter_records)
    _write(PA_PARQUET, pa_records)
    return len(hitter_records), len(pa_records)


if __name__ == "__main__":
    sys.exit(main())
