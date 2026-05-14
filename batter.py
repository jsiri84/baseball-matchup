"""
Pull a hitter's Statcast data from Baseball Savant and produce a
markdown summary that mirrors Savant's percentile-rankings card.

Player page reference:
  https://baseballsavant.mlb.com/savant-player/yordan-alvarez-670541
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from pybaseball import cache, playerid_lookup, statcast_batter

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)

cache.enable()

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

PLAYER_NAME = "Yordan Alvarez"
PLAYER_LAST = "alvarez"
PLAYER_FIRST = "yordan"
PLAYER_FALLBACK_ID = 670541

SEASON = 2026
SEASON_START = f"{SEASON}-03-27"
SEASON_END = date.today().isoformat()

# wOBA component weights (FanGraphs 2025; 2026 constants are not finalized
# until the offseason, so we use the most recent published values).
# Source: https://www.fangraphs.com/tools/guts
WOBA_BB = 0.691
WOBA_HBP = 0.722
WOBA_1B = 0.882
WOBA_2B = 1.252
WOBA_3B = 1.584
WOBA_HR = 2.037


# ---------- data access ----------------------------------------------------

def get_player_id(last: str, first: str, fallback: int | None = None) -> int:
    try:
        ids = playerid_lookup(last, first, fuzzy=True)
    except TypeError:
        ids = playerid_lookup(last, first)
    if ids is None or ids.empty:
        if fallback is not None:
            print(f"  lookup miss; using known MLBAM id {fallback}")
            return fallback
        raise SystemExit(f"No player found for {first} {last}")

    ids = ids.copy()
    ids["mlb_played_last"] = pd.to_numeric(ids["mlb_played_last"], errors="coerce").fillna(0)
    row = ids.sort_values("mlb_played_last", ascending=False).iloc[0]
    return int(row["key_mlbam"])


def pull(player_id: int, start: str, end: str) -> pd.DataFrame:
    year = start.split("-")[0]
    year_dir = DATA_DIR / year
    year_dir.mkdir(parents=True, exist_ok=True)
    cache_path = year_dir / f"statcast_{player_id}_{start}_{end}.parquet"

    # Back-compat: pick up legacy flat-layout files if a year-folder file does
    # not yet exist (pre-2026-05 layout was data/statcast_*.parquet).
    legacy_path = DATA_DIR / cache_path.name
    if not cache_path.exists() and legacy_path.exists():
        legacy_path.rename(cache_path)

    if cache_path.exists():
        print(f"  using cached file {year}/{cache_path.name}")
        return pd.read_parquet(cache_path)

    print(f"  fetching from Baseball Savant: {start} -> {end} ...")
    df = statcast_batter(start, end, player_id)
    df.to_parquet(cache_path, index=False)
    return df


# ---------- metric computation --------------------------------------------

@dataclass
class Stats:
    pa: int
    ab: int
    hits: int
    hr: int
    bb: int
    hbp: int
    so: int
    avg: float
    obp: float
    slg: float
    ops: float
    bbe: int
    avg_ev: float
    max_ev: float
    avg_la: float
    hard_hit_pct: float
    barrel_pct: float
    sweet_spot_pct: float
    bat_speed: float | None
    chase_pct: float
    whiff_pct: float
    k_pct: float
    bb_pct: float
    xba: float
    xslg: float
    xwoba: float
    batting_run_value: float


SWING_DESCRIPTIONS = {
    "hit_into_play", "foul", "foul_tip",
    "swinging_strike", "swinging_strike_blocked",
    "missed_bunt", "foul_bunt",
}
WHIFF_DESCRIPTIONS = {"swinging_strike", "swinging_strike_blocked"}
HIT_EVENTS = {"single", "double", "triple", "home_run"}
NON_AB_EVENTS = {"walk", "hit_by_pitch", "sac_fly", "sac_bunt", "catcher_interf"}

# Savant's three-group classification used on the "Pitch Tracking" card.
PITCH_GROUPS = {
    "Fastball": {"4-Seam Fastball", "Sinker", "Cutter"},
    "Breaking": {
        "Slider", "Curveball", "Sweeper", "Slurve",
        "Knuckle Curve", "Slow Curve", "Eephus", "Knuckleball",
    },
    "Offspeed": {"Changeup", "Split-Finger", "Forkball", "Screwball"},
}


def _pitch_group(name: str) -> str | None:
    for group, members in PITCH_GROUPS.items():
        if name in members:
            return group
    return None


def compute_stats(df: pd.DataFrame) -> Stats:
    pa_rows = df[df["events"].notna()].copy()
    n_pa = len(pa_rows)

    hits = pa_rows["events"].isin(HIT_EVENTS).sum()
    singles = (pa_rows["events"] == "single").sum()
    doubles = (pa_rows["events"] == "double").sum()
    triples = (pa_rows["events"] == "triple").sum()
    hr = (pa_rows["events"] == "home_run").sum()
    so = (pa_rows["events"] == "strikeout").sum()
    bb = (pa_rows["events"] == "walk").sum()
    hbp = (pa_rows["events"] == "hit_by_pitch").sum()

    ab_rows = pa_rows[~pa_rows["events"].isin(NON_AB_EVENTS)]
    ab = len(ab_rows)

    avg = hits / ab if ab else 0.0
    obp = (hits + bb + hbp) / n_pa if n_pa else 0.0
    tb = singles + 2 * doubles + 3 * triples + 4 * hr
    slg = tb / ab if ab else 0.0
    ops = obp + slg

    bbe = df[df["type"] == "X"].dropna(subset=["launch_speed"])
    n_bbe = len(bbe)
    avg_ev = bbe["launch_speed"].mean() if n_bbe else 0.0
    max_ev = bbe["launch_speed"].max() if n_bbe else 0.0
    avg_la = bbe["launch_angle"].mean() if n_bbe else 0.0
    hard_hit_pct = (bbe["launch_speed"] >= 95).mean() if n_bbe else 0.0
    barrel_pct = (bbe["launch_speed_angle"] == 6).mean() if n_bbe else 0.0
    sweet_spot_pct = bbe["launch_angle"].between(8, 32).mean() if n_bbe else 0.0

    bat_speed_series = (
        df["bat_speed"].dropna() if "bat_speed" in df.columns else pd.Series(dtype=float)
    )
    bat_speed = float(bat_speed_series.mean()) if len(bat_speed_series) else None

    swings = df["description"].isin(SWING_DESCRIPTIONS)
    whiffs = df["description"].isin(WHIFF_DESCRIPTIONS)
    out_of_zone = df["zone"].isin([11, 12, 13, 14])

    n_swings = swings.sum()
    whiff_pct = whiffs.sum() / n_swings if n_swings else 0.0
    chase_pct = (
        (swings & out_of_zone).sum() / out_of_zone.sum()
        if out_of_zone.sum() else 0.0
    )

    k_pct = so / n_pa if n_pa else 0.0
    bb_pct = bb / n_pa if n_pa else 0.0

    # Expected stats: Savant-style season aggregates.
    bbe_x = bbe.dropna(subset=["estimated_ba_using_speedangle"])
    sum_xba = bbe_x["estimated_ba_using_speedangle"].sum()
    sum_xslg = bbe_x["estimated_slg_using_speedangle"].sum()
    sum_xwoba_contact = bbe_x["estimated_woba_using_speedangle"].sum()

    # xBA: include K as 0 in denominator, exclude BB/HBP/sac
    xba_denom = len(bbe_x) + so
    xba = sum_xba / xba_denom if xba_denom else 0.0
    xslg = sum_xslg / xba_denom if xba_denom else 0.0

    # xwOBA: full PA-weighted, BB and HBP get fixed weights
    xwoba_num = sum_xwoba_contact + WOBA_BB * bb + WOBA_HBP * hbp
    xwoba = xwoba_num / n_pa if n_pa else 0.0

    batting_run_value = float(df["delta_run_exp"].sum()) if "delta_run_exp" in df.columns else 0.0

    return Stats(
        pa=int(n_pa), ab=int(ab), hits=int(hits), hr=int(hr), bb=int(bb),
        hbp=int(hbp), so=int(so), avg=avg, obp=obp, slg=slg, ops=ops,
        bbe=int(n_bbe), avg_ev=float(avg_ev), max_ev=float(max_ev),
        avg_la=float(avg_la), hard_hit_pct=float(hard_hit_pct),
        barrel_pct=float(barrel_pct), sweet_spot_pct=float(sweet_spot_pct),
        bat_speed=bat_speed, chase_pct=float(chase_pct), whiff_pct=float(whiff_pct),
        k_pct=float(k_pct), bb_pct=float(bb_pct),
        xba=float(xba), xslg=float(xslg), xwoba=float(xwoba),
        batting_run_value=batting_run_value,
    )


def _add_spray_direction(df: pd.DataFrame) -> pd.DataFrame:
    """Attach a Pull/Straight/Oppo label based on hit coordinates and handedness.

    Statcast hit coordinates put home plate near (125.42, 198.27). Negative
    spray angle = left-field side, positive = right-field side. We flip the
    sign for right-handed hitters so positive always means the batter's
    pull side, then bucket at +/- 15 degrees.
    """
    df = df.copy()
    has_xy = df["hc_x"].notna() & df["hc_y"].notna() & df["stand"].notna()
    raw = np.degrees(np.arctan2(df["hc_x"] - 125.42, 198.27 - df["hc_y"]))
    pull_signed = np.where(df["stand"].eq("R"), -raw, raw)
    df["spray_pull"] = np.where(has_xy, pull_signed, np.nan)
    df["direction"] = pd.cut(
        df["spray_pull"],
        bins=[-np.inf, -15, 15, np.inf],
        labels=["Oppo", "Straight", "Pull"],
    )
    return df


def batted_ball_profile(df: pd.DataFrame) -> dict:
    """GB/AIR/FB/LD/PU split, spray split, and 3x2 direction-by-trajectory grid."""
    bbe = df[df["type"] == "X"].dropna(subset=["bb_type"]).copy()
    if bbe.empty:
        return {}

    n = len(bbe)
    gb = (bbe["bb_type"] == "ground_ball").sum()
    fb = (bbe["bb_type"] == "fly_ball").sum()
    ld = (bbe["bb_type"] == "line_drive").sum()
    pu = (bbe["bb_type"] == "popup").sum()
    air = fb + ld + pu

    bbe = _add_spray_direction(bbe)
    bbe_dir = bbe.dropna(subset=["direction"])
    n_dir = len(bbe_dir)

    is_air = bbe_dir["bb_type"].isin(["fly_ball", "line_drive", "popup"])
    is_gb = bbe_dir["bb_type"].eq("ground_ball")
    direction = bbe_dir["direction"]

    def pct(num: float, denom: float) -> float:
        return (num / denom * 100) if denom else 0.0

    return {
        "n_bbe": int(n),
        "gb_pct": pct(gb, n),
        "air_pct": pct(air, n),
        "fb_pct": pct(fb, n),
        "ld_pct": pct(ld, n),
        "pu_pct": pct(pu, n),
        "pull_pct": pct((direction == "Pull").sum(), n_dir),
        "straight_pct": pct((direction == "Straight").sum(), n_dir),
        "oppo_pct": pct((direction == "Oppo").sum(), n_dir),
        "pull_gb_pct": pct(((direction == "Pull") & is_gb).sum(), n_dir),
        "straight_gb_pct": pct(((direction == "Straight") & is_gb).sum(), n_dir),
        "oppo_gb_pct": pct(((direction == "Oppo") & is_gb).sum(), n_dir),
        "pull_air_pct": pct(((direction == "Pull") & is_air).sum(), n_dir),
        "straight_air_pct": pct(((direction == "Straight") & is_air).sum(), n_dir),
        "oppo_air_pct": pct(((direction == "Oppo") & is_air).sum(), n_dir),
    }


def quality_of_contact(df: pd.DataFrame) -> dict:
    """Savant's six-bucket contact classification from launch_speed_angle (1..6)."""
    bbe = df[df["type"] == "X"].dropna(subset=["launch_speed_angle"]).copy()
    n_bbe = len(bbe)
    n_pa = int(df["events"].notna().sum())
    if n_bbe == 0:
        return {}

    counts = bbe["launch_speed_angle"].astype(int).value_counts()

    def share(bucket: int) -> float:
        return counts.get(bucket, 0) / n_bbe * 100

    return {
        "n_bbe": n_bbe,
        "weak_pct": share(1),
        "topped_pct": share(2),
        "under_pct": share(3),
        "flare_pct": share(4),
        "solid_pct": share(5),
        "barrel_pct": share(6),
        "barrel_per_pa": (counts.get(6, 0) / n_pa * 100) if n_pa else 0.0,
    }


def _subset_stats(group: pd.DataFrame, total_pitches: int) -> dict:
    """Compute one row's worth of pitch-level stats for a subset of pitches."""
    n_pitches = len(group)
    pa_rows = group[group["events"].notna()]
    n_pa = len(pa_rows)

    singles = int((pa_rows["events"] == "single").sum())
    doubles = int((pa_rows["events"] == "double").sum())
    triples = int((pa_rows["events"] == "triple").sum())
    hr = int((pa_rows["events"] == "home_run").sum())
    bb = int((pa_rows["events"] == "walk").sum())
    hbp = int((pa_rows["events"] == "hit_by_pitch").sum())
    sf = int((pa_rows["events"] == "sac_fly").sum())
    so = int((pa_rows["events"] == "strikeout").sum())

    hits = singles + doubles + triples + hr
    ab = len(pa_rows[~pa_rows["events"].isin(NON_AB_EVENTS)])
    tb = singles + 2 * doubles + 3 * triples + 4 * hr

    ba = hits / ab if ab else 0.0
    slg = tb / ab if ab else 0.0

    woba_num = (
        WOBA_BB * bb + WOBA_HBP * hbp
        + WOBA_1B * singles + WOBA_2B * doubles
        + WOBA_3B * triples + WOBA_HR * hr
    )
    woba_denom = ab + bb + sf + hbp
    woba = woba_num / woba_denom if woba_denom else 0.0

    swings = int(group["description"].isin(SWING_DESCRIPTIONS).sum())
    whiffs = int(group["description"].isin(WHIFF_DESCRIPTIONS).sum())
    whiff_pct = whiffs / swings * 100 if swings else 0.0
    k_pct = so / n_pa * 100 if n_pa else 0.0

    two_strike = group[group["strikes"] == 2]
    ks_2s = int((two_strike["events"] == "strikeout").sum())
    putaway = ks_2s / len(two_strike) * 100 if len(two_strike) else 0.0

    bbe_x = group[group["type"] == "X"].dropna(
        subset=["estimated_ba_using_speedangle"]
    )
    x_denom = len(bbe_x) + so
    xba = bbe_x["estimated_ba_using_speedangle"].sum() / x_denom if x_denom else 0.0
    xslg = bbe_x["estimated_slg_using_speedangle"].sum() / x_denom if x_denom else 0.0
    xwoba_num = (
        bbe_x["estimated_woba_using_speedangle"].sum()
        + WOBA_BB * bb + WOBA_HBP * hbp
    )
    xwoba = xwoba_num / woba_denom if woba_denom else 0.0

    bbe_all = group[group["type"] == "X"].dropna(subset=["launch_speed"])
    n_bbe = len(bbe_all)
    avg_ev = float(bbe_all["launch_speed"].mean()) if n_bbe else 0.0
    avg_la = float(bbe_all["launch_angle"].mean()) if n_bbe else 0.0
    hard_hit_pct = (
        (bbe_all["launch_speed"] >= 95).sum() / n_bbe * 100 if n_bbe else 0.0
    )

    rv = float(group["delta_run_exp"].sum()) if "delta_run_exp" in group.columns else 0.0
    rv_per_100 = (rv / n_pitches * 100) if n_pitches else 0.0

    return {
        "Pitches": n_pitches,
        "%": n_pitches / total_pitches * 100 if total_pitches else 0.0,
        "PA": n_pa, "AB": ab, "H": hits,
        "1B": singles, "2B": doubles, "3B": triples, "HR": hr, "SO": so,
        "BBE": n_bbe,
        "BA": ba, "SLG": slg, "wOBA": woba,
        "xBA": xba, "xSLG": xslg, "xwOBA": xwoba,
        "EV": avg_ev, "LA": avg_la,
        "Whiff %": whiff_pct, "K %": k_pct, "PutAway %": putaway,
        "Hard Hit %": hard_hit_pct,
        "RV/100": rv_per_100, "RV": rv,
    }


def pitch_type_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Per-pitch-name table mirroring Savant's 'Run Values by Pitch Type'."""
    if df.empty or "pitch_name" not in df.columns:
        return pd.DataFrame()

    total_pitches = len(df)
    rows = []
    for pitch_name, group in df.dropna(subset=["pitch_name"]).groupby("pitch_name"):
        rows.append({"Pitch": pitch_name, **_subset_stats(group, total_pitches)})

    return (
        pd.DataFrame(rows)
        .sort_values("Pitches", ascending=False)
        .reset_index(drop=True)
    )


def pitch_tracking_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Three-group (Fastball / Breaking / Offspeed) table from Savant's pitch-tracking card."""
    if df.empty or "pitch_name" not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df["pitch_group"] = df["pitch_name"].map(_pitch_group)
    df = df.dropna(subset=["pitch_group"])
    if df.empty:
        return pd.DataFrame()

    total_pitches = len(df)
    rows = []
    for group_name in ("Fastball", "Breaking", "Offspeed"):
        sub = df[df["pitch_group"] == group_name]
        if sub.empty:
            continue
        rows.append({"Pitch Type": group_name, **_subset_stats(sub, total_pitches)})

    return pd.DataFrame(rows).reset_index(drop=True)


def top_homers_table(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    hr = df[df["events"] == "home_run"].copy()
    if hr.empty:
        return pd.DataFrame()
    cols = ["game_date", "pitch_name", "launch_speed", "launch_angle", "hit_distance_sc"]
    hr = hr.sort_values("launch_speed", ascending=False).head(n)[cols]
    return hr.rename(columns={
        "game_date": "Date", "pitch_name": "Pitch",
        "launch_speed": "EV (mph)", "launch_angle": "LA (deg)",
        "hit_distance_sc": "Distance (ft)",
    })


# ---------- markdown rendering --------------------------------------------

def fmt_pct(x: float) -> str: return f"{x * 100:.1f}%"
def fmt3(x: float) -> str: return f"{x:.3f}"
def fmt1(x: float) -> str: return f"{x:.1f}"


def _format_pitch_table(table: pd.DataFrame) -> pd.DataFrame:
    """Pretty-print numeric columns for a pitch-breakdown DataFrame."""
    out = table.copy()
    pct_cols = ["%", "Whiff %", "K %", "PutAway %", "Hard Hit %"]
    rate_cols = ["BA", "SLG", "wOBA", "xBA", "xSLG", "xwOBA"]
    one_dec_cols = ["EV", "LA"]
    signed_cols = ["RV/100", "RV"]
    for col in pct_cols:
        if col in out.columns:
            out[col] = out[col].map(lambda v: f"{v:.1f}")
    for col in rate_cols:
        if col in out.columns:
            out[col] = out[col].map(lambda v: f"{v:.3f}")
    for col in one_dec_cols:
        if col in out.columns:
            out[col] = out[col].map(lambda v: f"{v:.1f}")
    for col in signed_cols:
        if col in out.columns:
            out[col] = out[col].map(lambda v: f"{v:+.1f}")
    return out


def to_markdown(name: str, mlbam_id: int, start: str, end: str,
                s: Stats, pitch_table: pd.DataFrame,
                hr_table: pd.DataFrame,
                batted_ball: dict, contact: dict,
                tracking_table: pd.DataFrame) -> str:
    bat_speed_str = f"{s.bat_speed:.1f} mph" if s.bat_speed is not None else "n/a"

    lines: list[str] = []
    lines.append(f"# {name} — Statcast summary")
    lines.append("")
    lines.append(
        f"_MLBAM id `{mlbam_id}` · window `{start}` to `{end}` · "
        f"source: Baseball Savant via pybaseball_"
    )
    lines.append("")

    lines.append("## Slash line")
    lines.append("")
    lines.append("| PA | AB | H | HR | BB | K | AVG | OBP | SLG | OPS |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    lines.append(
        f"| {s.pa} | {s.ab} | {s.hits} | {s.hr} | {s.bb} | {s.so} | "
        f"{fmt3(s.avg)} | {fmt3(s.obp)} | {fmt3(s.slg)} | {fmt3(s.ops)} |"
    )
    lines.append("")

    lines.append("## Value")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Batting Run Value (sum of run-expectancy delta) | {s.batting_run_value:+.1f} |")
    lines.append(f"| Baserunning Run Value | _not in this dataset_ |")
    lines.append(f"| Fielding Run Value    | _not in this dataset_ |")
    lines.append("")

    lines.append("## Batting (matches Savant percentile card)")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| xwOBA              | {fmt3(s.xwoba)} |")
    lines.append(f"| xBA                | {fmt3(s.xba)} |")
    lines.append(f"| xSLG               | {fmt3(s.xslg)} |")
    lines.append(f"| Avg Exit Velocity  | {fmt1(s.avg_ev)} mph |")
    lines.append(f"| Max Exit Velocity  | {fmt1(s.max_ev)} mph |")
    lines.append(f"| Barrel %           | {fmt_pct(s.barrel_pct)} |")
    lines.append(f"| Hard-Hit %         | {fmt_pct(s.hard_hit_pct)} |")
    lines.append(f"| LA Sweet-Spot %    | {fmt_pct(s.sweet_spot_pct)} |")
    lines.append(f"| Bat Speed          | {bat_speed_str} |")
    lines.append(f"| Squared-Up %       | _requires MLB's collision-EV formula; skipped_ |")
    lines.append(f"| Chase %            | {fmt_pct(s.chase_pct)} |")
    lines.append(f"| Whiff %            | {fmt_pct(s.whiff_pct)} |")
    lines.append(f"| K %                | {fmt_pct(s.k_pct)} |")
    lines.append(f"| BB %               | {fmt_pct(s.bb_pct)} |")
    lines.append("")

    lines.append("## Batted ball profile")
    lines.append("")
    if not batted_ball:
        lines.append("_No batted balls in this window._")
    else:
        b = batted_ball
        lines.append(f"_n = {b['n_bbe']} batted-ball events_")
        lines.append("")
        lines.append("**Trajectory mix**")
        lines.append("")
        lines.append("| GB% | AIR% | FB% | LD% | PU% |")
        lines.append("|---:|---:|---:|---:|---:|")
        lines.append(
            f"| {b['gb_pct']:.1f} | {b['air_pct']:.1f} | "
            f"{b['fb_pct']:.1f} | {b['ld_pct']:.1f} | {b['pu_pct']:.1f} |"
        )
        lines.append("")
        lines.append("**Spray (relative to batter handedness)**")
        lines.append("")
        lines.append("| Pull% | Straight% | Oppo% |")
        lines.append("|---:|---:|---:|")
        lines.append(
            f"| {b['pull_pct']:.1f} | {b['straight_pct']:.1f} | {b['oppo_pct']:.1f} |"
        )
        lines.append("")
        lines.append("**Direction by trajectory**")
        lines.append("")
        lines.append("|        | Pull | Straight | Oppo |")
        lines.append("|---|---:|---:|---:|")
        lines.append(
            f"| GB %   | {b['pull_gb_pct']:.1f} | "
            f"{b['straight_gb_pct']:.1f} | {b['oppo_gb_pct']:.1f} |"
        )
        lines.append(
            f"| AIR %  | {b['pull_air_pct']:.1f} | "
            f"{b['straight_air_pct']:.1f} | {b['oppo_air_pct']:.1f} |"
        )
    lines.append("")

    lines.append("## Quality of contact")
    lines.append("")
    if not contact:
        lines.append("_No batted balls in this window._")
    else:
        c = contact
        lines.append(f"_n = {c['n_bbe']} classified batted-ball events_")
        lines.append("")
        lines.append("| Weak% | Topped% | Under% | Flare/Burner% | Solid% | Barrel% | Barrel/PA% |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|")
        lines.append(
            f"| {c['weak_pct']:.1f} | {c['topped_pct']:.1f} | "
            f"{c['under_pct']:.1f} | {c['flare_pct']:.1f} | "
            f"{c['solid_pct']:.1f} | {c['barrel_pct']:.1f} | "
            f"{c['barrel_per_pa']:.1f} |"
        )
    lines.append("")

    lines.append("## Pitch tracking (Fastball / Breaking / Offspeed)")
    lines.append("")
    if tracking_table.empty:
        lines.append("_No classified pitches in this window._")
    else:
        tracking_cols = [
            "Pitch Type", "Pitches", "%", "PA", "AB", "H",
            "1B", "2B", "3B", "HR", "SO", "BBE",
            "BA", "xBA", "SLG", "xSLG", "wOBA", "xwOBA",
            "EV", "LA", "Whiff %", "PutAway %",
        ]
        cols = [c for c in tracking_cols if c in tracking_table.columns]
        lines.append(_format_pitch_table(tracking_table[cols]).to_markdown(index=False))
    lines.append("")

    lines.append("## Run values by pitch type")
    lines.append("")
    if pitch_table.empty:
        lines.append("_No pitches in this window._")
    else:
        detail_cols = [
            "Pitch", "Pitches", "%", "PA", "BA", "SLG", "wOBA",
            "xBA", "xSLG", "xwOBA",
            "Whiff %", "K %", "PutAway %", "Hard Hit %",
            "RV/100", "RV",
        ]
        cols = [c for c in detail_cols if c in pitch_table.columns]
        lines.append(_format_pitch_table(pitch_table[cols]).to_markdown(index=False))
    lines.append("")

    lines.append("## Hardest-hit home runs")
    lines.append("")
    if hr_table.empty:
        lines.append("_No home runs in this window._")
    else:
        lines.append(hr_table.to_markdown(index=False))
    lines.append("")

    lines.append("## Notes on percentiles")
    lines.append("")
    lines.append(
        "Per-player percentiles cannot be computed from one player's pitches alone — "
        "they require a league reference distribution. The cheap way to add them is "
        "to call `pybaseball.statcast_batter_expected_stats(year)` and a few sibling "
        "leaderboard endpoints (one row per qualifying batter, ~150 rows total) and "
        "compute percentiles in pandas."
    )
    lines.append("")
    return "\n".join(lines)


# ---------- entry point ---------------------------------------------------

def main() -> None:
    print(f"Looking up {PLAYER_NAME}'s MLBAM id...")
    pid = get_player_id(PLAYER_LAST, PLAYER_FIRST, fallback=PLAYER_FALLBACK_ID)
    print(f"  MLBAM id = {pid}")

    print(f"\nPulling Statcast data for {SEASON_START} -> {SEASON_END}")
    df = pull(pid, SEASON_START, SEASON_END)
    print(f"  rows: {len(df):,}   columns: {len(df.columns)}")

    if len(df) == 0:
        print("\nNo Statcast rows in this window. Has the season started?")
        print("Try editing SEASON / SEASON_START / SEASON_END at the top of this file.")
        return

    s = compute_stats(df)
    pitch_table = pitch_type_breakdown(df)
    tracking_table = pitch_tracking_breakdown(df)
    hr_table = top_homers_table(df)
    batted_ball = batted_ball_profile(df)
    contact = quality_of_contact(df)

    md = to_markdown(
        PLAYER_NAME, pid, SEASON_START, SEASON_END,
        s, pitch_table, hr_table, batted_ball, contact, tracking_table,
    )
    out_path = Path(__file__).parent / f"{PLAYER_LAST}_{PLAYER_FIRST}_{SEASON}_summary.md"
    out_path.write_text(md, encoding="utf-8")

    print()
    print(md)
    print()
    print(f"Markdown summary written to {out_path}")


if __name__ == "__main__":
    main()
