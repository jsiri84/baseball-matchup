"""
Pull a pitcher's Statcast data from Baseball Savant and produce a
markdown summary that mirrors Savant's pitcher percentile-rankings card.

Player page reference:
  https://baseballsavant.mlb.com/savant-player/paul-skenes-694973

Edit PLAYER_* and SEASON_* below to switch pitchers / windows.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
from pybaseball import cache, playerid_lookup, statcast_pitcher

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)

cache.enable()

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

PLAYER_NAME = "Paul Skenes"
PLAYER_LAST = "skenes"
PLAYER_FIRST = "paul"
PLAYER_FALLBACK_ID = 694973

SEASON = 2026
SEASON_START = f"{SEASON}-03-27"
SEASON_END = date.today().isoformat()

# wOBA component weights (FanGraphs 2025 GUTS! — most recent published values).
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
        try:
            ids = playerid_lookup(last, first, fuzzy=True)
        except TypeError:
            ids = playerid_lookup(last, first)
    except Exception as exc:                              # network/cert/etc.
        if fallback is not None:
            print(f"  lookup failed ({exc.__class__.__name__}); "
                  f"using known MLBAM id {fallback}")
            return fallback
        raise
    if ids is None or ids.empty:
        if fallback is not None:
            print(f"  lookup miss; using known MLBAM id {fallback}")
            return fallback
        raise SystemExit(f"No player found for {first} {last}")
    row = ids.sort_values("mlb_played_last", ascending=False).iloc[0]
    return int(row["key_mlbam"])


def pull(player_id: int, start: str, end: str) -> pd.DataFrame:
    year = start.split("-")[0]
    year_dir = DATA_DIR / year
    year_dir.mkdir(parents=True, exist_ok=True)
    cache_path = year_dir / f"statcast_pitcher_{player_id}_{start}_{end}.parquet"

    # Back-compat: pick up legacy flat-layout files if a year-folder file does
    # not yet exist (pre-2026-05 layout was data/statcast_pitcher_*.parquet).
    legacy_path = DATA_DIR / cache_path.name
    if not cache_path.exists() and legacy_path.exists():
        legacy_path.rename(cache_path)

    if cache_path.exists():
        print(f"  using cached file {year}/{cache_path.name}")
        return pd.read_parquet(cache_path)

    print(f"  fetching from Baseball Savant: {start} -> {end} ...")
    df = statcast_pitcher(start, end, player_id)
    df.to_parquet(cache_path, index=False)
    return df


# ---------- metric computation --------------------------------------------

@dataclass
class Stats:
    pitches: int
    pa: int
    ab: int
    hits: int
    hr: int
    bb: int
    hbp: int
    so: int
    ip_outs: int            # total outs recorded (innings = outs / 3)
    er_proxy: int           # earned-run proxy (events with rbi >= 1 result, fallback to hits-driven)
    ba: float               # opponent batting avg
    obp: float              # opponent OBP
    slg: float              # opponent SLG
    woba: float             # opponent wOBA
    bbe: int
    avg_ev_allowed: float
    max_ev_allowed: float
    hard_hit_pct: float     # share of BBE >= 95 mph
    barrel_pct: float       # share of BBE classified as barrels
    gb_pct: float           # share of BBE that are ground balls
    chase_pct: float
    whiff_pct: float
    k_pct: float
    bb_pct: float
    xba: float
    xslg: float
    xwoba: float
    xera: float             # rough xwOBA-based ERA approximation
    pitching_run_value: float
    fb_velo: float | None   # avg 4-seam / sinker / cutter velocity
    fb_spin: float | None
    cb_spin: float | None
    extension: float | None
    arm_angle: float | None


SWING_DESCRIPTIONS = {
    "hit_into_play", "foul", "foul_tip",
    "swinging_strike", "swinging_strike_blocked",
    "missed_bunt", "foul_bunt",
}
WHIFF_DESCRIPTIONS = {"swinging_strike", "swinging_strike_blocked"}
HIT_EVENTS = {"single", "double", "triple", "home_run"}
NON_AB_EVENTS = {"walk", "hit_by_pitch", "sac_fly", "sac_bunt", "catcher_interf"}
OUT_EVENTS = {
    "strikeout", "field_out", "force_out", "grounded_into_double_play",
    "double_play", "triple_play", "sac_fly", "sac_bunt", "sac_fly_double_play",
    "sac_bunt_double_play", "fielders_choice", "fielders_choice_out",
    "strikeout_double_play",
}
DOUBLE_PLAY_EVENTS = {
    "grounded_into_double_play", "double_play", "strikeout_double_play",
    "sac_fly_double_play", "sac_bunt_double_play",
}
TRIPLE_PLAY_EVENTS = {"triple_play"}

PITCH_GROUPS = {
    "Fastball": {"4-Seam Fastball", "Sinker", "Cutter"},
    "Breaking": {
        "Slider", "Curveball", "Sweeper", "Slurve",
        "Knuckle Curve", "Slow Curve", "Eephus", "Knuckleball",
    },
    "Offspeed": {"Changeup", "Split-Finger", "Forkball", "Screwball"},
}
FASTBALL_NAMES = {"4-Seam Fastball", "Sinker", "Cutter"}
CURVE_NAMES = {"Curveball", "Knuckle Curve", "Slow Curve"}


def _pitch_group(name: str) -> str | None:
    for group, members in PITCH_GROUPS.items():
        if name in members:
            return group
    return None


def _outs_recorded(pa_rows: pd.DataFrame) -> int:
    """Outs recorded across all PA terminating events, counting DPs as 2 and TPs as 3."""
    if pa_rows.empty:
        return 0
    is_out = pa_rows["events"].isin(OUT_EVENTS)
    is_dp = pa_rows["events"].isin(DOUBLE_PLAY_EVENTS)
    is_tp = pa_rows["events"].isin(TRIPLE_PLAY_EVENTS)
    return int(is_out.sum() + is_dp.sum() + 2 * is_tp.sum())


def compute_stats(df: pd.DataFrame) -> Stats:
    n_pitches = len(df)
    pa_rows = df[df["events"].notna()].copy()
    n_pa = len(pa_rows)

    singles = int((pa_rows["events"] == "single").sum())
    doubles = int((pa_rows["events"] == "double").sum())
    triples = int((pa_rows["events"] == "triple").sum())
    hr = int((pa_rows["events"] == "home_run").sum())
    hits = singles + doubles + triples + hr
    so = int((pa_rows["events"] == "strikeout").sum())
    so += int((pa_rows["events"] == "strikeout_double_play").sum())
    bb = int((pa_rows["events"] == "walk").sum())
    hbp = int((pa_rows["events"] == "hit_by_pitch").sum())
    sf = int((pa_rows["events"] == "sac_fly").sum())

    ab_rows = pa_rows[~pa_rows["events"].isin(NON_AB_EVENTS)]
    ab = len(ab_rows)
    tb = singles + 2 * doubles + 3 * triples + 4 * hr

    ba = hits / ab if ab else 0.0
    obp = (hits + bb + hbp) / n_pa if n_pa else 0.0
    slg = tb / ab if ab else 0.0
    woba_num = (
        WOBA_BB * bb + WOBA_HBP * hbp
        + WOBA_1B * singles + WOBA_2B * doubles
        + WOBA_3B * triples + WOBA_HR * hr
    )
    woba_denom = ab + bb + sf + hbp
    woba = woba_num / woba_denom if woba_denom else 0.0

    ip_outs = _outs_recorded(pa_rows)
    # Earned-run proxy: pybaseball doesn't carry "earned" flag, so we score
    # runs charged to this pitcher's PA via post-bases minus pre-bases logic.
    # As a pragmatic fallback, use total runs scored on PAs ending while this
    # pitcher was on the mound (post_bat_score - bat_score on that pitch).
    if {"post_bat_score", "bat_score"}.issubset(pa_rows.columns):
        runs = (pa_rows["post_bat_score"] - pa_rows["bat_score"]).clip(lower=0).sum()
        er_proxy = int(runs)
    else:
        er_proxy = 0

    bbe = df[df["type"] == "X"].dropna(subset=["launch_speed"])
    n_bbe = len(bbe)
    avg_ev = float(bbe["launch_speed"].mean()) if n_bbe else 0.0
    max_ev = float(bbe["launch_speed"].max()) if n_bbe else 0.0
    hard_hit_pct = float((bbe["launch_speed"] >= 95).mean()) if n_bbe else 0.0
    barrel_pct = float((bbe["launch_speed_angle"] == 6).mean()) if n_bbe else 0.0

    bbe_typed = df[df["type"] == "X"].dropna(subset=["bb_type"])
    gb_pct = float((bbe_typed["bb_type"] == "ground_ball").mean()) if len(bbe_typed) else 0.0

    swings = df["description"].isin(SWING_DESCRIPTIONS)
    whiffs = df["description"].isin(WHIFF_DESCRIPTIONS)
    out_of_zone = df["zone"].isin([11, 12, 13, 14])

    n_swings = int(swings.sum())
    whiff_pct = whiffs.sum() / n_swings if n_swings else 0.0
    chase_pct = (
        (swings & out_of_zone).sum() / out_of_zone.sum()
        if out_of_zone.sum() else 0.0
    )

    k_pct = so / n_pa if n_pa else 0.0
    bb_pct = bb / n_pa if n_pa else 0.0

    bbe_x = bbe.dropna(subset=["estimated_ba_using_speedangle"])
    sum_xba = bbe_x["estimated_ba_using_speedangle"].sum()
    sum_xslg = bbe_x["estimated_slg_using_speedangle"].sum()
    sum_xwoba_contact = bbe_x["estimated_woba_using_speedangle"].sum()

    xba_denom = len(bbe_x) + so
    xba = sum_xba / xba_denom if xba_denom else 0.0
    xslg = sum_xslg / xba_denom if xba_denom else 0.0

    xwoba_num = sum_xwoba_contact + WOBA_BB * bb + WOBA_HBP * hbp
    xwoba = xwoba_num / n_pa if n_pa else 0.0

    # Rough xERA proxy: scale xwOBA by the league ERA/xwOBA ratio
    # (~4.20 / 0.315 ~ 13.33). Savant's official xERA uses a more complex
    # model; this is a directional approximation.
    xera = xwoba * (4.20 / 0.315)

    pitching_run_value = (
        -float(df["delta_run_exp"].sum()) if "delta_run_exp" in df.columns else 0.0
    )

    fb = df[df["pitch_name"].isin(FASTBALL_NAMES)] if "pitch_name" in df.columns else df.iloc[0:0]
    cb = df[df["pitch_name"].isin(CURVE_NAMES)] if "pitch_name" in df.columns else df.iloc[0:0]

    def _mean_or_none(s: pd.Series) -> float | None:
        s = s.dropna()
        return float(s.mean()) if len(s) else None

    fb_velo = _mean_or_none(fb["release_speed"]) if "release_speed" in fb.columns else None
    fb_spin = _mean_or_none(fb["release_spin_rate"]) if "release_spin_rate" in fb.columns else None
    cb_spin = _mean_or_none(cb["release_spin_rate"]) if "release_spin_rate" in cb.columns else None
    extension = (
        _mean_or_none(df["release_extension"]) if "release_extension" in df.columns else None
    )
    arm_angle = (
        _mean_or_none(df["arm_angle"]) if "arm_angle" in df.columns else None
    )

    return Stats(
        pitches=n_pitches, pa=int(n_pa), ab=int(ab), hits=int(hits),
        hr=int(hr), bb=int(bb), hbp=int(hbp), so=int(so),
        ip_outs=int(ip_outs), er_proxy=int(er_proxy),
        ba=ba, obp=obp, slg=slg, woba=woba,
        bbe=int(n_bbe),
        avg_ev_allowed=avg_ev, max_ev_allowed=max_ev,
        hard_hit_pct=hard_hit_pct, barrel_pct=barrel_pct, gb_pct=gb_pct,
        chase_pct=float(chase_pct), whiff_pct=float(whiff_pct),
        k_pct=float(k_pct), bb_pct=float(bb_pct),
        xba=float(xba), xslg=float(xslg), xwoba=float(xwoba), xera=float(xera),
        pitching_run_value=pitching_run_value,
        fb_velo=fb_velo, fb_spin=fb_spin, cb_spin=cb_spin,
        extension=extension, arm_angle=arm_angle,
    )


def _subset_stats(group: pd.DataFrame, total_pitches: int) -> dict:
    """Per-pitch-name stats row from the pitcher's POV."""
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
    so += int((pa_rows["events"] == "strikeout_double_play").sum())

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
    ks_2s += int((two_strike["events"] == "strikeout_double_play").sum())
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
    hard_hit_pct = (
        (bbe_all["launch_speed"] >= 95).sum() / n_bbe * 100 if n_bbe else 0.0
    )

    velo = group["release_speed"].dropna()
    spin = group["release_spin_rate"].dropna() if "release_spin_rate" in group.columns else pd.Series(dtype=float)
    ivb = group["pfx_z"].dropna() * 12.0 if "pfx_z" in group.columns else pd.Series(dtype=float)
    hb = group["pfx_x"].dropna() * 12.0 if "pfx_x" in group.columns else pd.Series(dtype=float)
    extension = (
        group["release_extension"].dropna() if "release_extension" in group.columns else pd.Series(dtype=float)
    )

    rv_offense = float(group["delta_run_exp"].sum()) if "delta_run_exp" in group.columns else 0.0
    rv = -rv_offense                                  # pitcher POV: positive = good
    rv_per_100 = (rv / n_pitches * 100) if n_pitches else 0.0

    return {
        "Pitches": n_pitches,
        "%": n_pitches / total_pitches * 100 if total_pitches else 0.0,
        "Velo": float(velo.mean()) if len(velo) else float("nan"),
        "Max Velo": float(velo.max()) if len(velo) else float("nan"),
        "Spin": float(spin.mean()) if len(spin) else float("nan"),
        "IVB (in)": float(ivb.mean()) if len(ivb) else float("nan"),
        "HB (in)": float(hb.mean()) if len(hb) else float("nan"),
        "Ext (ft)": float(extension.mean()) if len(extension) else float("nan"),
        "PA": n_pa, "AB": ab, "H": hits,
        "1B": singles, "2B": doubles, "3B": triples, "HR": hr, "SO": so,
        "BBE": n_bbe,
        "BA": ba, "SLG": slg, "wOBA": woba,
        "xBA": xba, "xSLG": xslg, "xwOBA": xwoba,
        "EV": avg_ev,
        "Whiff %": whiff_pct, "K %": k_pct, "PutAway %": putaway,
        "Hard Hit %": hard_hit_pct,
        "RV/100": rv_per_100, "RV": rv,
    }


def pitch_type_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Per-pitch-name arsenal table (Savant's 'Pitch Arsenal' + 'Run Values by Pitch Type')."""
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
    """Three-group (Fastball / Breaking / Offspeed) summary, pitcher POV."""
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


def hardest_hit_allowed_table(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    bbe = df[df["type"] == "X"].dropna(subset=["launch_speed"]).copy()
    if bbe.empty:
        return pd.DataFrame()
    cols = [
        "game_date", "events", "pitch_name",
        "launch_speed", "launch_angle", "hit_distance_sc",
    ]
    cols = [c for c in cols if c in bbe.columns]
    out = bbe.sort_values("launch_speed", ascending=False).head(n)[cols]
    return out.rename(columns={
        "game_date": "Date", "events": "Result", "pitch_name": "Pitch",
        "launch_speed": "EV (mph)", "launch_angle": "LA (deg)",
        "hit_distance_sc": "Distance (ft)",
    })


# ---------- markdown rendering --------------------------------------------

def fmt_pct(x: float) -> str: return f"{x * 100:.1f}%"
def fmt3(x: float) -> str: return f"{x:.3f}"
def fmt1(x: float) -> str: return f"{x:.1f}"


def _format_pitch_table(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    pct_cols = ["%", "Whiff %", "K %", "PutAway %", "Hard Hit %"]
    rate_cols = ["BA", "SLG", "wOBA", "xBA", "xSLG", "xwOBA"]
    one_dec_cols = ["Velo", "Max Velo", "IVB (in)", "HB (in)", "Ext (ft)", "EV"]
    int_cols = ["Spin"]
    signed_cols = ["RV/100", "RV"]
    for col in pct_cols:
        if col in out.columns:
            out[col] = out[col].map(lambda v: f"{v:.1f}")
    for col in rate_cols:
        if col in out.columns:
            out[col] = out[col].map(lambda v: f"{v:.3f}")
    for col in one_dec_cols:
        if col in out.columns:
            out[col] = out[col].map(lambda v: "" if pd.isna(v) else f"{v:.1f}")
    for col in int_cols:
        if col in out.columns:
            out[col] = out[col].map(lambda v: "" if pd.isna(v) else f"{v:.0f}")
    for col in signed_cols:
        if col in out.columns:
            out[col] = out[col].map(lambda v: f"{v:+.1f}")
    return out


def _ip_str(outs: int) -> str:
    whole, third = divmod(outs, 3)
    return f"{whole}.{third}"


def to_markdown(name: str, mlbam_id: int, start: str, end: str,
                s: Stats,
                pitch_table: pd.DataFrame,
                tracking_table: pd.DataFrame,
                hh_table: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append(f"# {name} — Statcast summary (pitcher)")
    lines.append("")
    lines.append(
        f"_MLBAM id `{mlbam_id}` · window `{start}` to `{end}` · "
        f"source: Baseball Savant via pybaseball_"
    )
    lines.append("")

    ip = _ip_str(s.ip_outs)
    era_proxy = (s.er_proxy * 9 * 3 / s.ip_outs) if s.ip_outs else 0.0

    lines.append("## Line")
    lines.append("")
    lines.append("| IP | TBF | H | HR | BB | HBP | K | K% | BB% | RA9* | Opp AVG | Opp SLG | Opp wOBA |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    lines.append(
        f"| {ip} | {s.pa} | {s.hits} | {s.hr} | {s.bb} | {s.hbp} | {s.so} | "
        f"{fmt_pct(s.k_pct)} | {fmt_pct(s.bb_pct)} | {fmt1(era_proxy)} | "
        f"{fmt3(s.ba)} | {fmt3(s.slg)} | {fmt3(s.woba)} |"
    )
    lines.append("")
    lines.append(
        "_*RA9 is a runs-allowed-per-9 proxy from `post_bat_score - bat_score`; "
        "it doesn't separate earned/unearned runs._"
    )
    lines.append("")

    lines.append("## Value")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(
        f"| Pitching Run Value (-sum of run-expectancy delta) | "
        f"{s.pitching_run_value:+.1f} |"
    )
    lines.append("")

    fb_velo = f"{s.fb_velo:.1f} mph" if s.fb_velo is not None else "n/a"
    fb_spin = f"{s.fb_spin:.0f} rpm" if s.fb_spin is not None else "n/a"
    cb_spin = f"{s.cb_spin:.0f} rpm" if s.cb_spin is not None else "n/a"
    ext = f"{s.extension:.1f} ft" if s.extension is not None else "n/a"
    arm = f"{s.arm_angle:.1f}°" if s.arm_angle is not None else "n/a"

    lines.append("## Pitching (matches Savant percentile card)")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| xERA (rough xwOBA-based proxy) | {fmt1(s.xera)} |")
    lines.append(f"| xwOBA              | {fmt3(s.xwoba)} |")
    lines.append(f"| xBA                | {fmt3(s.xba)} |")
    lines.append(f"| xSLG               | {fmt3(s.xslg)} |")
    lines.append(f"| Avg Exit Velocity allowed | {fmt1(s.avg_ev_allowed)} mph |")
    lines.append(f"| Max Exit Velocity allowed | {fmt1(s.max_ev_allowed)} mph |")
    lines.append(f"| Barrel % allowed   | {fmt_pct(s.barrel_pct)} |")
    lines.append(f"| Hard-Hit % allowed | {fmt_pct(s.hard_hit_pct)} |")
    lines.append(f"| Ground-Ball %      | {fmt_pct(s.gb_pct)} |")
    lines.append(f"| Chase %            | {fmt_pct(s.chase_pct)} |")
    lines.append(f"| Whiff %            | {fmt_pct(s.whiff_pct)} |")
    lines.append(f"| K %                | {fmt_pct(s.k_pct)} |")
    lines.append(f"| BB %               | {fmt_pct(s.bb_pct)} |")
    lines.append(f"| Fastball velocity  | {fb_velo} |")
    lines.append(f"| Fastball spin rate | {fb_spin} |")
    lines.append(f"| Curve spin rate    | {cb_spin} |")
    lines.append(f"| Extension          | {ext} |")
    lines.append(f"| Arm angle          | {arm} |")
    lines.append("")

    lines.append("## Pitch tracking (Fastball / Breaking / Offspeed)")
    lines.append("")
    if tracking_table.empty:
        lines.append("_No classified pitches in this window._")
    else:
        tracking_cols = [
            "Pitch Type", "Pitches", "%", "Velo", "Spin",
            "IVB (in)", "HB (in)", "Ext (ft)",
            "PA", "BBE", "BA", "xBA", "SLG", "xSLG", "wOBA", "xwOBA",
            "EV", "Whiff %", "PutAway %", "Hard Hit %",
        ]
        cols = [c for c in tracking_cols if c in tracking_table.columns]
        lines.append(_format_pitch_table(tracking_table[cols]).to_markdown(index=False))
    lines.append("")

    lines.append("## Pitch arsenal & run values")
    lines.append("")
    if pitch_table.empty:
        lines.append("_No pitches in this window._")
    else:
        detail_cols = [
            "Pitch", "Pitches", "%", "Velo", "Max Velo", "Spin",
            "IVB (in)", "HB (in)", "Ext (ft)",
            "PA", "BA", "SLG", "wOBA", "xBA", "xSLG", "xwOBA",
            "Whiff %", "K %", "PutAway %", "Hard Hit %",
            "RV/100", "RV",
        ]
        cols = [c for c in detail_cols if c in pitch_table.columns]
        lines.append(_format_pitch_table(pitch_table[cols]).to_markdown(index=False))
    lines.append("")

    lines.append("## Hardest-hit balls allowed")
    lines.append("")
    if hh_table.empty:
        lines.append("_No batted balls in this window._")
    else:
        lines.append(hh_table.to_markdown(index=False))
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- Pitching Run Value is computed as the negative sum of "
        "`delta_run_exp` over every pitch thrown — positive numbers mean "
        "the pitcher saved runs relative to a league-average outcome."
    )
    lines.append(
        "- `xERA` here is a rough linear map from xwOBA to ERA, not Savant's "
        "official model. Treat it as a directional figure."
    )
    lines.append(
        "- Per-player percentiles require a league reference distribution. "
        "To add them, pull `pybaseball.statcast_pitcher_expected_stats(year)` "
        "and sibling leaderboard endpoints and rank in pandas."
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
    hh_table = hardest_hit_allowed_table(df)

    md = to_markdown(
        PLAYER_NAME, pid, SEASON_START, SEASON_END,
        s, pitch_table, tracking_table, hh_table,
    )
    out_path = (
        Path(__file__).parent
        / f"{PLAYER_LAST}_{PLAYER_FIRST}_{SEASON}_pitcher_summary.md"
    )
    out_path.write_text(md, encoding="utf-8")

    print()
    print(md)
    print()
    print(f"Markdown summary written to {out_path}")


if __name__ == "__main__":
    main()
