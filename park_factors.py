"""Park factor loader and xwOBA-equivalent blender.

Three data sources, all BallparkPal:

1. **Daily player** xlsx at ``parkdata/PlayerParkFactors_<date>*.xlsx`` --
   per-player HR / 2B-3B / 1B factors at the venue they're playing in
   tonight.  Already bakes in handedness, batted-ball profile, and spray
   tendency.  When available, this is the preferred per-batter source.
2. **Daily game** CSVs at ``parkdata/ParkFactors_<date>*.csv`` -- one row
   per game with combined park + today's-weather factors at the team-blended
   level.  Fallback when a batter isn't in the player file, and also used
   for the lineup-level banner context.
3. **Annual** JSON at ``parkdata/annual_park_factors_2023_2025.json`` --
   per-team multi-year baseline (BPP "Model" tab), used to back out the
   schedule-mix bias already embedded in each team's season-to-date stats.

The runtime call site combines them as::

    effective_pf = todays_full_pf / batter_input_bias / pitcher_input_bias

where ``todays_full_pf`` comes from the daily CSV at FULL strength (no
heuristic 0.5 dampener) and ``input_bias[team] = 1 + (annual_pf - 1) * 0.5``
reflects the ~50% home / ~50% road schedule mix in the player inputs.

Result is then applied as a multiplier on ``proj_xwoba``::

    proj_xwoba_final = (proj_xwoba_bbtype + count_shift) * effective_pf

Limitations:
- No handedness split (CSV is one number per game, not L/R separated).
- No K% / BB% / hard-hit% park adjustment (deferred; BPP exposes annual
  K% and BB% factors but daily isn't broken out, and we keep scope tight).
- 2B and 3B share one PF (daily CSV lumps them).
"""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from datetime import date
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).parent
PARKDATA_DIR = ROOT / "parkdata"
ANNUAL_PF_JSON = PARKDATA_DIR / "annual_park_factors_2023_2025.json"

# wOBA values (FanGraphs 2025 guts; matches batter.py / matchup.py).
_WOBA_1B = 0.882
_WOBA_2B = 1.254
_WOBA_3B = 1.583
_WOBA_HR = 2.027

# League-baseline outcome shares per PA (rough, used only for the blend
# weights -- doesn't have to track the live LG_OUTCOMES exactly).
_SHARE_1B = 0.145
_SHARE_2B = 0.045
_SHARE_3B = 0.004
_SHARE_HR = 0.033

# Pre-computed contribution weights of each outcome to xwOBA:
_CONTRIB_1B = _WOBA_1B * _SHARE_1B   # 0.128
_CONTRIB_2B = _WOBA_2B * _SHARE_2B   # 0.056
_CONTRIB_3B = _WOBA_3B * _SHARE_3B   # 0.006
_CONTRIB_HR = _WOBA_HR * _SHARE_HR   # 0.067
_CONTRIB_HITS_SUM = _CONTRIB_1B + _CONTRIB_2B + _CONTRIB_3B + _CONTRIB_HR

# Fraction of league xwOBA that comes from hits (the rest is BB + HBP,
# which park factors don't really move).
_LG_XWOBA = 0.315
_HITS_FRAC = _CONTRIB_HITS_SUM / _LG_XWOBA   # ~0.816

# Half-strength constant -- used both by the deprecated
# ``effective_xwoba_pf`` shim AND by the new ``input_bias`` formula
# (50% of schedule is home, 50% road, so the input is half-influenced
# by the team's home park).
_STRENGTH = 0.5


def _parse_pct(text: str) -> float:
    """Convert ``"+23%"`` / ``"-5%"`` / ``"0%"`` to a multiplier (1.23, 0.95, 1.00).

    Returns 1.0 (neutral) if the text is empty or unparseable.
    """
    if text is None:
        return 1.0
    s = str(text).strip()
    if not s:
        return 1.0
    m = re.match(r"^([+-]?)(\d+(?:\.\d+)?)\s*%?$", s)
    if not m:
        return 1.0
    sign = -1.0 if m.group(1) == "-" else 1.0
    try:
        pct = float(m.group(2))
    except ValueError:
        return 1.0
    return 1.0 + sign * pct / 100.0


# MLB.com / Statcast lineup codes don't always match BPP's BBRef-flavored
# codes.  Map both sides onto a single canonical token so lookups work.
# Add entries here when a mismatch surfaces (e.g. a city moves, a code
# changes mid-season).
_TEAM_ALIASES = {
    "AZ": "ARI",     # Statcast uses AZ; BPP uses ARI
    "WSH": "WAS",    # Statcast/MLB.com uses WSH; BPP uses WAS
    "CWS": "CHW",    # Statcast uses CWS; BPP uses CHW
    "OAK": "ATH",    # team rebrand mid-2025 -- both codes still appear
}


def _norm_team(code: str) -> str:
    """Strip whitespace, uppercase, and canonicalize a team code.

    The BPP CSV has inconsistent padding around team codes
    (e.g. ``"SF  @ ATH"``) AND uses BBRef-flavored codes for a handful
    of clubs (``ARI`` not ``AZ``, ``WAS`` not ``WSH``, ``CHW`` not ``CWS``)
    where the MLB.com lineup feed disagrees.  Canonicalize before keying
    so a single lookup table works for both sources.
    """
    raw = (code or "").strip().upper()
    return _TEAM_ALIASES.get(raw, raw)


def _split_game_cell(cell: str) -> tuple[str, str] | None:
    """Parse ``"ARI @ COL"`` (with possible extra spaces) into ``("ARI", "COL")``.

    Returns ``None`` if the cell doesn't look like a matchup.
    """
    if not cell or "@" not in cell:
        return None
    parts = cell.split("@", 1)
    away = _norm_team(parts[0])
    home = _norm_team(parts[1])
    if not away or not home:
        return None
    return away, home


def _find_csv_for_date(report_date: date) -> Path | None:
    """Return the first ``parkdata/ParkFactors_<YYYY-MM-DD>*.csv`` for a date.

    BPP files are named like ``ParkFactors_2026-05-16 00_00_00.csv`` so we
    glob with a date prefix and pick the lexicographically-first match.
    """
    if not PARKDATA_DIR.exists():
        return None
    pattern = f"ParkFactors_{report_date.isoformat()}*.csv"
    matches = sorted(PARKDATA_DIR.glob(pattern))
    return matches[0] if matches else None


def load_park_factors(report_date: date) -> dict[tuple[str, str], dict]:
    """Load the daily park factor CSV for ``report_date``.

    Returns a dict keyed by ``(away_code, home_code)`` with values:

        {"pf_HR": float, "pf_2B3B": float, "pf_1B": float,
         "venue": str, "raw_xwoba_pf": float, "effective_xwoba_pf": float}

    Empty dict if the file is missing or unparseable -- callers should
    treat missing keys as a neutral 1.0 multiplier.
    """
    path = _find_csv_for_date(report_date)
    if path is None:
        return {}

    out: dict[tuple[str, str], dict] = {}
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                game_key = _split_game_cell(row.get("Game", ""))
                if game_key is None:
                    continue
                pf_hr = _parse_pct(row.get("HR %", ""))
                pf_xb = _parse_pct(row.get("2B/3B %", ""))
                pf_1b = _parse_pct(row.get("1B %", ""))
                raw = xwoba_pf(pf_hr, pf_xb, pf_1b)
                out[game_key] = {
                    "pf_HR": pf_hr,
                    "pf_2B3B": pf_xb,
                    "pf_1B": pf_1b,
                    "venue": (row.get("Venue") or "").strip(),
                    "raw_xwoba_pf": raw,
                    "effective_xwoba_pf": effective_xwoba_pf(raw),
                }
    except (OSError, csv.Error):
        return {}

    return out


def xwoba_pf(pf_hr: float, pf_2b3b: float, pf_1b: float) -> float:
    """Blend per-outcome park factors into a single xwOBA multiplier.

    Uses the wOBA-value-weighted contribution of each outcome to xwOBA:

        delta_hits_xwoba = sum_o (contrib_o * (pf_o - 1))
                         / sum_o contrib_o
        pf_xwoba = 1 + HITS_FRAC * delta_hits_xwoba

    HITS_FRAC ~ 0.816 (fraction of league xwOBA coming from hits; BB/HBP
    are unaffected by park).  Result is the FULL-strength park effect;
    apply ``effective_xwoba_pf`` for the half-strength heuristic.
    """
    if _CONTRIB_HITS_SUM <= 0:
        return 1.0
    delta_hits = (
        _CONTRIB_1B * (pf_1b - 1.0)
        + _CONTRIB_2B * (pf_2b3b - 1.0)
        + _CONTRIB_3B * (pf_2b3b - 1.0)
        + _CONTRIB_HR * (pf_hr - 1.0)
    ) / _CONTRIB_HITS_SUM
    return 1.0 + _HITS_FRAC * delta_hits


def effective_xwoba_pf(raw_pf: float) -> float:
    """DEPRECATED -- legacy half-strength heuristic on the daily PF.

    Kept only so older code paths (and historical sidecars built with this
    formula) keep evaluating consistently.  The runtime pipeline now uses
    :func:`effective_xwoba_pf_deparked`, which is mathematically cleaner
    because it backs out each team's annual schedule-mix bias before
    applying today's PF at full strength.
    """
    return 1.0 + (raw_pf - 1.0) * _STRENGTH


def lookup(park_factors: dict[tuple[str, str], dict],
           away: str, home: str) -> dict | None:
    """Look up a matchup, normalizing team codes for whitespace/case.

    Returns ``None`` on miss; caller should treat that as neutral.
    """
    return park_factors.get((_norm_team(away), _norm_team(home)))


# ---- Annual de-parking ---------------------------------------------------


@lru_cache(maxsize=1)
def load_annual_park_factors() -> dict[str, dict]:
    """Load the BPP annual archive and precompute per-team xwOBA PF / bias.

    Reads ``parkdata/annual_park_factors_2023_2025.json`` (transcribed
    BallparkPal multi-year archive) and returns a dict keyed by canonical
    team code (post-alias):

        {
          "COL": {
            "stadium": "Coors Field",
            "annual_xwoba_pf": 1.131,     # full-strength annual
            "input_bias":      1.065,     # 1 + (annual - 1) * 0.5
            "model": {"runs": 132, "hits": 114, "hr": 116, ...}
          },
          ...
        }

    Cached with ``lru_cache`` so the JSON parse + blend happens once per
    process.  Returns an empty dict if the file is missing or unparseable
    -- the rest of the pipeline treats that as "no de-parking available"
    and falls back to neutral.
    """
    if not ANNUAL_PF_JSON.exists():
        return {}
    try:
        with ANNUAL_PF_JSON.open("r", encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}

    teams_raw = doc.get("teams") or {}
    out: dict[str, dict] = {}
    for code, entry in teams_raw.items():
        model = entry.get("model") or {}
        # Annual factors are 100-baseline -- convert to multipliers.
        pf_hr   = float(model.get("hr",  100)) / 100.0
        pf_xbh  = float(model.get("xbh", 100)) / 100.0
        pf_1b   = float(model.get("1b",  100)) / 100.0
        annual  = xwoba_pf(pf_hr, pf_xbh, pf_1b)
        out[_norm_team(code)] = {
            "stadium": entry.get("stadium", ""),
            "annual_xwoba_pf": annual,
            "input_bias":      1.0 + (annual - 1.0) * _STRENGTH,
            "model":           model,
        }
    return out


def effective_xwoba_pf_deparked(todays_full_pf: float,
                                 batter_team: str,
                                 pitcher_team: str,
                                 annual_table: dict[str, dict]) -> dict:
    """Combine today's daily PF with batter/pitcher de-parking factors.

    Returns a dict carrying both the final multiplier AND its component
    breakdown, so callers can render a transparent footer in HTML reports::

        {
          "effective":     0.950,
          "todays_full":   0.940,
          "batter_bias":   1.020,
          "pitcher_bias":  0.970,
          "batter_team":   "NYY",
          "pitcher_team":  "SD"
        }

    Math::

        effective = todays_full_pf / batter_input_bias / pitcher_input_bias

    Rationale: the batter's season-to-date xwOBA input is biased by ~50%
    of their home-park PF (since half their PAs were at home).  Same for
    the pitcher input.  Dividing those out before re-parking by today's
    venue is the proper way to combine the two sources without double-
    counting -- replaces the older heuristic 0.5 dampener.

    Falls back to identity (``effective = 1.0`` and both biases = 1.0) if
    either team is missing from the annual table or the table is empty.
    """
    batter_key  = _norm_team(batter_team)
    pitcher_key = _norm_team(pitcher_team)

    bat_entry = annual_table.get(batter_key) if annual_table else None
    pit_entry = annual_table.get(pitcher_key) if annual_table else None

    batter_bias  = float(bat_entry["input_bias"]) if bat_entry else 1.0
    pitcher_bias = float(pit_entry["input_bias"]) if pit_entry else 1.0

    if batter_bias <= 0 or pitcher_bias <= 0:
        return {"effective": 1.0, "todays_full": float(todays_full_pf),
                "batter_bias": 1.0, "pitcher_bias": 1.0,
                "batter_team": batter_key, "pitcher_team": pitcher_key}

    effective = float(todays_full_pf) / batter_bias / pitcher_bias
    return {"effective":    effective,
            "todays_full":  float(todays_full_pf),
            "batter_bias":  batter_bias,
            "pitcher_bias": pitcher_bias,
            "batter_team":  batter_key,
            "pitcher_team": pitcher_key}


# ---- Daily PER-PLAYER park factors (BPP "Todays Park Factors" xlsx) -----


def _normalize_player_name(name: str) -> str:
    """Lowercase, strip accents/punctuation, collapse whitespace.

    BPP uses standard MLB.com-style names ("Jose Ramirez", "Aaron Judge")
    but a handful are accent-decorated.  Our lineup feed sometimes uses
    decorated forms.  Normalizing both sides to ASCII-lower-stripped lets
    a single key resolve both spellings.
    """
    if not name:
        return ""
    decomposed = unicodedata.normalize("NFKD", str(name))
    ascii_only = decomposed.encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^a-zA-Z0-9 ]+", " ", ascii_only)
    return " ".join(cleaned.lower().split())


def _find_player_pf_xlsx(report_date: date) -> Path | None:
    """Return the first ``parkdata/PlayerParkFactors_<YYYY-MM-DD>*.xlsx``
    for a date, or ``None`` if not present.
    """
    if not PARKDATA_DIR.exists():
        return None
    pattern = f"PlayerParkFactors_{report_date.isoformat()}*.xlsx"
    matches = sorted(PARKDATA_DIR.glob(pattern))
    return matches[0] if matches else None


def load_player_park_factors(report_date: date) -> dict[tuple[str, str], dict]:
    """Load the BPP daily per-player park factor xlsx for ``report_date``.

    Returns a dict keyed by ``(team_canonical, normalized_name)`` with
    values:

        {
          "pf_HR":   float,
          "pf_2B3B": float,
          "pf_1B":   float,
          "park":    str,   # venue display name
          "name":    str,   # raw player name from BPP
          "team":    str    # canonical team code
        }

    Empty dict if the file is missing or unparseable.  Callers fall back
    to the game-level PF (existing daily CSV) for any batter not found
    here.

    Why both ``team`` and ``name`` in the key?  BPP names like "Jose
    Fernandez" are not unique across teams; team disambiguates.
    """
    path = _find_player_pf_xlsx(report_date)
    if path is None:
        return {}

    try:
        import pandas as pd  # lazy import to keep park_factors import-cheap
    except ImportError:
        return {}

    try:
        # Row 0 of the xlsx is the BPP banner; real headers are on row 1.
        df = pd.read_excel(path, header=1)
    except (OSError, ValueError, ImportError):
        return {}

    out: dict[tuple[str, str], dict] = {}
    needed = {"Tm", "Player", "Park", "HR", "2B/3B", "1B"}
    if not needed.issubset(df.columns):
        return {}

    for _, row in df.iterrows():
        team_raw = row.get("Tm")
        name_raw = row.get("Player")
        if team_raw is None or name_raw is None:
            continue
        team = _norm_team(team_raw)
        name = _normalize_player_name(name_raw)
        if not team or not name:
            continue
        try:
            pf_hr = float(row.get("HR"))
            pf_xb = float(row.get("2B/3B"))
            pf_1b = float(row.get("1B"))
        except (TypeError, ValueError):
            continue
        out[(team, name)] = {
            "pf_HR":   pf_hr,
            "pf_2B3B": pf_xb,
            "pf_1B":   pf_1b,
            "park":    str(row.get("Park") or "").strip(),
            "name":    str(name_raw).strip(),
            "team":    team,
        }
    return out


def lookup_player(player_pf_table: dict[tuple[str, str], dict],
                  team: str, name: str) -> dict | None:
    """Look up a batter by ``(team, name)``, normalizing both sides.

    Returns ``None`` if the player isn't in today's BPP file -- caller
    should fall back to the game-level PF for that batter.
    """
    if not player_pf_table:
        return None
    return player_pf_table.get((_norm_team(team), _normalize_player_name(name)))
