"""Fetch confirmed lineups for past dates from MLB Stats API boxscores.

The live ``fetch_lineups.py`` scrapes the MLB.com starting-lineups page
which only ever shows today's slate.  For the calibration backfill we
need confirmed lineups for completed games -- those live in the Stats
API boxscore endpoint:

    1. GET ``schedule?date=D`` -> games on date D + team abbreviations
    2. GET ``game/<gamePk>/boxscore`` -> ``battingOrder`` arrays + the
       starting pitcher (first entry in ``pitchers``) for each side

We then write a matchups CSV in the same format the rest of the pipeline
consumes::

    MATCHUP_KEY, Batter Name, Pitcher Name, batting_order, status,
    hitter_team, batter_id, pitcher_id

so ``matchup.py --batch <csv>`` can produce a baseline slate.json and
``postgame.py --date D`` can grade it.

Output filenames are timestamped ``matchups_<D>_backfill.csv`` and land
in ``matchups/`` by default (use ``--out-dir`` to redirect).
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import date as dateclass
from pathlib import Path
from typing import Any

import requests

from fetch_lineups import MATCHUPS_DIR, STATSAPI

# Stats API status states that mean the game was actually played (and
# therefore has a real boxscore lineup).  Postponed / cancelled games
# either have no boxscore or one with empty battingOrder arrays.
_PLAYED_STATES = {"Final", "Game Over", "Completed Early"}


def _get_json(url: str, params: dict | None = None,
              retries: int = 3, sleep: float = 1.5) -> dict | None:
    """GET ``url`` and decode JSON, retrying transient errors."""
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(sleep * (attempt + 1))
    print(f"[historical] HTTP error after {retries} tries on {url}: {last_err}",
          file=sys.stderr)
    return None


def fetch_games_for_date(d: dateclass) -> list[dict]:
    """Return a list of completed games on date ``d``.

    Each dict has::

        gamePk, status, away_abbrev, home_abbrev,
        away_pitcher_id, away_pitcher_name,
        home_pitcher_id, home_pitcher_name,
        away_lineup [{name, id, spot}], home_lineup [...]
    """
    data = _get_json(
        f"{STATSAPI}/schedule",
        params={"sportId": 1, "date": d.isoformat(), "hydrate": "team"},
    )
    if not data:
        return []

    games: list[dict] = []
    for date_entry in data.get("dates", []):
        for g in date_entry.get("games", []):
            status = g.get("status", {}).get("detailedState", "")
            if status not in _PLAYED_STATES:
                continue  # skip postponed / suspended / scheduled
            away = g["teams"]["away"]["team"]
            home = g["teams"]["home"]["team"]
            games.append({
                "gamePk": g.get("gamePk"),
                "status": status,
                "away_abbrev": away.get("abbreviation", ""),
                "home_abbrev": home.get("abbreviation", ""),
                # Pitcher ids/names will be filled in from the boxscore below.
                "away_pitcher_id": None,
                "away_pitcher_name": "",
                "home_pitcher_id": None,
                "home_pitcher_name": "",
                "away_lineup": [],
                "home_lineup": [],
            })
    return games


def enrich_with_boxscore(game: dict) -> bool:
    """Fill in ``game`` with batting order + starting pitcher from boxscore.

    Returns True if both sides have a non-empty 9-deep order.  We skip
    games with truncated orders (rain-shortened bullpen games, etc.).
    """
    pk = game.get("gamePk")
    if not pk:
        return False
    data = _get_json(f"{STATSAPI}/game/{pk}/boxscore")
    if not data:
        return False

    teams = data.get("teams", {})
    ok = True
    for side in ("away", "home"):
        team_data = teams.get(side, {})
        order_ids = team_data.get("battingOrder", []) or []
        players = team_data.get("players", {}) or {}
        pitcher_ids = team_data.get("pitchers", []) or []

        # Starting pitcher = first entry in pitchers (in order of appearance).
        sp_id = pitcher_ids[0] if pitcher_ids else None
        sp_name = ""
        if sp_id is not None:
            sp_player = players.get(f"ID{sp_id}", {})
            sp_name = sp_player.get("person", {}).get("fullName", "")
        game[f"{side}_pitcher_id"] = sp_id
        game[f"{side}_pitcher_name"] = sp_name

        lineup: list[dict] = []
        for spot_idx, pid in enumerate(order_ids[:9], start=1):
            pid_int = int(pid) if str(pid).isdigit() else pid
            key = f"ID{pid}"
            player = players.get(key, {})
            name = player.get("person", {}).get("fullName", "")
            lineup.append({"name": name, "id": pid_int, "spot": spot_idx})
        game[f"{side}_lineup"] = lineup
        if len(lineup) < 9 or not sp_id:
            ok = False
    return ok


def build_matchups_rows(games: list[dict]) -> list[tuple]:
    """Convert enriched games into matchups CSV rows.

    Schema (matches ``fetch_lineups.generate_csv_rows``):
        matchup_key, hitter_name, pitcher_name, batting_order, status,
        hitter_team, hitter_mlbam_id, pitcher_mlbam_id

    Backfill rows always carry status="confirmed" because they come
    straight from the played-game boxscore.
    """
    rows: list[tuple] = []
    for g in games:
        away = g["away_abbrev"]
        home = g["home_abbrev"]
        if not away or not home:
            continue
        matchup_key = f"{away}@{home}"

        for h in g.get("away_lineup", []):
            rows.append((
                matchup_key, h["name"], g["home_pitcher_name"],
                str(h["spot"]), "confirmed", away,
                str(h["id"]) if h["id"] else "",
                str(g["home_pitcher_id"]) if g["home_pitcher_id"] else "",
            ))
        for h in g.get("home_lineup", []):
            rows.append((
                matchup_key, h["name"], g["away_pitcher_name"],
                str(h["spot"]), "confirmed", home,
                str(h["id"]) if h["id"] else "",
                str(g["away_pitcher_id"]) if g["away_pitcher_id"] else "",
            ))
    return rows


def write_matchups_csv(rows: list[tuple], d: dateclass,
                       out_dir: Path) -> Path:
    """Write ``rows`` to ``out_dir/matchups_<d>_backfill.csv`` and return path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"matchups_{d.isoformat()}_backfill.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        for row in rows:
            w.writerow(row)
    return out_path


def fetch_historical(d: dateclass, out_dir: Path | None = None,
                     verbose: bool = True) -> tuple[Path | None, int, int]:
    """End-to-end: fetch games for date ``d``, enrich, write CSV.

    Returns ``(csv_path_or_None, n_games_kept, n_games_total)``.
    """
    if out_dir is None:
        out_dir = MATCHUPS_DIR
    games = fetch_games_for_date(d)
    total = len(games)
    if not games:
        if verbose:
            print(f"[historical] {d}: no completed games")
        return (None, 0, 0)

    kept: list[dict] = []
    for g in games:
        if enrich_with_boxscore(g):
            kept.append(g)
        elif verbose:
            print(f"[historical] {d}: skipped {g['away_abbrev']}@{g['home_abbrev']} "
                  f"(pk={g['gamePk']}) -- incomplete lineup")

    if not kept:
        if verbose:
            print(f"[historical] {d}: no complete games kept out of {total}")
        return (None, 0, total)

    rows = build_matchups_rows(kept)
    out_path = write_matchups_csv(rows, d, out_dir)
    if verbose:
        print(f"[historical] {d}: wrote {out_path} "
              f"({len(rows)} rows, {len(kept)}/{total} games)")
    return (out_path, len(kept), total)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", required=True, help="YYYY-MM-DD (past date)")
    ap.add_argument("--out-dir", default=str(MATCHUPS_DIR),
                    help=f"output directory (default: {MATCHUPS_DIR})")
    args = ap.parse_args(argv)

    try:
        d = dateclass.fromisoformat(args.date)
    except ValueError:
        print(f"[historical] invalid --date {args.date!r}; expected YYYY-MM-DD",
              file=sys.stderr)
        return 2

    out_path, kept, total = fetch_historical(d, Path(args.out_dir))
    return 0 if out_path else 1


if __name__ == "__main__":
    sys.exit(main())
