#!/usr/bin/env python3
"""
fetch_lineups.py - Generate matchup CSV from MLB starting lineups.

Fetches today's games and starting lineups from MLB.com starting lineups page,
parses team names, pitcher names, and hitter positions,
and writes a timestamped CSV with format:
    <away>@<home>,hitter_name,pitcher_name,lineup_position

Usage:
    python fetch_lineups.py
"""

import argparse
import re
import requests
import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import sys
from bs4 import BeautifulSoup

from log_setup import setup_logging

STARTING_LINEUPS_URL = "https://www.mlb.com/starting-lineups"
STATSAPI = "https://statsapi.mlb.com/api/v1"

_team_id_cache: Optional[Dict[str, int]] = None


def _team_id_map() -> Dict[str, int]:
    """Map team abbreviation (e.g. 'SF') to MLB team id."""
    global _team_id_cache
    if _team_id_cache is None:
        r = requests.get(
            f"{STATSAPI}/teams",
            params={"sportId": 1, "season": datetime.now().year},
            timeout=15,
        )
        r.raise_for_status()
        _team_id_cache = {
            t["abbreviation"]: t["id"] for t in r.json().get("teams", [])
        }
    return _team_id_cache


def _pitcher_hand(name: str) -> Optional[str]:
    """Return 'L' or 'R' for a pitcher's throwing hand, or None."""
    r = requests.get(
        f"{STATSAPI}/people/search",
        params={"names": name, "sportId": 1},
        timeout=15,
    )
    r.raise_for_status()
    for p in r.json().get("people", []):
        if p.get("primaryPosition", {}).get("type") == "Pitcher":
            return (p.get("pitchHand") or {}).get("code")
    return None


def get_projected_lineup(team_abbrev: str, opp_pitcher_name: str) -> List[Tuple[str, int, int]]:
    """Find the team's most recent batting order vs a same-handed starting pitcher.
    Returns list of (name, lineup_position, mlbam_id) or [] if nothing usable."""
    team_id = _team_id_map().get(team_abbrev)
    target_hand = _pitcher_hand(opp_pitcher_name)
    if not team_id or not target_hand:
        return []

    today = datetime.now().date()
    start = (today - timedelta(days=30)).isoformat()
    end = (today - timedelta(days=1)).isoformat()
    r = requests.get(
        f"{STATSAPI}/schedule",
        params={"sportId": 1, "teamId": team_id, "startDate": start, "endDate": end},
        timeout=15,
    )
    r.raise_for_status()

    games: List[Tuple[str, int, dict]] = []
    for d in r.json().get("dates", []):
        for g in d.get("games", []):
            if g.get("status", {}).get("abstractGameState") == "Final":
                games.append((d["date"], g["gamePk"], g["teams"]))
    games.sort(key=lambda x: x[0], reverse=True)

    for _date_str, game_pk, teams in games:
        if teams["away"]["team"]["id"] == team_id:
            our_side, opp_side = "away", "home"
        elif teams["home"]["team"]["id"] == team_id:
            our_side, opp_side = "home", "away"
        else:
            continue

        br = requests.get(f"{STATSAPI}/game/{game_pk}/boxscore", timeout=15)
        if br.status_code != 200:
            continue
        box = br.json()
        opp_pitchers = box["teams"][opp_side].get("pitchers") or []
        if not opp_pitchers:
            continue
        opp_sp_id = opp_pitchers[0]
        opp_player = box["teams"][opp_side]["players"].get(f"ID{opp_sp_id}", {})
        opp_hand = (opp_player.get("person", {}).get("pitchHand") or {}).get("code")
        if not opp_hand:
            pr = requests.get(f"{STATSAPI}/people/{opp_sp_id}", timeout=15)
            if pr.status_code == 200:
                opp_hand = (pr.json()["people"][0].get("pitchHand") or {}).get("code")
        if opp_hand != target_hand:
            continue

        bo = box["teams"][our_side].get("battingOrder") or []
        players = box["teams"][our_side].get("players", {})
        result: List[Tuple[str, int, int]] = []
        for pos, pid in enumerate(bo, 1):
            name = players.get(f"ID{pid}", {}).get("person", {}).get("fullName")
            if name:
                result.append((name, pos, pid))
        if len(result) == 9:
            return result
    return []


def fetch_starting_lineups_page() -> Optional[str]:
    """
    Fetch the MLB starting lineups page.
    
    Returns:
        HTML content or None if fetch fails
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(STARTING_LINEUPS_URL, headers=headers, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"✗ Error fetching starting lineups page: {e}")
        return None


def _extract_mlbam_id(href: str) -> Optional[int]:
    """Extract MLBAM player ID from an MLB.com player link like /player/name-123456."""
    if not href:
        return None
    m = re.search(r'-(\d+)$', href)
    return int(m.group(1)) if m else None


def parse_games(html_content: str) -> List[Dict]:
    """
    Parse games, pitchers, and lineups from HTML.
    
    Returns:
        List of game dictionaries with structure:
        {
            'away_team': str,
            'home_team': str,
            'away_pitcher': str,
            'home_pitcher': str,
            'away_hitters': [(name, position), ...],
            'home_hitters': [(name, position), ...]
        }
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find all pitcher overviews
    pitcher_overviews = soup.find_all('div', class_='starting-lineups__pitcher-overview')
    print(f"  Found {len(pitcher_overviews)} pitcher overview sections")
    
    # Find all team lineup containers
    all_containers = soup.find_all('div', class_='starting-lineups__teams')
    print(f"  Found {len(all_containers)} lineup containers (may include duplicates)")
    
    # Deduplicate containers by team codes
    seen_matchups = {}  # Map of (away_code, home_code) -> container
    unique_containers = []
    
    for container in all_containers:
        header = container.find('div', class_='starting-lineups__teams--header')
        if header:
            away_head = header.find('div', class_='starting-lineups__teams--away-head')
            home_head = header.find('div', class_='starting-lineups__teams--home-head')
            
            if away_head and home_head:
                away_code = away_head.get_text().strip().split()[0]
                home_code = home_head.get_text().strip().split()[0]
                key = (away_code, home_code)
                
                # Only keep the first occurrence of each matchup
                if key not in seen_matchups:
                    seen_matchups[key] = container
                    unique_containers.append((key, container))
    
    print(f"  After deduplication: {len(unique_containers)} unique games\n")
    
    # Parse each unique container and match with pitchers
    games = []
    for game_idx, (matchup_key, container) in enumerate(unique_containers):
        game = {}
        away_code, home_code = matchup_key
        game['away_code'] = away_code
        game['home_code'] = home_code
        
        try:
            # Get away team hitters (always set the key, even if empty)
            game['away_hitters'] = []
            away_list = container.find('ol', class_='starting-lineups__team--away')
            if away_list:
                for pos, li in enumerate(away_list.find_all('li', class_='starting-lineups__player'), 1):
                    name_elem = li.find('a', class_='starting-lineups__player--link')
                    if name_elem:
                        hitter_name = name_elem.get_text().strip()
                        mlbam_id = _extract_mlbam_id(name_elem.get('href', ''))
                        game['away_hitters'].append((hitter_name, pos, mlbam_id))

            # Get home team hitters (always set the key, even if empty)
            game['home_hitters'] = []
            home_list = container.find('ol', class_='starting-lineups__team--home')
            if home_list:
                for pos, li in enumerate(home_list.find_all('li', class_='starting-lineups__player'), 1):
                    name_elem = li.find('a', class_='starting-lineups__player--link')
                    if name_elem:
                        hitter_name = name_elem.get_text().strip()
                        mlbam_id = _extract_mlbam_id(name_elem.get('href', ''))
                        game['home_hitters'].append((hitter_name, pos, mlbam_id))
            
            # Match with pitcher info from the same game index
            if game_idx < len(pitcher_overviews):
                overview = pitcher_overviews[game_idx]
                summaries = overview.find_all('div', class_='starting-lineups__pitcher-summary')
                
                away_pitcher = ""
                home_pitcher = ""
                away_pitcher_id: Optional[int] = None
                home_pitcher_id: Optional[int] = None
                pitcher_count = 0
                
                # Extract non-empty pitcher summaries
                for summary in summaries:
                    pitcher_name_elem = summary.find('div', class_='starting-lineups__pitcher-name')
                    if pitcher_name_elem:
                        name_link = pitcher_name_elem.find('a')
                        if name_link:
                            name = name_link.get_text().strip()
                            if name:
                                pid = _extract_mlbam_id(name_link.get('href', ''))
                                if pitcher_count == 0:
                                    away_pitcher = name
                                    away_pitcher_id = pid
                                elif pitcher_count == 1:
                                    home_pitcher = name
                                    home_pitcher_id = pid
                                pitcher_count += 1
                
                game['away_pitcher'] = away_pitcher
                game['home_pitcher'] = home_pitcher
                game['away_pitcher_id'] = away_pitcher_id
                game['home_pitcher_id'] = home_pitcher_id
            
            if game and 'away_hitters' in game and 'home_hitters' in game:
                games.append(game)
        
        except Exception as e:
            print(f"  ⚠ Error parsing game {game_idx} ({away_code}@{home_code}): {e}")
    
    return games


def _normal_pitcher_names(pitcher_name: str, opener_bulk: Dict[str, str]) -> List[str]:
    names: list[str] = []
    if pitcher_name:
        names.append(pitcher_name)
        bulk_name = opener_bulk.get(pitcher_name)
        if bulk_name and bulk_name != pitcher_name:
            names.append(bulk_name)
    return names


def _enrich_pitcher_ids_from_statsapi(games: List[Dict]) -> None:
    """Fill in missing pitcher MLBAM IDs using the StatsAPI schedule endpoint."""
    today = datetime.now().strftime("%Y-%m-%d")
    try:
        r = requests.get(
            f"{STATSAPI}/schedule",
            params={"sportId": 1, "date": today, "hydrate": "probablePitcher,team"},
            timeout=15,
        )
        r.raise_for_status()
    except requests.RequestException:
        return

    api_games = {}
    for d in r.json().get("dates", []):
        for g in d.get("games", []):
            away_abbr = g["teams"]["away"]["team"].get("abbreviation", "")
            home_abbr = g["teams"]["home"]["team"].get("abbreviation", "")
            away_pp = g["teams"]["away"].get("probablePitcher", {})
            home_pp = g["teams"]["home"].get("probablePitcher", {})
            api_games[(away_abbr, home_abbr)] = {
                "away_pitcher_id": away_pp.get("id"),
                "home_pitcher_id": home_pp.get("id"),
            }

    for game in games:
        key = (game.get("away_code", ""), game.get("home_code", ""))
        api = api_games.get(key)
        if not api:
            continue
        if not game.get("away_pitcher_id") and api.get("away_pitcher_id"):
            game["away_pitcher_id"] = api["away_pitcher_id"]
        if not game.get("home_pitcher_id") and api.get("home_pitcher_id"):
            game["home_pitcher_id"] = api["home_pitcher_id"]


def _resolve_hitter_ids_from_statsapi(games: List[Dict]) -> None:
    """For confirmed lineups missing MLBAM IDs, look up each hitter by name
    via the StatsAPI people/search endpoint."""
    for game in games:
        for side in ("away", "home"):
            hitters = game.get(f"{side}_hitters") or []
            updated = []
            for entry in hitters:
                name, pos = entry[0], entry[1]
                mlbam_id = entry[2] if len(entry) > 2 else None
                if mlbam_id is None:
                    mlbam_id = _search_player_id(name)
                updated.append((name, pos, mlbam_id))
            game[f"{side}_hitters"] = updated


def _search_player_id(name: str) -> Optional[int]:
    """Look up a player's MLBAM ID via StatsAPI people/search."""
    try:
        r = requests.get(
            f"{STATSAPI}/people/search",
            params={"names": name, "sportId": 1, "active": True},
            timeout=10,
        )
        r.raise_for_status()
        people = r.json().get("people", [])
        if people:
            return people[0].get("id")
    except requests.RequestException:
        pass
    return None


def generate_csv_rows(games: List[Dict], opener_bulk: Dict[str, str]) -> List[Tuple[str, str, str, str, str, str, str, str]]:
    """
    Generate CSV rows from parsed games.

    Returns:
        List of tuples
            (matchup, hitter_name, pitcher_name, lineup_position, status,
             hitter_team, hitter_mlbam_id, pitcher_mlbam_id)
        where status is "projected" or "confirmed", hitter_team is the
        2-3 letter team code of the hitter's own team, and the MLBAM ID
        columns are numeric strings (or empty if unavailable).
    """
    rows = []

    for game in games:
        away_team = game.get('away_code', 'UNK')
        home_team = game.get('home_code', 'UNK')
        matchup_key = f"{away_team}@{home_team}"

        away_pitcher = game.get('away_pitcher', '')
        home_pitcher = game.get('home_pitcher', '')
        away_pitcher_id = game.get('away_pitcher_id')
        home_pitcher_id = game.get('home_pitcher_id')
        away_status = "projected" if game.get('away_projected') else "confirmed"
        home_status = "projected" if game.get('home_projected') else "confirmed"

        home_pitcher_names = _normal_pitcher_names(home_pitcher, opener_bulk)
        away_pitcher_names = _normal_pitcher_names(away_pitcher, opener_bulk)

        for entry in game.get('away_hitters', []):
            hitter_name, position = entry[0], entry[1]
            hitter_id = entry[2] if len(entry) > 2 else None
            for pitcher_name in home_pitcher_names:
                rows.append((matchup_key, hitter_name, pitcher_name, str(position),
                             away_status, away_team,
                             str(hitter_id) if hitter_id else "",
                             str(home_pitcher_id) if home_pitcher_id else ""))

        for entry in game.get('home_hitters', []):
            hitter_name, position = entry[0], entry[1]
            hitter_id = entry[2] if len(entry) > 2 else None
            for pitcher_name in away_pitcher_names:
                rows.append((matchup_key, hitter_name, pitcher_name, str(position),
                             home_status, home_team,
                             str(hitter_id) if hitter_id else "",
                             str(away_pitcher_id) if away_pitcher_id else ""))

    return rows


def fill_missing_lineups(games: List[Dict]) -> None:
    """For any game side with fewer than 9 hitters, fill in the most recent
    same-handed batting order from MLB StatsAPI and tag the side as projected."""
    for game in games:
        for side in ("away", "home"):
            hitters = game.get(f"{side}_hitters") or []
            if len(hitters) >= 9:
                continue
            team_code = game.get(f"{side}_code")
            opp_pitcher = game.get(f"{'home' if side == 'away' else 'away'}_pitcher")
            if not team_code or not opp_pitcher:
                continue
            try:
                projected = get_projected_lineup(team_code, opp_pitcher)
            except Exception as e:
                print(f"  ⚠ projection failed for {team_code} vs {opp_pitcher}: {e}")
                continue
            if projected:
                game[f"{side}_hitters"] = projected
                game[f"{side}_projected"] = True
                print(f"  ✓ projected lineup for {team_code} vs {opp_pitcher} "
                      f"(based on most recent same-handed start)")
            else:
                print(f"  ⚠ no projected lineup found for {team_code} vs {opp_pitcher}")


def load_opener_bulk_map(path: Path) -> Dict[str, str]:
    opener_bulk: Dict[str, str] = {}
    if not path.exists():
        return opener_bulk

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            parts = [part.strip() for part in text.split(",", 1)]
            if len(parts) != 2 or not parts[0] or not parts[1]:
                print(f"⚠ skipping invalid opener/bulk line {line_no}: {text}")
                continue
            opener, bulk = parts
            if opener == bulk:
                continue
            opener_bulk[opener] = bulk
    return opener_bulk


def main(argv=None):
    """Main entry point."""
    ap = argparse.ArgumentParser(description="Fetch MLB starting lineups and write matchups CSV")
    ap.add_argument("--opener-bulk-file", type=str, default="opener_bulk.csv",
                    help="optional opener/bulk mapping file with lines opener_name,bulk_name")
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    log_path = setup_logging("fetch_lineups")
    today = datetime.now().strftime("%Y-%m-%d")

    opener_bulk_map = load_opener_bulk_map(Path(args.opener_bulk_file))
    if opener_bulk_map:
        print(f"✓ Loaded {len(opener_bulk_map)} opener/bulk mapping(s) from {args.opener_bulk_file}")
    elif args.opener_bulk_file and Path(args.opener_bulk_file).exists():
        print(f"✓ No opener/bulk mappings found in {args.opener_bulk_file}")
    elif args.opener_bulk_file != "opener_bulk.csv":
        print(f"✓ opener/bulk file {args.opener_bulk_file} not found; continuing without opener/bulk mappings")

    print(f"🔄 Fetching starting lineups for {today}...")
    print(f"   logging to {log_path}")
    
    # Fetch page
    html = fetch_starting_lineups_page()
    if not html:
        return 1
    
    print("✓ Page fetched successfully\n")
    
    # Parse games
    print("Parsing games and lineups...")
    games = parse_games(html)
    
    if not games:
        print("✗ No games found")
        return 1
    
    print(f"✓ Found {len(games)} games\n")

    # Fill in missing lineups using most recent same-handed batting order
    print("Filling in missing lineups (if any)...")
    fill_missing_lineups(games)

    # Enrich with MLBAM IDs from StatsAPI (fills in any missing pitcher/hitter IDs)
    _enrich_pitcher_ids_from_statsapi(games)
    _resolve_hitter_ids_from_statsapi(games)

    projected_sides = []
    for g in games:
        if g.get('away_projected'):
            projected_sides.append(g.get('away_code'))
        if g.get('home_projected'):
            projected_sides.append(g.get('home_code'))
    if projected_sides:
        print(f"⚠ Projected (not confirmed) lineups: {', '.join(projected_sides)}")
    print()

    # Generate CSV rows
    all_rows = generate_csv_rows(games, opener_bulk_map)
    
    if not all_rows:
        print("✗ No lineup data collected")
        return 1
    
    # Write CSV file.  If the target already exists (i.e. fetch_lineups was
    # re-run later the same day), back the old copy up to a timestamped .bak
    # rather than silently overwriting.  This makes accidental clobbering of
    # a prior day's lineup file (which has burned us in the past) recoverable.
    output_filename = f"matchups_{today}.csv"
    output_path = Path(output_filename)

    if output_path.exists():
        stamp = datetime.now().strftime("%H%M%S")
        backup_path = output_path.with_suffix(f".csv.{stamp}.bak")
        try:
            output_path.replace(backup_path)
            print(f"ℹ {output_filename} already existed; backed up to "
                  f"{backup_path.name} before rewriting")
        except OSError as exc:
            print(f"✗ Could not back up existing {output_filename}: {exc}")
            return 1

    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(all_rows)

        print(f"✓ CSV written to {output_filename}")
        print(f"  - Games: {len(games)}")
        print(f"  - Rows written: {len(all_rows)}")
        return 0

    except IOError as e:
        print(f"✗ Error writing CSV file: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
