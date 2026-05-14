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

import requests
import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import sys
from bs4 import BeautifulSoup

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


def get_projected_lineup(team_abbrev: str, opp_pitcher_name: str) -> List[Tuple[str, int]]:
    """Find the team's most recent batting order vs a same-handed starting pitcher.
    Returns [] if nothing usable is found."""
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
        result: List[Tuple[str, int]] = []
        for pos, pid in enumerate(bo, 1):
            name = players.get(f"ID{pid}", {}).get("person", {}).get("fullName")
            if name:
                result.append((name, pos))
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
                        game['away_hitters'].append((hitter_name, pos))

            # Get home team hitters (always set the key, even if empty)
            game['home_hitters'] = []
            home_list = container.find('ol', class_='starting-lineups__team--home')
            if home_list:
                for pos, li in enumerate(home_list.find_all('li', class_='starting-lineups__player'), 1):
                    name_elem = li.find('a', class_='starting-lineups__player--link')
                    if name_elem:
                        hitter_name = name_elem.get_text().strip()
                        game['home_hitters'].append((hitter_name, pos))
            
            # Match with pitcher info from the same game index
            if game_idx < len(pitcher_overviews):
                overview = pitcher_overviews[game_idx]
                summaries = overview.find_all('div', class_='starting-lineups__pitcher-summary')
                
                away_pitcher = ""
                home_pitcher = ""
                pitcher_count = 0
                
                # Extract non-empty pitcher summaries
                for summary in summaries:
                    pitcher_name_elem = summary.find('div', class_='starting-lineups__pitcher-name')
                    if pitcher_name_elem:
                        name_link = pitcher_name_elem.find('a')
                        if name_link:
                            name = name_link.get_text().strip()
                            if name:
                                if pitcher_count == 0:
                                    away_pitcher = name
                                elif pitcher_count == 1:
                                    home_pitcher = name
                                pitcher_count += 1
                
                game['away_pitcher'] = away_pitcher
                game['home_pitcher'] = home_pitcher
            
            if game and 'away_hitters' in game and 'home_hitters' in game:
                games.append(game)
        
        except Exception as e:
            print(f"  ⚠ Error parsing game {game_idx} ({away_code}@{home_code}): {e}")
    
    return games


def generate_csv_rows(games: List[Dict]) -> List[Tuple[str, str, str, str, str, str]]:
    """
    Generate CSV rows from parsed games.

    Returns:
        List of tuples
            (matchup, hitter_name, pitcher_name, lineup_position, status, hitter_team)
        where status is "projected" or "confirmed" and hitter_team is the
        2-3 letter team code of the hitter's own team.
    """
    rows = []

    for game in games:
        away_team = game.get('away_code', 'UNK')
        home_team = game.get('home_code', 'UNK')
        matchup_key = f"{away_team}@{home_team}"

        away_pitcher = game.get('away_pitcher', '')
        home_pitcher = game.get('home_pitcher', '')
        away_status = "projected" if game.get('away_projected') else "confirmed"
        home_status = "projected" if game.get('home_projected') else "confirmed"

        for hitter_name, position in game.get('away_hitters', []):
            rows.append((matchup_key, hitter_name, home_pitcher, str(position),
                         away_status, away_team))

        for hitter_name, position in game.get('home_hitters', []):
            rows.append((matchup_key, hitter_name, away_pitcher, str(position),
                         home_status, home_team))

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


def main():
    """Main entry point."""
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    today = datetime.now().strftime("%Y-%m-%d")
    
    print(f"🔄 Fetching starting lineups for {today}...")
    
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
    all_rows = generate_csv_rows(games)
    
    if not all_rows:
        print("✗ No lineup data collected")
        return 1
    
    # Write CSV file
    output_filename = f"matchups_{today}.csv"
    output_path = Path(output_filename)
    
    try:
        with open(output_path, 'w', newline='') as csvfile:
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
