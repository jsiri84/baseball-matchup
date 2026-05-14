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
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import sys
from bs4 import BeautifulSoup

STARTING_LINEUPS_URL = "https://www.mlb.com/starting-lineups"


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
            # Get away team hitters
            away_list = container.find('ol', class_='starting-lineups__team--away')
            if away_list:
                hitters = []
                for pos, li in enumerate(away_list.find_all('li', class_='starting-lineups__player'), 1):
                    name_elem = li.find('a', class_='starting-lineups__player--link')
                    if name_elem:
                        hitter_name = name_elem.get_text().strip()
                        hitters.append((hitter_name, pos))
                game['away_hitters'] = hitters
            
            # Get home team hitters
            home_list = container.find('ol', class_='starting-lineups__team--home')
            if home_list:
                hitters = []
                for pos, li in enumerate(home_list.find_all('li', class_='starting-lineups__player'), 1):
                    name_elem = li.find('a', class_='starting-lineups__player--link')
                    if name_elem:
                        hitter_name = name_elem.get_text().strip()
                        hitters.append((hitter_name, pos))
                game['home_hitters'] = hitters
            
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


def generate_csv_rows(games: List[Dict]) -> List[Tuple[str, str, str, str]]:
    """
    Generate CSV rows from parsed games.
    
    Args:
        games: List of game dictionaries
    
    Returns:
        List of tuples (matchup, hitter_name, pitcher_name, lineup_position)
    """
    rows = []
    
    for game in games:
        away_team = game.get('away_code', 'UNK')
        home_team = game.get('home_code', 'UNK')
        matchup_key = f"{away_team}@{home_team}"
        
        away_pitcher = game.get('away_pitcher', '')
        home_pitcher = game.get('home_pitcher', '')
        
        # Away team hitters face the home team's pitcher
        for hitter_name, position in game.get('away_hitters', []):
            rows.append((matchup_key, hitter_name, home_pitcher, str(position)))
        
        # Home team hitters face the away team's pitcher
        for hitter_name, position in game.get('home_hitters', []):
            rows.append((matchup_key, hitter_name, away_pitcher, str(position)))
    
    return rows


def main():
    """Main entry point."""
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
