#!/usr/bin/env python3
"""Build a navigable static site over the daily HTML reports.

Produces three kinds of files, all served by GitHub Pages from `main`:

* ``index.html``                       - meta-refresh redirect to the most recent
                                         date's hub, with a small "All dates" link.
* ``archive.html``                     - chronological list of every date that has
                                         a reports/<date>/ folder, newest first.
* ``reports/<date>/index.html``        - per-day hub: roundup CTAs at the top,
                                         then one card per game grouping each
                                         starting pitcher's report.

Per-pitcher report files are produced by ``matchup.py`` using the filename
pattern ``<away>_at_<home>_vs_<pitcher_slug>_<year>.html``. Roundups produced
by ``roundup.py`` use ``top<N>_<date>.html`` / ``bottom<N>_<date>.html``.

Usage::

    python build_site.py                   # rebuild root + every per-day index
    python build_site.py --date 2026-05-14 # rebuild just one day's hub + root
"""

from __future__ import annotations

import argparse
import html
import re
import sys
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).parent
REPORTS_DIR = ROOT / "reports"

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
ROUNDUP_RE = re.compile(r"^(top|bottom)(\d+)_(\d{4}-\d{2}-\d{2})\.html$", re.IGNORECASE)
PITCHER_RE = re.compile(
    r"^(?P<away>[a-z]{2,4})_at_(?P<home>[a-z]{2,4})_vs_(?P<slug>.+)_(?P<year>\d{4})\.html$",
    re.IGNORECASE,
)


SITE_CSS = """
:root {
  --bg: #f7f8fa;
  --card: #ffffff;
  --ink: #1f2937;
  --muted: #64748b;
  --line: #e5e7eb;
  --accent: #2563eb;
  --accent-soft: #eff6ff;
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; background: var(--bg); color: var(--ink);
             font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI',
                          Helvetica, Arial, sans-serif;
             font-size: 14px; line-height: 1.45; }
.container { max-width: 1180px; margin: 0 auto; padding: 24px 20px 80px; }
.page-head { padding-bottom: 16px; margin-bottom: 18px; border-bottom: 1px solid var(--line); }
.page-head h1 { margin: 0 0 6px 0; font-size: 26px; font-weight: 700; }
.page-head .meta { color: var(--muted); font-size: 13px; }
.page-head .meta a { color: var(--accent); text-decoration: none; }
.page-head .meta a:hover { text-decoration: underline; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 10px;
        padding: 18px 22px; margin-bottom: 16px; box-shadow: 0 1px 2px rgba(0,0,0,0.03); }
.card h2 { margin: 0 0 12px 0; font-size: 14px; font-weight: 700; color: var(--muted);
           letter-spacing: 0.6px; text-transform: uppercase; }
.note { color: var(--muted); font-size: 12px; margin: 6px 0 0 0; }

/* Roundup CTA buttons */
.cta-row { display: flex; flex-wrap: wrap; gap: 12px; }
.cta { flex: 1 1 240px; display: block; padding: 16px 18px; border-radius: 10px;
       text-decoration: none; color: var(--ink); border: 1px solid var(--line);
       background: var(--card); transition: transform 0.05s ease, box-shadow 0.1s ease; }
.cta:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.06); transform: translateY(-1px); }
.cta.top { background: linear-gradient(180deg, #ecfdf5 0%, #ffffff 70%); border-color: #a7f3d0; }
.cta.bot { background: linear-gradient(180deg, #fef2f2 0%, #ffffff 70%); border-color: #fecaca; }
.cta .cta-title { font-size: 16px; font-weight: 700; margin: 0; }
.cta .cta-sub { color: var(--muted); font-size: 12px; margin-top: 4px; }

/* Game grid */
.game-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
             gap: 14px; }
.game { background: var(--card); border: 1px solid var(--line); border-radius: 10px;
        padding: 14px 16px; box-shadow: 0 1px 2px rgba(0,0,0,0.03); }
.game h3 { margin: 0 0 10px 0; font-size: 14px; font-weight: 700;
           letter-spacing: 0.4px; }
.game h3 .away { color: var(--muted); }
.game h3 .at { color: var(--muted); margin: 0 6px; font-weight: 400; }
.game ul { list-style: none; padding: 0; margin: 0; }
.game li { margin: 0; padding: 6px 0; border-top: 1px solid #f1f5f9; }
.game li:first-child { border-top: none; padding-top: 2px; }
.game li a { color: var(--accent); text-decoration: none; font-weight: 600;
             font-size: 13px; }
.game li a:hover { text-decoration: underline; }
.game li .vs { color: var(--muted); font-weight: 400; margin-right: 4px; }

/* Archive list */
.archive { padding: 0; margin: 0; list-style: none; }
.archive li { padding: 12px 14px; border: 1px solid var(--line); border-radius: 8px;
              background: var(--card); margin-bottom: 8px;
              display: flex; justify-content: space-between; align-items: center; }
.archive li a { color: var(--accent); text-decoration: none; font-weight: 700;
                font-size: 15px; }
.archive li a:hover { text-decoration: underline; }
.archive li .count { color: var(--muted); font-size: 12px; }

footer { margin-top: 24px; color: var(--muted); font-size: 12px; }
footer a { color: var(--accent); text-decoration: none; }
footer a:hover { text-decoration: underline; }
"""


def _h(text) -> str:
    return html.escape("" if text is None else str(text))


def _href(filename: str) -> str:
    """URL-encode a filename for use in an href (handles non-ASCII like u-acute)."""
    return quote(filename, safe="")


def _list_dates() -> list[str]:
    if not REPORTS_DIR.exists():
        return []
    out = []
    for p in REPORTS_DIR.iterdir():
        if p.is_dir() and DATE_RE.match(p.name):
            out.append(p.name)
    out.sort(reverse=True)
    return out


def _classify_date_files(date_dir: Path) -> dict:
    """Bucket all .html files in a date dir into roundups vs per-game pitcher reports."""
    roundups: list[dict] = []
    games: dict[str, dict] = {}
    other: list[str] = []
    for p in sorted(date_dir.glob("*.html")):
        name = p.name
        if name == "index.html":
            continue
        m = ROUNDUP_RE.match(name)
        if m:
            kind, n, _date = m.groups()
            roundups.append({
                "kind": kind.lower(),
                "n": int(n),
                "filename": name,
            })
            continue
        m = PITCHER_RE.match(name)
        if m:
            away = m.group("away").upper()
            home = m.group("home").upper()
            slug = m.group("slug")
            game_key = f"{away}_at_{home}"
            pitcher_name = slug.replace("_", " ").title()
            games.setdefault(game_key, {
                "away": away, "home": home, "pitchers": [],
            })["pitchers"].append({
                "name": pitcher_name,
                "filename": name,
            })
            continue
        other.append(name)

    roundups.sort(key=lambda r: (r["kind"] != "top", r["n"]))
    for g in games.values():
        g["pitchers"].sort(key=lambda p: p["name"].lower())

    return {"roundups": roundups, "games": games, "other": other}


def _render_day_index(date_str: str, date_dir: Path) -> str:
    bucket = _classify_date_files(date_dir)
    roundups = bucket["roundups"]
    games = bucket["games"]
    other = bucket["other"]

    parts: list[str] = []
    parts.append("<!doctype html>")
    parts.append('<html lang="en">')
    parts.append("<head>")
    parts.append('<meta charset="utf-8">')
    parts.append('<meta name="viewport" content="width=device-width, initial-scale=1">')
    parts.append(f"<title>Matchup reports - {_h(date_str)}</title>")
    parts.append(f"<style>{SITE_CSS}</style>")
    parts.append("</head>")
    parts.append("<body>")
    parts.append('<main class="container">')

    parts.append('<header class="page-head">')
    parts.append(f"<h1>Matchup reports &middot; {_h(date_str)}</h1>")
    parts.append('<div class="meta">'
                 f'{len(games)} game{"s" if len(games) != 1 else ""} &middot; '
                 f'{sum(len(g["pitchers"]) for g in games.values())} pitcher report'
                 f'{"s" if sum(len(g["pitchers"]) for g in games.values()) != 1 else ""} &middot; '
                 '<a href="../../archive.html">All dates</a> &middot; '
                 '<a href="../../">Home</a>'
                 '</div>')
    parts.append("</header>")

    if roundups:
        parts.append('<section class="card">')
        parts.append("<h2>Slate roundup</h2>")
        parts.append('<div class="cta-row">')
        for r in roundups:
            cls = "top" if r["kind"] == "top" else "bot"
            label = f"Top {r['n']} hitters" if r["kind"] == "top" else f"Bottom {r['n']} hitters"
            sub = ("Best projected matchups across the slate"
                   if r["kind"] == "top"
                   else "Worst projected matchups across the slate")
            parts.append(
                f'<a class="cta {cls}" href="{_href(r["filename"])}">'
                f'<p class="cta-title">{_h(label)}</p>'
                f'<p class="cta-sub">{_h(sub)}</p>'
                f'</a>'
            )
        parts.append('</div>')
        parts.append('</section>')

    if games:
        parts.append('<section class="card">')
        parts.append("<h2>Games</h2>")
        parts.append('<div class="game-grid">')
        for game_key in sorted(games.keys()):
            g = games[game_key]
            parts.append('<div class="game">')
            parts.append(
                f'<h3><span class="away">{_h(g["away"])}</span>'
                f'<span class="at">@</span>'
                f'<span class="home">{_h(g["home"])}</span></h3>'
            )
            parts.append("<ul>")
            for pitcher in g["pitchers"]:
                parts.append(
                    f'<li><a href="{_href(pitcher["filename"])}">'
                    f'<span class="vs">vs</span>{_h(pitcher["name"])}'
                    f'</a></li>'
                )
            parts.append("</ul>")
            parts.append('</div>')
        parts.append('</div>')
        parts.append('</section>')

    if other:
        parts.append('<section class="card">')
        parts.append("<h2>Other reports</h2>")
        parts.append("<ul>")
        for fn in other:
            parts.append(f'<li><a href="{_href(fn)}">{_h(fn)}</a></li>')
        parts.append("</ul>")
        parts.append('</section>')

    if not roundups and not games and not other:
        parts.append('<section class="card">')
        parts.append('<p class="note">No HTML reports found in this folder yet.</p>')
        parts.append('</section>')

    parts.append("<footer>")
    parts.append('Generated by <a href="https://github.com/jsiri84/baseball-matchup">'
                 'baseball-matchup</a> &middot; <code>build_site.py</code>')
    parts.append("</footer>")

    parts.append("</main>")
    parts.append("</body></html>")
    return "\n".join(parts)


def _render_root_index(latest: str | None) -> str:
    """Root index.html: meta-refresh to the most recent date's hub."""
    if latest is None:
        return (
            "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<title>baseball-matchup</title>"
            f"<style>{SITE_CSS}</style></head><body>"
            "<main class='container'>"
            "<header class='page-head'><h1>baseball-matchup</h1>"
            "<div class='meta'>No reports generated yet.</div></header>"
            "<section class='card'><p class='note'>Run <code>python daily.py</code> "
            "to produce today's reports.</p></section>"
            "</main></body></html>"
        )
    target = f"reports/{latest}/index.html"
    target_url = f"reports/{quote(latest, safe='')}/index.html"
    return (
        "<!doctype html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f'<meta http-equiv="refresh" content="0; url={_h(target_url)}">\n'
        f'<link rel="canonical" href="{_h(target_url)}">\n'
        f"<title>baseball-matchup &middot; {_h(latest)}</title>\n"
        f"<style>{SITE_CSS}</style>\n"
        "</head>\n"
        "<body>\n"
        '<main class="container">\n'
        '<header class="page-head">\n'
        f"<h1>Latest reports: {_h(latest)}</h1>\n"
        '<div class="meta">Redirecting to today\'s hub&hellip; '
        '<a href="archive.html">All dates</a></div>\n'
        "</header>\n"
        '<section class="card">\n'
        f'<p>If you are not redirected, <a href="{_h(target_url)}">click here</a> '
        f"to open <code>{_h(target)}</code>.</p>\n"
        "</section>\n"
        "</main>\n"
        "</body></html>\n"
    )


def _render_archive(dates: list[str]) -> str:
    parts: list[str] = []
    parts.append("<!doctype html>")
    parts.append('<html lang="en">')
    parts.append("<head>")
    parts.append('<meta charset="utf-8">')
    parts.append('<meta name="viewport" content="width=device-width, initial-scale=1">')
    parts.append("<title>Archive &middot; baseball-matchup</title>")
    parts.append(f"<style>{SITE_CSS}</style>")
    parts.append("</head>")
    parts.append("<body>")
    parts.append('<main class="container">')
    parts.append('<header class="page-head">')
    parts.append("<h1>Report archive</h1>")
    parts.append(f'<div class="meta">{len(dates)} date'
                 f'{"s" if len(dates) != 1 else ""} available &middot; '
                 '<a href="index.html">Latest</a></div>')
    parts.append("</header>")
    parts.append('<section class="card">')
    if not dates:
        parts.append('<p class="note">No reports yet.</p>')
    else:
        parts.append('<ul class="archive">')
        for d in dates:
            date_dir = REPORTS_DIR / d
            n_html = sum(1 for p in date_dir.glob("*.html") if p.name != "index.html")
            parts.append(
                f'<li><a href="reports/{_href(d)}/index.html">{_h(d)}</a>'
                f'<span class="count">{n_html} report'
                f'{"s" if n_html != 1 else ""}</span></li>'
            )
        parts.append("</ul>")
    parts.append("</section>")
    parts.append("<footer>")
    parts.append('Generated by <a href="https://github.com/jsiri84/baseball-matchup">'
                 'baseball-matchup</a> &middot; <code>build_site.py</code>')
    parts.append("</footer>")
    parts.append("</main>")
    parts.append("</body></html>")
    return "\n".join(parts)


def build_one_day(date_str: str) -> Path:
    date_dir = REPORTS_DIR / date_str
    if not date_dir.exists():
        sys.exit(f"[build_site] reports dir not found: {date_dir}")
    out = date_dir / "index.html"
    out.write_text(_render_day_index(date_str, date_dir), encoding="utf-8")
    print(f"[build_site] wrote {out.relative_to(ROOT)}")
    return out


def build_root(dates: list[str]) -> None:
    latest = dates[0] if dates else None
    (ROOT / "index.html").write_text(_render_root_index(latest), encoding="utf-8")
    print(f"[build_site] wrote index.html (latest: {latest or 'none'})")
    (ROOT / "archive.html").write_text(_render_archive(dates), encoding="utf-8")
    print(f"[build_site] wrote archive.html ({len(dates)} date(s))")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", default=None,
                    help="Build only this date's hub (plus root). Default: rebuild every date.")
    args = ap.parse_args()

    dates = _list_dates()
    if not dates:
        print("[build_site] no reports/<date>/ folders found; writing empty root")
        build_root([])
        return 0

    if args.date:
        if not DATE_RE.match(args.date):
            sys.exit(f"[build_site] --date must be YYYY-MM-DD, got {args.date!r}")
        if args.date not in dates:
            sys.exit(f"[build_site] no folder at reports/{args.date}/")
        build_one_day(args.date)
    else:
        for d in dates:
            build_one_day(d)

    build_root(dates)
    print(f"[build_site] done. {len(dates)} date(s) indexed; latest = {dates[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
