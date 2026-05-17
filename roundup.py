#!/usr/bin/env python3
"""Day-level roundup: generate top-50 / bottom-50 hitter reports.

After `matchup.py --batch matchups_<date>.csv` completes, every per-team
lineup writes a small JSON sidecar to ``reports/<date>/_data/`` containing the
summary rows + the pre-rendered per-batter HTML body. This script reads those
sidecars, ranks hitters across the entire slate by projected xwOBA, and emits
two standalone HTML reports:

* ``reports/<date>/top50_<date>.html``    -- 50 highest projected xwOBA
* ``reports/<date>/bottom50_<date>.html`` -- 50 lowest projected xwOBA

Each report mirrors the per-team lineup format (same grid + per-batter detail
blocks), with two additional columns: the hitter's team and the opposing
pitcher.

Usage::

    python roundup.py                  # use today's reports folder
    python roundup.py --date 2026-05-13
    python roundup.py --top 25 --bottom 25
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date as date_cls
from pathlib import Path

from matchup import (
    LG_BB_PCT,
    LG_K_PCT,
    LG_OUTCOMES,
    LG_XBA,
    LG_XSLG,
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

ROOT = Path(__file__).parent

LG_HIT_PCT = sum(LG_OUTCOMES[k] for k in ("1B", "2B", "3B", "HR"))
LG_OB_PCT = sum(LG_OUTCOMES[k] for k in ("1B", "2B", "3B", "HR", "BB", "HBP"))


def _load_sidecars(report_dir: Path) -> list[dict]:
    """Load all pregame game-entries for a date.

    Prefers the consolidated ``_data/slate.json`` (single canonical source of
    truth, regenerated each batch run); falls back to legacy per-game JSONs
    for older report directories that predate the slate format.
    """
    data_dir = report_dir / "_data"
    if not data_dir.exists():
        return []
    slate_path = data_dir / "slate.json"
    if slate_path.exists():
        try:
            payload = json.loads(slate_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[roundup] failed to read {slate_path.name}: {exc}",
                  file=sys.stderr)
        else:
            games = payload.get("games", [])
            if games:
                return list(games)
    out: list[dict] = []
    for p in sorted(data_dir.glob("*.json")):
        if p.name == "slate.json" or p.name.startswith("_postgame"):
            continue
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception as exc:
            print(f"[roundup] skipping {p.name}: {exc}", file=sys.stderr)
    return out


def _flatten_batters(sidecars: list[dict]) -> list[dict]:
    """Flatten sidecars into one row per batter, attaching team + pitcher."""
    rows: list[dict] = []
    for sc in sidecars:
        team = sc.get("hitter_team") or ""
        pitcher_name = sc.get("pitcher_name") or sc.get("pitcher_meta", {}).get("name", "")
        p_throws = sc.get("pitcher_meta", {}).get("p_throws") or "?"
        projected = bool(sc.get("projected"))
        matchup_key = sc.get("matchup_key") or ""
        out_stem = sc.get("out_stem") or ""
        season = sc.get("season")
        summary_rows = sc.get("summary_rows") or []
        per_batter_html = sc.get("per_batter_html") or []

        # Summary rows and per-batter HTML are 1:1 by index from analyze_lineup.
        for idx, sr in enumerate(summary_rows):
            body_html = per_batter_html[idx] if idx < len(per_batter_html) else ""
            rows.append({
                "team": team,
                "pitcher_name": pitcher_name,
                "p_throws": p_throws,
                "matchup_key": matchup_key,
                "projected": projected,
                "out_stem": out_stem,
                "season": season,
                "sr": sr,
                "body_html": body_html,
            })
    return rows


def _verdict_pill_html(label: str, css: str) -> str:
    return f'<span class="verdict-pill {css}">{_h(label)}</span>'


def _summary_line(rank: int, row: dict) -> str:
    """Inline summary line for a per-batter <details><summary>."""
    sr = row["sr"]
    delta_str = f"{sr['delta_pts']:+.0f} pts"
    proj_flag = ' <span class="badge" title="Projected lineup">PROJ</span>' if row["projected"] else ""
    slash_bits = [f"xwOBA <b>{sr['proj_xwoba']:.3f}</b>"]
    xba = sr.get("proj_xba")
    xslg = sr.get("proj_xslg")
    if xba is not None and not (isinstance(xba, float) and math.isnan(xba)):
        slash_bits.append(f"xBA <b>{xba:.3f}</b>")
    if xslg is not None and not (isinstance(xslg, float) and math.isnan(xslg)):
        slash_bits.append(f"xSLG <b>{xslg:.3f}</b>")
    slash_str = " / ".join(slash_bits)
    return (
        f'<span class="spot">{rank}.</span>'
        f'<span class="name">{_h(sr["name"])}</span>'
        f'<span class="badge">{_h(sr.get("stand") or "?")}HB</span>'
        f'<span class="summary-stat">{_h(row["team"] or "?")} '
        f'vs <b>{_h(row["pitcher_name"] or "?")}</b> '
        f'({_h(row["p_throws"])}HP){proj_flag}</span>'
        f'<span class="summary-stat">proj {slash_str} ({delta_str})</span>'
        f'<span class="summary-stat">K <b>{sr["k_pct"]*100:.1f}%</b></span>'
        f'<span class="summary-stat">BB <b>{sr["bb_pct"]*100:.1f}%</b></span>'
        f'<span class="summary-stat">HR <b>{sr["hr_pct"]*100:.1f}%</b></span>'
        f'<span class="summary-stat">Hit <b>{sr["hit_pct"]*100:.1f}%</b></span>'
        f'{_verdict_pill_html(sr["verdict_label"], sr["verdict_css"])}'
    )


def _grid_row_html(rank: int, row: dict) -> str:
    sr = row["sr"]
    proj_cls = edge_class(sr["proj_xwoba"], LG_XWOBA, 0.025, batter_favors_high=True)
    xba = sr.get("proj_xba")
    xslg = sr.get("proj_xslg")
    xba_cls = edge_class(xba, LG_XBA, 0.025, batter_favors_high=True)
    xslg_cls = edge_class(xslg, LG_XSLG, 0.040, batter_favors_high=True)
    delta_cls = (
        "bat-edge-strong" if sr["delta_pts"] >= 50
        else "bat-edge-mild" if sr["delta_pts"] >= 25
        else "pit-edge-strong" if sr["delta_pts"] <= -50
        else "pit-edge-mild" if sr["delta_pts"] <= -25
        else ""
    )
    k_cls = edge_class(sr["k_pct"]*100, LG_K_PCT*100, 3.0, batter_favors_high=False)
    bb_cls = edge_class(sr["bb_pct"]*100, LG_BB_PCT*100, 2.0, batter_favors_high=True)
    hr_cls = edge_class(sr["hr_pct"]*100, LG_OUTCOMES["HR"]*100, 1.0, batter_favors_high=True)
    hit_cls = edge_class(sr["hit_pct"]*100, LG_HIT_PCT*100, 2.0, batter_favors_high=True)
    # Park (pts) replaces OB% to match the per-game lineup grid in
    # matchup.py.  Same 15.0-pt edge scale so coloring is consistent
    # across the report.
    park_pts = float(sr.get("park_pts", 0.0) or 0.0)
    park_cls = edge_class(park_pts, 0.0, 15.0, batter_favors_high=True)
    park_sign = "+" if park_pts >= 0 else ""

    anchor = f"r{rank}-{sr.get('anchor', 'batter')}"
    proj_marker = ' <span class="badge" title="Projected lineup">P</span>' if row["projected"] else ""

    return (
        "<tr>"
        f'<td class="spot">{rank}</td>'
        f'<td class="handpill"><span class="pill">{_h(row["team"] or "?")}</span></td>'
        f'<td class="name"><a href="#{_h(anchor)}">{_h(sr["name"])}</a>{proj_marker}</td>'
        f'<td class="handpill"><span class="pill">{_h(sr.get("stand") or "?")}HB</span></td>'
        f'<td class="name">{_h(row["pitcher_name"] or "?")} '
        f'<span class="pill">{_h(row["p_throws"])}HP</span></td>'
        f'{_td(f"{sr['proj_xwoba']:.3f}", proj_cls)}'
        f'{_td(fmt3(xba), xba_cls)}'
        f'{_td(fmt3(xslg), xslg_cls)}'
        f'{_td(f"{sr['delta_pts']:+.0f}", delta_cls)}'
        f'{_td(f"{sr['k_pct']*100:.1f}%", k_cls)}'
        f'{_td(f"{sr['bb_pct']*100:.1f}%", bb_cls)}'
        f'{_td(f"{sr['hr_pct']*100:.1f}%", hr_cls)}'
        f'{_td(f"{sr['hit_pct']*100:.1f}%", hit_cls)}'
        f'{_td(f"{park_sign}{park_pts:.0f}", park_cls)}'
        f'<td class="pitch-cell">{_h(sr.get("best_pitch") or "—")}</td>'
        f'<td class="pitch-cell">{_h(sr.get("worst_pitch") or "—")}</td>'
        f'<td class="verdict {sr["verdict_css"]}">{_h(sr["verdict_label"])}</td>'
        "</tr>"
    )


def _build_html(title: str, subtitle: str, rows: list[dict],
                date_str: str, total_pool: int) -> str:
    parts: list[str] = []
    parts.append("<!doctype html>")
    parts.append('<html lang="en">')
    parts.append("<head>")
    parts.append('<meta charset="utf-8">')
    parts.append(f"<title>{_h(title)}</title>")
    parts.append(f"<style>{_HTML_CSS}</style>")
    parts.append("</head>")
    parts.append("<body>")
    parts.append('<main class="container">')

    # Header
    parts.append('<header class="page-head">')
    parts.append(f"<h1>{_h(title)}</h1>")
    parts.append(
        f'<div class="meta">{_h(subtitle)} &middot; '
        f'showing {len(rows)} of {total_pool} hitters from {_h(date_str)}\'s slate</div>'
    )
    parts.append("</header>")

    # Grid
    parts.append('<section class="card">')
    parts.append("<h2>Ranking grid</h2>")
    parts.append('<table class="lineup-grid"><thead><tr>'
                 '<th>#</th><th>Team</th>'
                 '<th style="text-align:left">Batter</th><th>Hand</th>'
                 '<th style="text-align:left">Pitcher</th>'
                 '<th>Proj xwOBA</th><th>Proj xBA</th><th>Proj xSLG</th>'
                 '<th>&Delta; (pts)</th>'
                 '<th>K%</th><th>BB%</th><th>HR%</th><th>Hit%</th><th>Park (pts)</th>'
                 '<th style="text-align:left">Best pitch</th>'
                 '<th style="text-align:left">Worst pitch</th>'
                 '<th>Verdict</th>'
                 '</tr></thead><tbody>')
    for rank, row in enumerate(rows, start=1):
        parts.append(_grid_row_html(rank, row))
    parts.append("</tbody></table>")
    parts.append(
        '<p class="note">Click a batter\'s name (or the row below) to expand the full '
        'per-batter matchup report. Cells are colored vs the league baseline for that '
        'metric. Hitters tagged <span class="badge">P</span> come from a projected '
        '(not-yet-confirmed) lineup.</p>'
    )
    parts.append("</section>")

    # Per-batter detail
    parts.append('<section class="card">')
    parts.append("<h2>Per-batter detail</h2>")
    for rank, row in enumerate(rows, start=1):
        anchor = f"r{rank}-{row['sr'].get('anchor', 'batter')}"
        parts.append(f'<details class="batter-block" id="{_h(anchor)}">')
        parts.append(f'<summary>{_summary_line(rank, row)}</summary>')
        parts.append('<div class="batter-body">')
        parts.append(row["body_html"] or "<p class='note'>(detail body unavailable)</p>")
        parts.append("</div>")
        parts.append("</details>")
    parts.append("</section>")

    # Footer
    parts.append("<footer>")
    parts.append("<ul class='note-list'>")
    parts.append(
        "<li>Hitters are ranked across every posted (or projected) lineup for the "
        "day, by projected xwOBA against their listed opposing starter.</li>"
    )
    parts.append(
        "<li>Each per-batter detail block is identical to the one in the source "
        "lineup report; see <code>README.md</code> for the layer-by-layer methodology.</li>"
    )
    parts.append("</ul>")
    parts.append("</footer>")
    parts.append(report_timestamp_html())

    parts.append("</main>")
    parts.append(sortable_html())
    parts.append("</body></html>")
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", default=date_cls.today().isoformat(),
                    help="Report date (YYYY-MM-DD). Defaults to today.")
    ap.add_argument("--top", type=int, default=50, help="Number of top hitters (default 50).")
    ap.add_argument("--bottom", type=int, default=50, help="Number of bottom hitters (default 50).")
    ap.add_argument("--reports-dir", default=None,
                    help="Override reports directory (default reports/<date>/).")
    args = ap.parse_args()

    log_path = setup_logging("roundup")
    print(f"[roundup] logging to {log_path}")

    if args.reports_dir:
        report_dir = Path(args.reports_dir)
    else:
        report_dir = ROOT / "reports" / args.date

    if not report_dir.exists():
        sys.exit(f"[roundup] reports dir not found: {report_dir}")

    sidecars = _load_sidecars(report_dir)
    if not sidecars:
        sys.exit(f"[roundup] no sidecar JSONs under {report_dir}/_data/. "
                 f"Run `matchup.py --batch` first.")

    all_rows = _flatten_batters(sidecars)
    if not all_rows:
        sys.exit("[roundup] sidecars contained no batter summary rows.")

    print(f"[roundup] loaded {len(sidecars)} lineup sidecar(s) "
          f"= {len(all_rows)} hitter row(s)")

    sorted_desc = sorted(all_rows, key=lambda r: r["sr"]["proj_xwoba"], reverse=True)
    sorted_asc = sorted(all_rows, key=lambda r: r["sr"]["proj_xwoba"])

    top_n = min(args.top, len(sorted_desc))
    bot_n = min(args.bottom, len(sorted_asc))
    top_rows = sorted_desc[:top_n]
    bot_rows = sorted_asc[:bot_n]

    top_html = _build_html(
        title=f"Top {top_n} hitters by projected xwOBA",
        subtitle="Best matchups across the slate (highest projected xwOBA first)",
        rows=top_rows,
        date_str=args.date,
        total_pool=len(all_rows),
    )
    bot_html = _build_html(
        title=f"Bottom {bot_n} hitters by projected xwOBA",
        subtitle="Worst matchups across the slate (lowest projected xwOBA first)",
        rows=bot_rows,
        date_str=args.date,
        total_pool=len(all_rows),
    )

    top_path = report_dir / f"top{top_n}_{args.date}.html"
    bot_path = report_dir / f"bottom{bot_n}_{args.date}.html"
    top_path.write_text(top_html, encoding="utf-8")
    bot_path.write_text(bot_html, encoding="utf-8")

    print(f"[roundup] wrote {top_path.relative_to(ROOT)}")
    print(f"[roundup] wrote {bot_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
