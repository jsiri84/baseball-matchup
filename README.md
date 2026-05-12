# baseball-matchup

A Python toolkit for **batter-vs-pitcher** and **lineup-vs-pitcher** matchup analysis built on [pybaseball](https://github.com/jldbc/pybaseball) and Statcast / Baseball Savant data. Generates rich markdown + styled HTML reports with projected xwOBA, per-PA outcome odds, multi-PA outlooks, count-conditional pitch mix, shape-aware comps, vertical-approach-angle bat-tracking interaction, and more.

```bash
# single matchup
python matchup.py --batter "Yordan Alvarez" --pitcher "Paul Skenes"

# entire lineup vs one pitcher
python matchup.py --pitcher "Paul Skenes" --lineup "Yordan Alvarez,Aaron Judge,Jose Altuve"
```

The rest of this document is the reading guide for the reports the tool produces.

---

# Matchup report — reading guide

A reference for `<batter>_vs_<pitcher>_<season>_matchup.{md,html}` and `lineup_vs_<pitcher>_<season>.{md,html}` produced by [`matchup.py`](matchup.py). Walks through every section, every metric, the formulas behind the projections, and the cell-coloring rules in the HTML version.

---

## Table of contents

1. [Quick start](#quick-start)
2. [Where the data comes from](#where-the-data-comes-from)
3. [Methodology overview](#methodology-overview)
4. [Section-by-section reading guide](#section-by-section-reading-guide)
   1. [Header](#1-header)
   2. [Verdict](#2-verdict)
   3. [Per-PA outcome distribution](#3-per-pa-outcome-distribution)
   4. [Multi-PA outlook](#4-multi-pa-outlook)
   5. [Projection by times through the order](#5-projection-by-times-through-the-order)
   6. [Side-by-side profile](#6-side-by-side-profile-platoon-filtered)
   7. [Pitch-mix projection](#7-pitch-mix-projection)
   8. [Count-state pitch mix](#8-count-state-pitch-mix)
   9. [Shape-aware comps](#9-shape-aware-comps)
   10. [Zone overlay](#10-zone-overlay)
   11. [Bat-tracking interaction](#11-bat-tracking-interaction)
   12. [First-pitch & two-strike sub-profiles](#12-first-pitch--two-strike-sub-profiles)
   13. [Edge analysis](#13-edge-analysis)
   14. [Deception & shape signature](#14-deception--shape-signature)
   15. [Defensive alignment](#15-defensive-alignment)
   16. [Notes & caveats](#16-notes--caveats)
5. [Cell-coloring legend (HTML)](#cell-coloring-legend-html)
6. [Glossary of metrics](#glossary-of-metrics)
7. [Caveats and what is not modeled](#caveats-and-what-is-not-modeled)

---

## Quick start

```bash
python matchup.py                                                     # default: Yordan Alvarez vs Paul Skenes
python matchup.py --batter "Aaron Judge" --pitcher "Tarik Skubal"
python matchup.py --batter-id 670541 --pitcher-id 694973 --season 2026
python matchup.py --batch matchups.csv                                # one row per matchup

# Lineup mode: one report covering N batters vs the same pitcher
python matchup.py --pitcher "Paul Skenes" --lineup "Yordan Alvarez,Aaron Judge,Jose Altuve"
python matchup.py --pitcher "Tarik Skubal" --lineup-csv lineup.txt --pa-per-batter 3
```

Each single-matchup invocation writes two files:

- `<batter_last>_vs_<pitcher_last>_<season>_matchup.md` — plain markdown
- `<batter_last>_vs_<pitcher_last>_<season>_matchup.html` — styled, color-highlighted

Lineup mode writes:

- `lineup_vs_<pitcher_last>_<season>.md` and `.html` — a single document with a top-level outlook, a one-row-per-batter grid, and the full per-batter detail collapsed inside expandable blocks (HTML) or nested under `### N. Batter` headers (markdown).

The HTML version is the better read; the markdown version is git-friendly and embeds cleanly in chat.

### Lineup mode at a glance

The lineup HTML report is structured for easy digestion at scale:

1. **Lineup outlook hero** — the headline numbers in one row: lineup-wide projected xwOBA, total expected K / BB / hits / HR / on-base across all PAs, and `P(>=1 HR somewhere in the lineup)`. The xwOBA and `Δ vs league` cells are color-coded.
2. **Lineup grid** — one row per batter (in lineup order) with: spot, name, hand, projected xwOBA, Δ pts vs league, K%, BB%, HR%, Hit%, OB%, best pitch (with batter xwOBA), worst pitch (with batter xwOBA), and a verdict pill (`Even` / `Slight Hitter` / `Edge Hitter` / `Strong Hitter` and the pitcher equivalents). Every metric cell is colored vs its league baseline so the eye can find the matchups that matter at a glance. Click a batter's name to jump to (and expand) their full detail block.
3. **Per-batter detail** — each batter gets a collapsible `<details>` block whose `<summary>` shows the headline numbers and verdict pill. Expanding it reveals the complete per-batter report (verdict + narrative + outcome distribution + multi-PA outlook + TTO curve + side-by-side profile + pitch-mix + count-mix + shape comps + zone overlay + bat-tracking + sub-profiles + edge analysis + deception + alignment), styled exactly like the standalone single-matchup HTML.
4. **Footer notes** — the standard methodology caveats appear once at the bottom (not per batter).

CLI flags specific to lineup mode:

- `--lineup "Name1,Name2,..."` or `--lineup-csv path.txt` (one entry per line; numeric MLBAM IDs work as well as names; lines starting with `#` are comments)
- `--pa-per-batter N` (default 3) — used only for the rollup math (expected K / BB / hits / HR / on-base, and the `P(>=1 HR)` independence calculation)

If a single batter in the lineup fails to resolve or has no rows, the lineup run logs a warning and continues with the rest — it doesn't abort the whole report.

---

## Where the data comes from

Every report uses **only one pull per player per season**, via `pybaseball.statcast_batter()` and `pybaseball.statcast_pitcher()`. No league-wide pull, no leaderboard pull.

Each per-player dataset is a concatenation of one pull per season in `SEASON_WEIGHTS`, with a `weight` column attached. The default is a BPP-style 3-season decay:

| Window           | Date range                              | Weight (`SEASON_WEIGHTS`) |
|------------------|-----------------------------------------|--------------------------:|
| Current season   | `SEASON-03-27` → today                  | 1.0                       |
| Prior season     | `(SEASON-1)-03-27` → `(SEASON-1)-11-01` | 0.5                       |
| Two-prior season | `(SEASON-2)-03-27` → `(SEASON-2)-11-01` | 0.25                      |

Every aggregation in every layer of the report uses these weights. The header line tells you the raw row counts per season:

```
window: 2026 ×1, 2025 ×0.5, 2024 ×0.25 ·
        batter rows 688+824+612, pitcher rows 608+2997+2615
```

That means the batter has 688 current-season pitches seen, 824 from the prior season, and 612 from two seasons ago; the pitcher has 608 current-season pitches thrown, 2997 from the prior season, and 2615 from two seasons ago. The "effective sample size" `eff_n` shown in some tables is the dot product of row counts with the season weights.

Tune `SEASON_WEIGHTS` at the top of `matchup.py` (index 0 = current, index 1 = prior, etc.):

- `[1.0]` — current season only
- `[1.0, 0.5]` — old 2-season blend
- `[1.0, 0.5, 0.25]` — default; BPP-style 3-season decay
- `[1.0, 0.5, 0.25, 0.1]` — extend further back (slower pulls, more sample)

### Caching

- All pulls are cached to `data/statcast_*_<start>_<end>.parquet`
- The `end` date is rounded to **yesterday** so all matchups run on a given day share the same cache file
- An in-process `dict[player_id, DataFrame]` deduplicates pulls within a `--batch` run, so a 50-matchup batch across 30 unique players does 30 disk reads, not 100

---

## Methodology overview

### Platoon filter (Layer 0)

Before any computation, the batter's pitches seen are filtered to the pitcher's `p_throws` (the batter's history vs same-handed pitchers as the opponent). Symmetrically, the pitcher's pitches thrown are filtered to the batter's `stand`. This is the single biggest accuracy lever — a slider from a RHP to a LHB plays totally differently than to a RHB.

### Combining batter rate and pitcher rate

Two complementary methods are used depending on the metric type.

**Log5 (odds-ratio)** for binary rate stats — `K%`, `BB%`, `HBP%`, `Whiff%`, `Hard Hit %`:

```
log5(b, p, lg) = (b * p / lg) / ((b * p / lg) + (1 - b)(1 - p)/(1 - lg))
```

This is the standard sabermetric formulation for combining two rates against a league baseline, derived from the assumption that batter and pitcher each have a multiplicative effect on the outcome odds.

**Additive** for continuous rate stats — `xwOBA`, `xBA`, `xSLG`:

```
matchup = batter + pitcher - league                  (clipped to [0, 1])
```

Used because additive matches the way Statcast's expected stats are constructed (per-PA averages of per-event expectations) better than log5.

### Count-conditional pitch mix (Layer 1)

The pitcher's pitch mix isn't flat across counts — they throw a fastball ~45% of the time in 0-0 and a splitter heavy in 0-2. The batter's PAs visit different counts at different rates depending on their approach. The marginal pitch usage in *this matchup* is:

```
P(pitch_i) = sum_c P(c | batter) * P(pitch_i | c, pitcher)
```

This typically shifts the projected outcomes 5-15% versus a flat-usage projection, because put-away pitches are over-represented in deep counts.

### Headline projection

For each pitch type `i` in the pitcher's arsenal with marginal usage `p_i`:

```
projected_xwOBA       = sum_i p_i * additive(batter_xwOBA_i, pitcher_xwOBA_i, LG_XWOBA)
projected_Whiff%      = sum_i p_i * log5(batter_Whiff_i, pitcher_Whiff_i, LG_WHIFF)
projected_Hard_Hit%   = sum_i p_i * log5(batter_HH_i, pitcher_HH_i, LG_HARD_HIT)
projected_K%          = log5(batter_K%, pitcher_K%, LG_K_PCT)        # PA-level, not pitch-level
projected_BB%         = log5(batter_BB%, pitcher_BB%, LG_BB_PCT)
projected_HBP%        = log5(batter_HBP%, pitcher_HBP%, LG_HBP_PCT)
```

K%, BB%, and HBP% use overall rates because they're determined by the whole PA, not any single pitch. Per-pitch-type values (Whiff%, Hard Hit%, xwOBA) are weighted by the marginal pitch mix.

### League baselines

Hardcoded constants at the top of `matchup.py`, sourced from 2025 league averages (2026 finals are not finalized until the offseason):

| Constant       | Value | Description                          |
|----------------|------:|--------------------------------------|
| `LG_XWOBA`     | .315  | League average xwOBA                 |
| `LG_XBA`       | .245  | League average xBA                   |
| `LG_XSLG`      | .405  | League average xSLG                  |
| `LG_K_PCT`     | .225  | League strikeout rate                |
| `LG_BB_PCT`    | .085  | League walk rate                     |
| `LG_HBP_PCT`   | .012  | League hit-by-pitch rate             |
| `LG_WHIFF`     | .245  | League whiff rate (per swing)        |
| `LG_HARD_HIT`  | .400  | League hard-hit rate (≥95 mph EV)    |

`LG_OUTCOMES` (per-PA outcome distribution) holds the same baselines for the per-outcome rows.

---

## Section-by-section reading guide

### 1. Header

```
# Yordan Alvarez vs Paul Skenes — matchup
window: 2026 ×1, 2025 ×0.5, 2024 ×0.25 ·
        batter rows 688+824+612, pitcher rows 608+2997+2615
handedness: LHB vs RHP
```

- **Player names** — display form (`First Last`); `player_name` from Statcast comes as `Last, First` and is normalized
- **Window line** — one entry per season in `SEASON_WEIGHTS` with its weight, plus raw row counts for both players (one number per season, joined with `+`)
- **Handedness** — `B`HB (batter stands) vs `P`HP (pitcher throws). All downstream layers are platoon-filtered to this matchup.

### 2. Verdict

This is the headline takeaway. Three components, top to bottom:

#### Narrative (auto-generated, 1-2 sentences)

Templated from the projection, the top batter-favoring pitch in the arsenal, the top pitcher-favoring pitch, and the platoon note. Example:

> Skenes' split-finger is the projected matchup advantage, while Yordan Alvarez should look to drive the 4-seam fastball. Projection: 0.398 xwOBA (+83 pts vs league, Hitter). LHB vs RHP — opposite-side platoon advantage to the hitter.

#### Verdict box (3 rows)

| Frame                       | Projected xwOBA | Baseline | Δ (wOBA pts) | Read                              |
|-----------------------------|----------------:|---------:|-------------:|-----------------------------------|
| vs league avg               | 0.398           | 0.315    | +83          | **Edge: Hitter**                  |
| vs Skenes' baseline         | 0.398           | 0.252    | +146         | Edge: Hitter (vs Pitcher norm)    |
| vs Alvarez's baseline       | 0.398           | 0.433    | -35          | Edge: Pitcher (vs Batter norm)    |

- **vs league avg** — the absolute strength of this matchup. Above .315 means the batter projects above an average AB.
- **vs pitcher's baseline** — does the batter do better or worse than the pitcher's typical opponent? Yordan vs Skenes projects +146 wOBA points above Skenes' typical hitter — Skenes is great, but this hitter is exceptional.
- **vs batter's baseline** — does the pitcher hold the batter below their typical AB? Yordan projects -35 wOBA points below his own season norm — Skenes is genuinely difficult for him too.

The Δ-pts and Read columns are colored in the HTML by edge magnitude:
- `>+15 wOBA pts` → strong batter edge (deep green)
- `+5 to +15`  → mild batter edge (light green)
- `-5 to +5` → neutral (no color)
- `-5 to -15` → mild pitcher edge (light red)
- `<-15` → strong pitcher edge (deep red)

### 3. Per-PA outcome distribution

The per-plate-appearance outcome breakdown, summing to 100%. Derived from the headline projection plus a hit-type mix (1B/2B/3B/HR shares within hits) pulled from the batter's per-pitch-type table, then weighted by the marginal pitch mix.

Formula:

```
K%, BB%, HBP%        from log5 (Layer 1)
BIP%                 = 1 - K% - BB% - HBP%
hit_per_BIP          = projected_xBA * (1 - BB% - HBP%) / BIP%
hit-type mix         = sum_i marginal_usage_i * batter_hit_mix_vs_pitch_i
1B%, 2B%, 3B%, HR%   = BIP% * hit_per_BIP * hit-type mix shares
BIP_out%             = BIP% - (1B% + 2B% + 3B% + HR%)
```

A consistency check then reconstructs wOBA from the outcome shares × the official wOBA weights and rescales the hit components if it drifts more than 5 wOBA points from the projected xwOBA.

#### Columns

- **Outcome** — one of K, BB, HBP, 1B, 2B, 3B, HR, BIP-out (mutually exclusive, sum to 100%); plus convenience rollups Hit (any) and On-base (which do not sum into the total)
- **Prob** — projected per-PA probability
- **American** — sportsbook-style American odds (e.g., `+395` for a 20.2% event, `-150` for a 60% event). Converted from `Prob` via:
  ```
  +X = round(100 * (1 - p) / p)   if p < 0.5
  -X = round(-100 * p / (1 - p))  if p >= 0.5
  ```
- **League** — the league-average rate for that outcome (from `LG_OUTCOMES`)

The Prob column is colored in the HTML by deviation from the League column. Direction depends on the outcome:
- For outcomes where high is good for the batter (Walk, HBP, Single, Double, Triple, Home Run, Hit (any), On-base): high = green, low = red
- For outcomes where high is good for the pitcher (Strikeout, In-play out): low = green, high = red

### 4. Multi-PA outlook

Models a typical 2 / 3 / 4-PA night. For each per-PA outcome with probability `p`:

```
P(at least one in N PAs) = 1 - (1 - p)^N
E[count over N PAs]      = N * p
```

#### Columns

- **Outcome** — same set as the per-PA distribution
- **N PA: ≥1** — chance the outcome happens at least once across N PAs
- **N PA: E[#]** — expected count across N PAs

Cell coloring uses the per-PA league baseline scaled to N — e.g., the league chance of at least one HR in 4 PAs is `1 - (1 - 0.030)^4 = 11.5%`, so a 19.5% projection shades green.

> **Assumption**: PAs are treated as independent draws from the per-PA distribution. Real PAs share context (pitcher fatigue, count-state momentum, lineup turn), which the TTO section addresses separately.

### 5. Projection by times through the order

Pitchers degrade ~10-20 wOBA points per time through the order. This table shows the headline projection bumped by the per-TTO xwOBA delta the pitcher has historically allowed:

```
projected_xwOBA(TTO=t) = base_projection_xwOBA + (pitcher_xwOBA_allowed_at_TTO_t
                                                - pitcher_xwOBA_allowed_overall)
```

#### Columns

- **TTO** — 1 (first time through), 2 (second), 3+ (third or later)
- **Proj xwOBA** — projection for this PA position
- **Proj K %**, **Proj Whiff %**, **Proj Hard Hit %** — base headline rates (constant across TTOs in v1)
- **Sample (eff PA)** — weighted PA count from the pitcher's TTO bucket; small samples mean the delta is noisy

Use this when a player will face the pitcher more than once in a game ("leadoff in the first" projects differently than "with two on in the sixth").

### 6. Side-by-side profile (platoon-filtered)

Both players' overall numbers, restricted to the relevant platoon:

| Metric        | Batter (vs same-hand pitchers)  | Pitcher (vs same-hand batters)         |
|---------------|--------------------------------:|---------------------------------------:|
| Chase %       | swings on pitches outside the zone | swings forced on pitches outside the zone |
| Whiff %       | misses per swing                | misses generated per swing             |
| K %           | strikeout rate per PA           | strikeout rate forced per PA           |
| BB %          | walk rate per PA                | walks allowed per PA                   |
| Hard Hit %    | rate of contact ≥ 95 mph EV     | rate of contact ≥ 95 mph EV allowed    |
| Barrel %      | rate of barrels (Statcast classification 6) | barrels allowed                |
| GB %          | ground-ball share of BBE        | ground-ball share allowed              |
| Air %         | fly + line + popup share of BBE | air-ball share allowed                 |
| xwOBA         | per-PA expected wOBA            | per-PA expected wOBA allowed           |

Cell coloring uses the **same direction in both columns** because the metric semantics are symmetric: a high Whiff% is bad for the batter whether you're looking at the batter's own rate or the pitcher's whiff-generating rate.

Plain-English notes are auto-generated when threshold combinations fire (e.g., GB-heavy pitcher meeting an air-ball hitter).

### 7. Pitch-mix projection

The Layer 1 projection broken down by pitch type. One row per pitch in the pitcher's arsenal, sorted by marginal usage.

#### Columns

- **Pitch** — pitch name from Statcast (e.g., `4-Seam Fastball`, `Sweeper`, `Split-Finger`)
- **Marginal Usage %** — count-conditional pitch-mix marginal `p_i` (see [Methodology](#count-conditional-pitch-mix-layer-1))
- **Batter xwOBA** — batter's xwOBA against this pitch type (platoon-filtered)
- **Pitcher xwOBA allowed** — pitcher's xwOBA allowed on this pitch (platoon-filtered)
- **Projected xwOBA** — `additive(batter, pitcher, LG_XWOBA)` for this pitch
- **Projected Whiff %** — `log5(batter, pitcher, LG_WHIFF)` for this pitch

All three xwOBA columns are colored against `LG_XWOBA = 0.315`. A row with three green cells means: the batter is good against this pitch, the pitcher gives up a lot on this pitch, and the projected matchup result is well above league.

### 8. Count-state pitch mix

Compact pitch × count matrix from `pit_vs_bat`. Rows are the pitcher's top 6 pitches (by overall usage); columns are the most common counts plus 0-0 / 0-2 / 3-2 always. Cell value is the percentage of pitches in that count that were of that type.

This shows the pitcher's *sequencing* — fastball-heavy in 0-0, splitter-heavy in 1-2, etc. The Layer 1 projection already incorporates this; the table is here for context.

### 9. Shape-aware comps

Per arsenal pitch, finds comparable pitches in the batter's history and reports the batter's results against that shape — not just against pitches with the same name.

A comp matches when, in the batter's pitch-by-pitch history (platoon-filtered):

- Same `pitch_group` (Fastball / Breaking / Offspeed)
- `|effective_speed - arsenal_velo| ≤ 2.0 mph`
- `|api_break_x_batter_in - arsenal_HB| ≤ 3.0 in`
- `|api_break_z_with_gravity - arsenal_IVB| ≤ 3.0 in`

#### Columns

- **Pitch** — arsenal pitch name
- **Shape (eff velo / IVB / HB-in)** — the arsenal pitch's average shape (perceived velo, gravity-included vertical break, batter-perspective horizontal break)
- **n comps (eff)** — weighted count of matching pitches in the batter's history
- **Whiff %**, **xwOBA**, **Hard Hit %** — batter's results on the matched comps
- **Confidence** — sample-size tier:
  - `high` — `eff_n ≥ 30`
  - `medium` — `eff_n` between 15 and 30
  - `low` — fell below 15 with the strict tolerance, so tolerances were widened by 1.5× to find more comps
  - `no comps` — no matches even after widening

xwOBA and Whiff% cells are colored vs league baselines. Treat `low` confidence rows skeptically.

#### Why this matters beyond the pitch-name table

The pitch-mix table aggregates "all sliders" — but a sweeper at 84 mph with 11 inches of horizontal break plays nothing like a slider at 89 mph with 4 inches of break. Shape comps cut through pitch-name aliasing. If the pitcher's sweeper has a tiny number of close comps (`low` confidence), the batter has rarely seen anything like it — a real edge for the pitcher even if the pitcher's "Sweeper" line in the pitch-mix table looks ordinary.

### 10. Zone overlay

Where the pitcher attacks with each pitch type, crossed with where the batter does damage.

#### Columns

- **Pitch** — arsenal pitch name
- **In-zone %** — share of this pitch type that lands in zones 1-9 (Statcast in-zone grid)
- **Top zones (attack share)** — the three most-attacked zones for this pitch, with shares (e.g., `z13=33%, z14=16%, z7=13%`)
- **Intersection xwOBA** — `sum_z (pitcher_attack_share_z * batter_xwOBA_z)`. Restricted to zones with batter data.

The Statcast zone grid:

```
Zones 1-9 in the strike zone (3x3):
  1 2 3   ← top row
  4 5 6
  7 8 9   ← bottom row
Zones 11-14 outside the strike zone:
  11 = up-left of zone   12 = up-right
  13 = down-left         14 = down-right (most splitter & changeup territory)
```

The intersection xwOBA is the most important number here. A pitcher who lives in zones the batter handles poorly (low batter xwOBA there) gets a low intersection — the pitch projects worse than its raw pitch-type xwOBA suggests. This catches the "Skenes lives down-and-away with the splitter where Yordan does nothing" effect.

### 11. Bat-tracking interaction

Compares the batter's average swing path to the pitcher's average pitch plane.

The header line shows the batter's overall bat-tracking averages (only ~50% of swings have bat-tracking data, so this is a partial measurement):

- **bat speed** — average swing velocity in mph
- **swing length** — average bat path length in feet
- **attack angle** — vertical angle of the bat at contact, in degrees (positive = upward swing path)

The per-arsenal-pitch table compares the **batter's attack angle on this specific pitch type** against the **pitcher's vertical approach angle (VAA) at the plate** for that pitch.

**VAA (deg)** — actual vertical approach angle at the front of home plate, in degrees, computed from Statcast's per-pitch initial conditions (`vy0`, `vz0`, `ay`, `az`) by solving the trajectory quadratic for time-to-plate and then taking `atan2(vz_plate, |vy_plate|)`. Negative for a descending pitch. Typical values: high four-seam −4° to −5°, sinker −6° to −7°, splitter −7° to −9°, curveball −8° to −12°. Weighted-average across all of the pitcher's pitches of that type vs same-handed batters.

**Bat attack (deg)** — batter's weighted-mean attack angle on swings against this specific pitch type (against same-handed pitchers). Hitters typically have flatter attack angles on fastballs and steeper on offspeed, so this varies meaningfully by pitch type. If the batter has fewer than 5 effective swings on this pitch type, the column shows the batter's overall attack angle with an asterisk (`*`) instead.

**Swings (n)** — effective sample size (weighted by `SEASON_WEIGHTS`) of bat-tracked swings on this pitch type. Bat-tracking data only became publicly available in mid-2024, so deep arsenals on rarely-used pitch types can be thin.

**Match gap (deg)** — `bat_attack + VAA`. Because VAA is negative and attack angle is positive, this is signed:
- `|gap| ≤ 3°` → "on plane" (swing and pitch share a plane; long contact window)
- `gap > +3°` → swing steeper than pitch — pop-up / swing-under risk (typical for steep-uppercut hitters vs. high four-seamers)
- `gap < −3°` → swing flatter than pitch — topped / grounder risk (typical for flat hitters vs. steep curveballs)

The HTML version colors the cell green for "on plane", mild red for `|gap| > 3°`, and strong red for `|gap| > 6°`.

### 12. First-pitch & two-strike sub-profiles

Two pitch-count slices that decide a lot of PAs.

**First pitch** (`pitch_number == 1`):
- **pitcher first-pitch strike%** — fraction of first pitches that go for a strike (called, swinging, foul, in-play)
- **batter first-pitch swing%** — fraction of first pitches the batter swings at
- **batter xwOBA on first-pitch swings** — what happens when the batter does swing on 0-0

**Two-strike** (`strikes == 2`):
- **pitcher putaway%** — `K's in two-strike counts / two-strike PAs`
- **batter K% in 2-strike counts** — same denominator, batter perspective
- **batter xwOBA in 2-strike counts** — overall xwOBA when batter is down to one strike
- **two-strike pitch mix** — pitcher's most-used pitches when ahead 0-2 / 1-2 / 2-2

If the pitcher's putaway% is high and the batter's two-strike K% is high, expect strikeouts. If the batter's two-strike xwOBA holds up well, they're a tough out even when behind.

### 13. Edge analysis

Two compact tables. For each pitch in the arsenal:

```
edge_score = marginal_usage * (batter_xwOBA - LG_XWOBA)
```

- **Pitches favoring the hitter** — top 3 by `edge_score` (most positive)
- **Pitches favoring the pitcher** — top 3 by `edge_score` (most negative)

Each row shows the usage %, batter xwOBA, and pitcher xwOBA allowed. The HTML highlights the batter xwOBA column green in the hitter-favoring table, and the pitcher xwOBA column red in the pitcher-favoring table.

This is a quick-glance answer to "which 3 pitches do I want to see, and which 3 should I lay off?".

### 14. Deception & shape signature

**Release-point cluster** — the average distance (in inches) of each pitch's release point from the pitcher's center of mass across all pitch types:
- `< 1.5 in` → **tight (deceptive)** — every pitch comes from nearly the same slot, the hitter can't tell pitch type from release
- `1.5 - 3.5 in` → **moderate**
- `> 3.5 in` → **loose** — the hitter has clear visual cues per pitch type

Per-arsenal-pitch table columns:
- **Δ from release centroid (in)** — how far this pitch's release point is from the pitcher's average. Outlier pitches tip themselves.
- **Spin axis (deg)** — clock-face equivalent. ~180° is pure backspin (4-seamer), ~0° is pure topspin (curveball), ~90° / 270° is gyro-style. Useful for distinguishing slider sub-types when `pitch_name` lumps them.

A handedness verdict (LHB vs RHP — opposite-side platoon advantage to the hitter, etc.) follows the table.

### 15. Defensive alignment

How the batter does on grounders by infield alignment, and how the pitcher's defense typically aligns:

- **Batter ground-ball BABIP by alignment** — split into `Standard` / `Strategic` / `Infield shade` / `Shift` (whatever shows up in `if_fielding_alignment` for this batter). League GB BABIP is ~ .240; cells are colored vs that baseline.
- **Pitcher's typical infield alignment usage** — what alignments the pitcher's defense most often uses. If the pitcher's defense usually plays Strategic and the batter does poorly against Strategic, expect fewer grounder hits than the raw BABIP would suggest.

The displayed sample sizes (`Sample (eff GB)`) are weighted GB counts. Single-season samples here are tiny — treat this as a directional adjustment, not a precise number.

### 16. Notes & caveats

A short footer block listing the methodology assumptions and what is *not* modeled. Always worth re-reading on your first matchup.

---

## Cell-coloring legend (HTML)

Four CSS classes are applied to numeric cells based on how far the value deviates from the relevant baseline:

| Class                | Visual                  | Meaning                                   |
|----------------------|-------------------------|-------------------------------------------|
| `bat-edge-strong`    | deep green, bold        | strong advantage to the hitter            |
| `bat-edge-mild`      | light green             | mild advantage to the hitter              |
| (no class)           | neutral                 | within ~half a "scale" unit of baseline   |
| `pit-edge-mild`      | light red               | mild advantage to the pitcher             |
| `pit-edge-strong`    | deep red, bold          | strong advantage to the pitcher           |

The threshold uses a per-metric `scale`:

```
delta = (value - baseline) * (1 if batter_favors_high else -1)
strong:  |delta| ≥ 1.5 * scale
mild:    0.5 * scale ≤ |delta| < 1.5 * scale
neutral: |delta| < 0.5 * scale
```

Per-metric scales (rough sense of "what counts as a notable edge"):

| Metric         | Direction (high = good for) | Scale  |
|----------------|----------------------------|-------:|
| xwOBA          | batter                      | 30 wOBA pts |
| K %            | pitcher                     | 4 pp        |
| BB %           | batter                      | 2.5 pp      |
| Whiff %        | pitcher                     | 4 pp        |
| Hard Hit %     | batter                      | 5 pp        |
| Barrel %       | batter                      | 2 pp        |
| Chase %        | pitcher                     | 4 pp        |
| GB %           | pitcher                     | 6 pp        |
| Air %          | batter                      | 6 pp        |
| HR % per PA    | batter                      | 1.2 pp      |
| Single % per PA| batter                      | 2.5 pp      |
| BIP-out % per PA | pitcher                  | 4 pp        |

---

## Glossary of metrics

### Outcome metrics

- **AVG / OBP / SLG / OPS** — traditional slash line. Not in the matchup report directly; the per-PA outcome shares can be summed to recover them.
- **wOBA** (weighted on-base average) — a single rate stat that gives credit per outcome via FanGraphs constants:
  ```
  wOBA = (0.691*BB + 0.722*HBP + 0.882*1B + 1.252*2B + 1.584*3B + 2.037*HR) / (AB + BB + SF + HBP)
  ```
  League average is ~ .315. The `WOBA_*` constants used here come from FanGraphs' 2025 GUTS! values (used as a stand-in until 2026 finals are published).
- **xwOBA** (expected wOBA) — the same formula, but every batted ball uses Statcast's `estimated_woba_using_speedangle` — the wOBA value MLB's model assigns based on launch speed and angle. Strips out defense and luck.
- **xBA** — expected batting average; per AB, K's count as 0.
- **xSLG** — expected slugging; same denominator as xBA.

### Plate-discipline metrics

- **Whiff %** — swings and misses per swing. League average ~ 24.5%.
- **Chase %** — swings on pitches outside the strike zone (zones 11-14), per out-of-zone pitch. League ~ 28.5%.
- **K %** — strikeouts per plate appearance. League ~ 22.5%.
- **BB %** — walks per plate appearance. League ~ 8.5%.
- **HBP %** — hit-by-pitches per PA. League ~ 1.2%.
- **PutAway %** — pitcher metric: strikeouts in two-strike counts / total two-strike counts.

### Batted-ball metrics

- **BBE** (batted ball events) — pitches put in play with a measured launch speed.
- **Hard Hit %** — share of BBE with launch speed ≥ 95 mph. League ~ 40%.
- **Barrel %** — share of BBE with `launch_speed_angle == 6` (Statcast's barrel classification, requires a velocity-dependent angle window).
- **GB / FB / LD / PU %** — share of BBE classified as `ground_ball` / `fly_ball` / `line_drive` / `popup`.
- **Air %** — `(FB + LD + PU)` share of BBE.
- **launch_speed** (EV) — exit velocity of contact in mph.
- **launch_angle** (LA) — vertical angle of contact in degrees.
- **hyper_speed** — Statcast's park-and-temperature adjusted exit velocity.

### Pitch-shape metrics

- **release_speed** — raw pitch velocity at release in mph.
- **effective_speed** — perceived velocity, adjusted for pitcher extension. A 97 mph fastball at 6.8 ft of extension plays like ~98.5 mph.
- **pfx_x** — raw horizontal break in feet (sign convention varies by handedness).
- **pfx_z** — raw vertical break in feet.
- **api_break_x_batter_in** — horizontal break in inches, **already sign-flipped to the batter's perspective**. Positive = breaks toward the batter (in), negative = away.
- **api_break_z_with_gravity** — vertical break in inches with gravity included. This is what the hitter actually sees, not the "spin-induced" component alone.
- **release_spin_rate** — pitch spin rate in rpm.
- **spin_axis** — clock-face spin axis in degrees. ~180° is pure backspin (rising fastball look), ~0° is topspin (curveball look), 90° or 270° is gyro/bullet spin (no break).
- **release_extension** — distance in feet in front of the rubber where the pitcher releases the ball.
- **arm_angle** — pitcher's arm slot in degrees.
- **release_pos_x / z** — release point coordinates in feet from the rubber.

### Bat-tracking metrics (~50% of swings have these)

- **bat_speed** — bat velocity at contact in mph.
- **swing_length** — total bat-path length in feet.
- **attack_angle** — vertical angle of the bat at contact in degrees.
- **swing_path_tilt** — lateral tilt of the swing path.

### Pitch-classification groups

Used by Layer 2 (shape comps) and the count-state mix.

- **Fastball** — 4-Seam Fastball, Sinker, Cutter
- **Breaking** — Slider, Curveball, Sweeper, Slurve, Knuckle Curve, Slow Curve, Eephus, Knuckleball
- **Offspeed** — Changeup, Split-Finger, Forkball, Screwball

### Run-value metrics

- **delta_run_exp** — Statcast's per-pitch change in run expectancy from the offense's perspective. Positive = good for offense.
- **delta_pitcher_run_exp** — same, from the pitcher's perspective. Positive = good for pitcher.
- **Batting / Pitching Run Value** — sum of `delta_run_exp` over a window. Used in the standalone `yordan.py` and `pitcher.py` summaries; not surfaced directly in the matchup report.

### Game-state metrics

- **balls / strikes** — count.
- **count_state** — `"balls-strikes"` (e.g., `"0-0"`, `"3-2"`).
- **pitch_number** — sequence number of pitch within the PA.
- **n_thruorder_pitcher** — times through the order for the pitcher (1, 2, 3, …).
- **tto_bucket** — `n_thruorder_pitcher` clipped to `[1, 2, 3]`; what Layer 4 groups by.
- **zone** — Statcast zone (1-9 in-zone, 11-14 out-of-zone).
- **stand** — batter's batting hand (`L` or `R`).
- **p_throws** — pitcher's throwing hand (`L` or `R`).
- **if_fielding_alignment** — `Standard`, `Strategic`, `Infield shade`, or `Shift`.

---

## Caveats and what is not modeled

- **Single-season cells are noisy.** Even with the current+prior blend, some per-pitch-type / per-zone cells have effective sample sizes in the single digits. Use the `Confidence` column in the shape-comps table as your guide; treat low-confidence rows as suggestive, not definitive.
- **Park effects.** Yordan plays his home games at MMP (Crawford Boxes inflate HR for LHB), Skenes pitches at PNC (suppresses HR). The matchup report uses `hyper_speed` (park-adjusted EV) where available, but does not apply a HR park factor to the projection.
- **Weather, wind, temperature.** Not modeled. Cold-weather April matchups will project ~ same as humid August matchups; in reality HR rates differ ~ 20%.
- **Catcher framing / umpire zone.** Both materially affect K/BB rates and are not in the data we pull. A great framer adds ~ 3-5% to a pitcher's strike rate; a tight ump compresses BB% and inflates K%.
- **Pitcher fatigue mid-start beyond TTO.** The TTO curve captures the broad pattern, but doesn't model in-game velocity decline (visible in `release_speed` per inning) or pitch-count fatigue.
- **Recent form / hot-cold streaks.** A rolling 14-day xwOBA could be added but at this sample size adds more noise than signal.
- **Batter swing decisions are static.** The projection assumes the batter's per-pitch-type approach against this pitch is the same as their season profile — it doesn't model "the batter knows Skenes throws a splitter and adjusts his approach in 2-strike counts."
- **Per-PA outcomes are independent.** The Multi-PA outlook treats each PA as an independent draw; in reality there's positive correlation (a pitcher who's locating well in PA 1 likely is in PA 2).
- **No leaderboard-based percentiles.** To add Statcast-style percentile rankings you'd need to pull `pybaseball.statcast_*_expected_stats(year)` and rank in pandas. Adding this is a small additional layer if useful.
