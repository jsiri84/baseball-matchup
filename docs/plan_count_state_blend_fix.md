# Plan: Count-State Blend Fix

## Problem Statement

The count-state xwOBA blend (`COUNT_XWOBA_BLEND_ALPHA = 0.5`) produces unreasonably large shifts when the batter has noisy per-count data. The Bauers vs Imanaga case shows a **+82 pt shift** — turning a below-average .324 xwOBA into a monster .448 projection.

### Root Cause

The mechanism computes:

```python
count_shift = sum(
    P(count | pitcher) * (batter_xwOBA_at_count - batter_xwOBA_overall)
) * COUNT_XWOBA_BLEND_ALPHA
```

The problem surfaces when:

1. **Small per-count samples** — `COUNT_XWOBA_MIN_PA_W = 5` is extremely low. A batter with 5 weighted PA in a count has a wildly noisy xwOBA estimate (a single HR makes the cell ~1.500+).
2. **Platoon-filtered data is inherently thinner** — Bauers as LHB vs LHP has far fewer PAs per count than he would vs RHP. The gate of 5 PA passes, but the estimates are garbage.
3. **No shrinkage on per-count cells** — unlike per-pitch xwOBA (which gets empirical-Bayes shrinkage), the per-count xwOBA is taken at face value.
4. **No cap on the shift magnitude** — the shift can be arbitrarily large if a few count cells are inflated. An 82-pt shift is larger than the entire difference between an elite hitter and a replacement-level one.

### Current Flow

```
batter_xwoba_by_count():
    for each count_state:
        if PA_weight >= 5:  ← too low
            emit batter_xwOBA_at_count  ← no shrinkage

project():
    count_shift = sum(P_pitcher(count) * delta_batter(count)) * 0.5  ← no cap
    final_xwoba = (per_pitch_projection + count_shift) * park_pf
```

## Proposed Fix: Three-Layer Defense

### Layer 1: Raise the minimum PA gate

**Current:** `COUNT_XWOBA_MIN_PA_W = 5`
**Proposed:** `COUNT_XWOBA_MIN_PA_W = 15`

With 5 PA, the standard error of xwOBA is ~0.14 (σ/√n ≈ 0.35/√5). With 15 PA it drops to ~0.09. This eliminates the most egregious noise from count cells that happen to have one HR in 5 PA.

Impact: Some count cells will drop out (reducing coverage of the `common` set), which naturally pulls `count_shift` toward 0 when data is sparse.

### Layer 2: Shrink per-count xwOBA toward overall

Apply the same `shrunk_rate` logic to per-count cells:

```python
def batter_xwoba_by_count(...):
    for c, grp in ...:
        n_pa_w = pa["weight"].sum()
        if n_pa_w < min_pa_w:
            continue
        raw_x = w_xwoba(grp)
        # Shrink toward the batter's overall xwOBA
        shrunk_x = shrunk_rate(raw_x, n_pa_w, bat_overall_xwoba, COUNT_XWOBA_SHRINK_K)
        out[c] = shrunk_x
```

**Proposed:** `COUNT_XWOBA_SHRINK_K = 30`

This means:
- 15 PA cell: shrunk rate = (15*raw + 30*overall) / 45 = 33% raw + 67% overall (heavy regression)
- 50 PA cell: (50*raw + 30*overall) / 80 = 63% raw + 37% overall (moderate)
- 150+ PA cell: mostly raw (~83%+)

The shrinkage target is the **batter's own overall xwOBA** (not league), so the delta `(shrunk_count_x - overall)` is naturally damped but still directional when the signal is real.

### Layer 3: Cap the shift magnitude

Add a hard cap on `|count_shift|` to prevent any single mechanism from dominating the projection:

```python
COUNT_SHIFT_CAP_PTS = 40.0  # max ±40 wOBA points from count blend

count_shift = sum(...) * COUNT_XWOBA_BLEND_ALPHA
count_shift = max(-COUNT_SHIFT_CAP_PTS/1000, min(COUNT_SHIFT_CAP_PTS/1000, count_shift))
```

40 pts is still a meaningful adjustment (larger than most park factors) but prevents the 80+ pt blowups. For reference:
- A +40 pt shift is the difference between a .315 league-average hitter and a .355 good hitter
- The Bauers case would be capped at +40 instead of +82

### Summary of Changes

| Defense | Constant | Old | New | Effect |
|---------|----------|-----|-----|--------|
| Min PA gate | `COUNT_XWOBA_MIN_PA_W` | 5 | 15 | Drops noisy cells entirely |
| Shrinkage | `COUNT_XWOBA_SHRINK_K` | (none) | 30 | Regresses thin cells toward batter overall |
| Hard cap | `COUNT_SHIFT_CAP_PTS` | (none) | 40 | Prevents blowups regardless |

## Implementation

### Files Modified

`matchup.py` only.

### Code Changes

1. **Add constants** (near line 997):
```python
COUNT_XWOBA_MIN_PA_W = 15.0      # was 5; raised to require meaningful sample
COUNT_XWOBA_SHRINK_K = 30.0      # shrink per-count cells toward batter overall
COUNT_SHIFT_CAP_PTS = 40.0       # max |shift| in wOBA points (0.040)
```

2. **Update `batter_xwoba_by_count()`** to accept `bat_overall_xwoba` and apply shrinkage:
```python
def batter_xwoba_by_count(bat_vs_pit: pd.DataFrame,
                          bat_overall_xwoba: float,
                          min_pa_w: float = COUNT_XWOBA_MIN_PA_W) -> dict[str, float]:
    ...
    for c, grp in ...:
        n_pa_w = float(pa["weight"].sum())
        if n_pa_w < min_pa_w:
            continue
        raw_x = w_xwoba(grp)
        if raw_x is None or math.isnan(raw_x):
            continue
        shrunk_x = shrunk_rate(raw_x, n_pa_w, bat_overall_xwoba, COUNT_XWOBA_SHRINK_K)
        out[str(c)] = float(shrunk_x)
    return out
```

3. **Update `project()`** — add cap after computing `count_shift`:
```python
count_shift = sum(...) * COUNT_XWOBA_BLEND_ALPHA
# Cap to prevent noisy count cells from dominating the projection
cap = COUNT_SHIFT_CAP_PTS / 1000.0
count_shift = max(-cap, min(cap, count_shift))
```

4. **Update call site** in `compute_matchup_pieces()` to pass `bat_overall_xwoba` to `batter_xwoba_by_count()`.

### Backward Compatibility

- The function signature change is internal (no external callers)
- Reports will show lower count-blend shifts (the `Count-state blend: +Xpts` narrative adjusts automatically)
- No CSV format changes

## Validation Plan

### Unit Check

Re-run the Bauers vs Imanaga matchup and confirm:
- Count-state blend drops from +82 pts to something ≤ +40 pts
- Overall projection drops from .448 to something more reasonable (~.350-.380 range)
- Established matchups with large samples (e.g., Alvarez vs Skenes) barely change

### Backtest

Score against the existing `pa_results.parquet` on 2-3 dates:
- xwOBA RMSE should improve (the +82pt type blowups inflate RMSE)
- Log-loss / Brier should hold or improve (overclaiming hurts calibration)
- The "extreme decile" pass rate should improve (these are the projections most affected)

## Expected Impact

| Matchup type | Old count_shift | New count_shift (est.) |
|-------------|-----------------|------------------------|
| Bauers vs Imanaga (thin LvL) | +82 pts | +30-40 pts (capped) |
| Established hitter vs SP (thick) | +5-15 pts | +5-12 pts (minimal change) |
| Average matchup | ±0-10 pts | ±0-10 pts (no change) |

The fix is conservative — it only materially affects projections where the count-blend was producing >40 pt shifts from noisy data. Well-sampled matchups are barely touched.

## Risk

Low. The three layers are independent and each individually defensible:
- Raising min PA from 5→15 is standard (5 was too permissive for any real estimate)
- Shrinking toward own overall is the mildest form of shrinkage (not even toward league)
- A 40-pt cap is generous (still allows the mechanism to contribute meaningfully)

The only risk is **under-powering** a real effect — but the calibration sweep showed that count-blend shifts above 40 pts almost always regress back (they're noise, not signal).
