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
2. **Platoon-filtered data is inherently thinner** — Bauers as LHB vs LHP has far fewer PAs per count than he would vs RHP. The gate of 5 PA passes, but the estimates are unreliable.
3. **No shrinkage on per-count cells** — unlike per-pitch xwOBA (which gets empirical-Bayes shrinkage), the per-count xwOBA is taken at face value.
4. **No sensitivity to total coverage** — the alpha is a fixed 0.5 regardless of whether the count cells cover 90% of the batter's PAs or 20%.

### Empirical Evidence (2026-05-18 slate)

Distribution of count-shifts across 184 matchups:

| Stat | Value |
|------|-------|
| Mean | +12.4 pts |
| Median | +15.0 pts |
| Std | 21.6 pts |
| P5 / P95 | -20 / +37 pts |
| P1 / P99 | -105 / +56 pts |
| \|shift\| > 40 | 6.0% (11/184) |
| \|shift\| > 60 | 3.3% (6/184) |

**Critical finding:** Large shifts do NOT correlate with small samples (r = -0.13). Players like Ben Rice (1959 eff rows, +84 pts) and Andrew Vaughn (1626 eff rows, +82 pts) also get large shifts. A hard cap at 40 pts would be **too aggressive** — it would clip legitimate signal from well-sampled players.

The problem is specifically **thin per-count cells within the platoon filter**, not overall sample size. Bauers has plenty of total data but very few LvL PAs at specific count states.

## Proposed Fix: Adaptive Alpha + Shrinkage

### Layer 1: Raise the minimum PA gate

**Current:** `COUNT_XWOBA_MIN_PA_W = 5`
**Proposed:** `COUNT_XWOBA_MIN_PA_W = 15`

With 5 PA, the standard error of xwOBA is ~0.14 (σ/√n ≈ 0.35/√5). With 15 PA it drops to ~0.09. This eliminates the most egregious noise from count cells that happen to have one HR in 5 PA.

Impact: Some count cells will drop out (reducing coverage), which naturally pulls the shift toward 0 when data is sparse.

### Layer 2: Shrink per-count xwOBA toward batter overall

Apply empirical-Bayes shrinkage to per-count cells:

```python
def batter_xwoba_by_count(..., bat_overall_xwoba: float):
    for c, grp in ...:
        n_pa_w = pa["weight"].sum()
        if n_pa_w < min_pa_w:
            continue
        raw_x = w_xwoba(grp)
        shrunk_x = shrunk_rate(raw_x, n_pa_w, bat_overall_xwoba, COUNT_XWOBA_SHRINK_K)
        out[c] = shrunk_x
```

**Proposed:** `COUNT_XWOBA_SHRINK_K = 30`

Effect by sample size:
- 15 PA cell: 33% raw + 67% overall → heavy regression (this is Bauers' thin LvL cells)
- 50 PA cell: 63% raw + 37% overall → moderate
- 150+ PA cell: 83%+ raw → barely touched (Ben Rice, Andrew Vaughn)

The shrinkage target is the **batter's own overall xwOBA** (not league), so the delta `(shrunk_count_x - overall)` is naturally damped for thin cells while preserving signal for deep cells. This is the key differentiator — it specifically fixes the Bauers-type problem without constraining well-sampled players.

### Layer 3: Coverage-scaled alpha (replaces hard cap)

Instead of a fixed `COUNT_XWOBA_BLEND_ALPHA = 0.5` regardless of how much data contributed to the shift, scale the alpha by **what fraction of the batter's total PA is covered by the qualifying count cells**:

```python
COUNT_XWOBA_BLEND_ALPHA_MAX = 0.5       # ceiling (same as current)
COUNT_XWOBA_COVERAGE_FULL = 0.60        # at 60%+ coverage, full alpha

# In project():
total_pa_in_counts = sum(n_pa_per_qualifying_count)
total_pa = batter_overall["n_pa_w"]
coverage = total_pa_in_counts / total_pa if total_pa > 0 else 0.0
alpha_scaled = COUNT_XWOBA_BLEND_ALPHA_MAX * min(1.0, coverage / COUNT_XWOBA_COVERAGE_FULL)

count_shift = sum(...) * alpha_scaled
```

How this works:
- **Bauers vs LHP** — platoon filter leaves few qualifying counts → coverage ~20-30% → alpha drops to ~0.17-0.25 → shift halved or more
- **Ben Rice (full sample)** — most counts qualify → coverage ~80%+ → alpha stays at 0.5 → no constraint
- **Average player** — typically 50-70% coverage → alpha at 0.42-0.50 → mild or no reduction

This is a **soft** constraint that naturally self-adjusts based on data quality, rather than a cliff at an arbitrary threshold.

### What about the hard cap?

Removed from the plan. The empirical data shows that 6% of projections exceed ±40 pts, and many of these are well-sampled players where the count-state interaction is real (e.g., a pitcher with extreme count-conditional pitch mix facing a hitter with extreme count-conditional outcomes). Capping these would remove legitimate differentiation.

If after implementing layers 1-3 there are still >80 pt shifts, they would come from well-sampled, high-coverage data — which is a defensible signal.

## Summary of Changes

| Layer | Mechanism | Handles |
|-------|-----------|---------|
| **1. Min PA gate** | 5 → 15 | Drops noisiest cells entirely |
| **2. Per-cell shrinkage** | k=30 toward batter overall | Regresses thin cells proportionally (Bauers case) |
| **3. Coverage-scaled alpha** | alpha × min(1, coverage/0.60) | Reduces influence when few counts qualify (thin platoon splits) |

## Implementation

### Files Modified

`matchup.py` only.

### Constants (near line 997):

```python
COUNT_XWOBA_MIN_PA_W = 15.0               # was 5; raised to require meaningful sample
COUNT_XWOBA_SHRINK_K = 30.0               # shrink per-count cells toward batter overall
COUNT_XWOBA_BLEND_ALPHA_MAX = 0.5         # ceiling alpha (same as current default)
COUNT_XWOBA_COVERAGE_FULL = 0.60          # coverage fraction for full alpha
```

### Function changes:

1. **`batter_xwoba_by_count()`** — add `bat_overall_xwoba` param, apply `shrunk_rate` to each cell, also return per-count PA weights for coverage calc.

2. **`project()`** — compute coverage from the qualifying count cells, scale alpha, apply shift.

3. **Call site in `compute_matchup_pieces()`** — pass batter overall xwOBA to `batter_xwoba_by_count()`.

## Expected Impact

| Matchup type | Old shift | New shift (est.) | Why |
|-------------|-----------|------------------|-----|
| Bauers vs Imanaga (thin LvL) | +82 pts | +20-30 pts | Low coverage → reduced alpha; thin cells shrunk |
| Ben Rice (full sample, extreme) | +84 pts | +75-84 pts | High coverage, deep cells → barely touched |
| Andrew Vaughn (full sample) | +82 pts | +70-80 pts | Same — real signal preserved |
| Average matchup | +12 pts | +10-12 pts | No meaningful change |
| Joe Mack (thin sample, extreme) | -105 pts | -40-60 pts | Both low coverage and cell shrinkage kick in |

## Validation

1. Re-run Bauers vs Imanaga — confirm shift drops to ~20-30 pts
2. Re-run Ben Rice and Andrew Vaughn — confirm they stay >60 pts (signal preserved)
3. Backtest on 2-3 dates — xwOBA RMSE should improve (blowups reduced), Brier should hold

## Risk

Low. Each layer is conservative and independently defensible:
- Min PA 15 is standard (5 was permissive for any PA-level estimate)
- Shrinkage toward own overall is the mildest regression form — requires strong evidence to deviate
- Coverage-scaled alpha is a smooth function with no cliff; at worst it's neutral for well-sampled data
