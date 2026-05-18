# Plan: Adaptive Sample-Size Shrinkage

## Problem Statement

The current shrinkage uses **fixed** `k` values regardless of player sample size:

```python
K_PCT_SHRINK_K = 100.0
BB_PCT_SHRINK_K = 75.0
XWOBA_SHRINK_K = 100.0
PITCHER_XWOBA_SHRINK_K = 75.0
HARD_HIT_SHRINK_K = 75.0
```

The formula is: `shrunk = (n * rate + k * lg) / (n + k)`

This means a September call-up with `n_pa_w = 50` gets shrunk the same proportion toward league as a full-season regular with `n_pa_w = 600`. The call-up's shrunk rate is `(50r + 100*lg) / 150 = 0.33r + 0.67*lg` — heavily regressed. The regular's is `(600r + 100*lg) / 700 = 0.86r + 0.14*lg` — mostly raw. That's actually working as intended for a *single-season* sample.

**The real asymmetry is between batter and pitcher.** Batter rates (K%, BB%) are shrunk toward league. Pitcher overall rates are NOT shrunk. This was a deliberate decision (commit comments say shrinking pitchers collapsed the spread ratio to 0.40), but it means:
- A pitcher with 80 PA in the current season has their K% taken at face value
- A batter with 80 PA has their K% pulled ~50% toward league average

This asymmetry biases projections toward "the pitcher is who he says he is" regardless of sample, while treating the batter with skepticism.

## Proposed Approach: Per-Player Adaptive k

Instead of a global `k`, use a `k` that scales with the **reliability** of the metric given the player's sample. The key insight: shrinkage strength should be proportional to the **noise variance** relative to the **signal variance** in the player's observed data.

### Formula

Replace:
```python
shrunk = (n * rate + k * lg) / (n + k)
```

With:
```python
k_eff = k_base * reliability_factor(n, metric)
shrunk = (n * rate + k_eff * lg) / (n + k_eff)
```

Where `reliability_factor` adjusts `k_base` based on how much information `n` actually carries for that metric.

### Option A: Constant k (current — baseline)

```
k_eff = k_base (fixed)
```

Pros: Simple, predictable. Cons: Treats 50-PA call-up and 600-PA veteran identically in terms of prior weight (though the math does work — the veteran's `n` dominates).

Actually, re-examining: the current system already does this correctly. With `k=100`:
- 50 PA player: 67% regressed → appropriate
- 200 PA player: 33% regressed → reasonable  
- 600 PA player: 14% regressed → mostly raw

The math is sound. The **real** issue is the **asymmetry**: pitchers not shrunk at all on K%/BB%.

### Option B: Add Pitcher Overall Shrinkage (targeted)

Apply shrinkage to pitcher overall rates too, but with a **larger k denominator** (less aggressive) to preserve spread:

```python
PITCHER_K_SHRINK_K = 40.0    # lighter than batter's 100
PITCHER_BB_SHRINK_K = 40.0   # lighter than batter's 75
```

This would give:
- Pitcher with 80 PA: `(80r + 40*lg) / 120 = 0.67r + 0.33*lg` (mild regression)
- Pitcher with 400 PA: `(400r + 40*lg) / 440 = 0.91r + 0.09*lg` (mostly raw)
- Pitcher with 2000+ PA (multi-season blend): essentially raw

The key constraint: the spread diagnostic is currently at 0.59 for K% (hedging). Adding pitcher shrinkage would **increase** hedging unless we simultaneously relax batter shrinkage or the soft_log5 alpha.

### Option C: Differential Shrinkage by Effective PA (recommended)

The multi-season blend means `n_pa_w` already accounts for decay (prior-season rows have weight 0.5 and 0.25). But the effective sample sizes vary wildly:

| Player type | Typical n_pa_w | Current shrinkage @ k=100 | 
|-------------|----------------|---------------------------|
| Full-season + 2 priors | 500-800 | 12-17% regressed |
| Full-season only | 250-400 | 20-29% regressed |
| Partial season (injury return) | 80-150 | 40-56% regressed |
| September call-up / new player | 20-60 | 63-83% regressed |
| Per-pitch cell (sparse pitch type) | 5-30 | 77-95% regressed |

This is already producing reasonable behavior for batter rates. The gains come from:

1. **Applying the same logic to pitcher overall rates** (Option B above)
2. **Varying k by confidence in the league prior** — for K%, the league prior is very stable (σ ≈ 0.02 year-to-year), so a strong k is fine. For hard-hit%, the league prior drifts more (ball changes), so k should be lighter.
3. **Per-pitch-type adaptive shrinkage** — pitches with tiny effective samples (< 30 weighted PA) should be shrunk harder than the main arsenal pitches.

## Implementation Plan

### Phase 1: Pitcher Rate Shrinkage (low risk, addresses asymmetry)

**Files:** `matchup.py`

1. Add constants:
```python
PITCHER_K_SHRINK_K = 50.0     # less aggressive than batter (100)
PITCHER_BB_SHRINK_K = 50.0    # less aggressive than batter (75)
```

2. In `project()`, shrink pitcher overall K% and BB% before `soft_log5`:
```python
p_pa_overall = float(pitcher_overall.get("n_pa_w", 0.0) or 0.0)
p_k_shrunk  = shrunk_rate(pitcher_overall["K_pct"],  p_pa_overall, LG_K_PCT,  PITCHER_K_SHRINK_K)
p_bb_shrunk = shrunk_rate(pitcher_overall["BB_pct"], p_pa_overall, LG_BB_PCT, PITCHER_BB_SHRINK_K)
proj_k  = soft_log5(b_k_shrunk, p_k_shrunk,  LG_K_PCT, K_PCT_ALPHA)
proj_bb = soft_log5(b_bb_shrunk, p_bb_shrunk, LG_BB_PCT, BB_PCT_ALPHA)
```

3. Simultaneously relax `K_PCT_ALPHA` from 0.8 → 1.0 (remove the soft_log5 damping that was compensating for the asymmetry).

**Validation:** Run the calibration sweep (or at minimum, regenerate the accuracy dashboard on existing dates) and check:
- Spread ratio for K% should move from 0.59 toward 0.85+
- Log-loss / Brier should not degrade
- RMSE should stay flat or improve

### Phase 2: Per-Pitch-Type Adaptive k (medium risk)

**Files:** `matchup.py`

Currently per-pitch xwOBA uses `XWOBA_SHRINK_K = 100` for all pitches regardless of how many PA the batter/pitcher has against that specific pitch type. A slider the batter has faced 200 times gets the same prior strength as a splitter they've seen 8 times.

1. Replace the fixed k with a function:
```python
def adaptive_shrink_k(n_pa: float, k_base: float, k_floor: float = 20.0) -> float:
    """Scale shrinkage so sparse cells get more prior, dense cells less.
    
    At n_pa = k_base, shrinkage is 50% (standard behavior).
    At n_pa < k_floor, cap at k_base * 2 (don't over-regress to meaninglessness).
    At n_pa > k_base * 3, reduce to k_base * 0.5 (trust the data).
    """
    if n_pa <= k_floor:
        return k_base * 2.0
    if n_pa >= k_base * 3:
        return k_base * 0.5
    return k_base
```

2. Apply in the per-pitch loop:
```python
k_b = adaptive_shrink_k(_n_pa_b, XWOBA_SHRINK_K)
k_p = adaptive_shrink_k(_n_pa_p, PITCHER_XWOBA_SHRINK_K)
b_x = shrunk_rate(b_x_raw, _n_pa_b, LG_XWOBA, k_b)
p_x = shrunk_rate(p_x_raw, _n_pa_p, LG_XWOBA, k_p)
```

**Validation:** Same as Phase 1, plus inspect the Waldschmidt-type cases (very few PA) to confirm they get heavier regression, and established players (high PA) maintain their edge.

### Phase 3: Calibration Sweep on k Values (optional, data-hungry)

Use `calibrate_mix_shift.py` as a template to sweep:
- `PITCHER_K_SHRINK_K` in {0, 25, 50, 75, 100}
- `K_PCT_ALPHA` in {0.7, 0.8, 0.9, 1.0}

Grid of 20 combinations evaluated on the backfill dates. Pick the Pareto-optimal point on (log-loss, spread-ratio-K%, RMSE-K%).

## Risk Assessment

| Phase | Risk | Mitigation |
|-------|------|------------|
| 1 | K% spread could over-correct (from hedging to overconfident) | Start with conservative k=50, monitor spread ratio target 0.85-1.00 |
| 1 | Log-loss regression if pitcher shrinkage removes real signal | Small k (50) preserves most pitcher signal for multi-season starters |
| 2 | Complexity for marginal gain on bulk population | Gate behind a constant; easy to revert to fixed k |
| 3 | Overfitting to 16-date sample | Report confidence intervals; require ≥2σ improvement to ship |

## Expected Outcomes

- **Phase 1:** Spread ratio K% moves from 0.59 → ~0.80-0.90. Low-PA pitchers (spot starters, call-ups) get appropriately regressed. Proper-score skill should hold or improve since we're reducing a systematic bias.
- **Phase 2:** Marginal gains on the extreme-decile pass rate (sparse per-pitch cells currently carry too much noise). Waldschmidt-type projections become properly uncertain rather than anchored to a thin sample.
- **Phase 3:** Empirically-grounded constants rather than hand-tuned values. Publishable to the calibration archive for future reference.

## Dependencies

- Accuracy backfill covering ≥10 dates (currently have 16 in the sweep — sufficient for Phase 1-2)
- `calibrate_mix_shift.py` as template for Phase 3 sweep infrastructure
- No external data dependencies — all changes are to constants and the `shrunk_rate` call sites
