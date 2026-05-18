# Model Assessment — May 2026

Evaluation of the matchup projection model based on committed accuracy artifacts, code review, and calibration sweep results.

---

## 1) Issues

### Calibration & Scoring

#### Hedging on K% and Hard-Hit%

The spread diagnostic shows K% at **0.59** and hard-hit% at **0.68** variance ratio — both firmly in "hedging" territory. Projections are too tight around the mean for these metrics. The `soft_log5` damping with `K_PCT_ALPHA=0.8` and `HARD_HIT_ALPHA=0.8` deliberately pulls extremes toward center, which helps avoid tail blowups but costs discrimination on the players who *are* genuinely extreme.

#### Tiny proper-score skill margin

Log-loss skill of **0.7%** and Brier skill of **0.5%** over the league prior is real but razor-thin. With only ~780 scored PAs in the dashboard window, that's within the range where a few bad days could flip the sign. The model is marginally better than "just use league-average outcome rates for everyone."

#### Asymmetric shrinkage (pitcher vs batter)

Batter K%/BB% are empirical-Bayes shrunk toward league, but pitcher overall rates are **not** shrunk. The commit messages note this was intentional (shrinking pitchers hurt spread), but it creates a structural asymmetry — elite pitchers' rates are taken at face value while elite batters are pulled toward the mean. This may explain some of the systematic K% hedging.

#### Small evaluation sample

The committed accuracy dashboard covers only 2 slate dates (780 PAs, 202 hitter-games). The calibration sweep has more history (13.5k PAs across 16 dates), but that's still a small-ish sample for the number of parameters and mechanisms being evaluated. The Spearman ρ diagnostic only has **1 qualifying date** (≥30 hitters) showing ρ = 0.204 for xwOBA — directionally correct but not strong.

### Structural

#### Independence assumption in Multi-PA outlook

PAs are treated as independent draws — `P(≥1 in N) = 1 − (1−p)^N`. In reality, if a pitcher is locating well in PA 1, PA 2 is positively correlated. This inflates the multi-PA "chance ≥1" numbers slightly for low-probability outcomes (HR, 2B).

#### Outcome distribution reconciliation via clipping

When the reconstructed wOBA from outcome shares drifts >5 pts from the headline projection, hit probabilities are scaled by a factor clamped to [0.5, 2.0]. This is a brute-force patch rather than a jointly coherent decomposition — it can create discontinuities where a small input change flips the reconciliation behavior.

#### League baselines are hardcoded to 2025

`LG_XWOBA`, `LG_K_PCT`, etc. are sourced from 2025 and won't update until manually changed. If 2026 run environment drifts (e.g., the ball changes), all projections carry a systematic bias until updated.

#### 191 missing PAs in the α>0 sweep

The calibration sweep loses 191 PAs whenever α>0 due to join gaps in the sandbox. This means the α=0 baseline isn't strictly comparable to the others on the same population — a subtle confound.

---

## 2) Areas for Improvement

### High Impact (likely to move accuracy metrics)

#### Increase shrinkage differentiation by sample size

Rather than fixed `k` values (e.g., `XWOBA_SHRINK_K=100`), consider per-player adaptive shrinkage based on their actual effective sample size. A September call-up with 50 PAs should be shrunk much harder than a full-season regular with 600+ PAs. This would help the Waldschmidt-type cases where you have minimal data.

#### Platoon-aware park factors

Currently a single `park_pf` scalar is applied to xwOBA. Park effects are asymmetric by handedness and batted-ball type (Coors inflates FB power more than GB; Crawford Boxes at Minute Maid favor LHB pull). A per-handedness park factor, even a simple one, would be higher-signal.

#### Pitcher fatigue / velocity decay model

TTO gives a coarse 3-bucket degradation curve but doesn't model in-game velocity decline. Statcast has `release_speed` per pitch — could compute a per-pitcher velocity decay slope and use it to adjust the TTO projections beyond the empirical xwOBA delta.

#### Catcher framing adjustment

Framing adds 3-5% to a pitcher's called-strike rate. With Statcast's `zone` and `description` fields, could estimate a per-catcher framing effect and adjust K%/BB% at the matchup level. This is one of the largest documented non-modeled effects.

### Medium Impact

#### Widen the proper-scoring evaluation window

16 dates in the α sweep and 2 in the dashboard isn't enough to confidently tune parameters. Building a full-season backtest harness (even just May-through-September on 2025 data, which is cached) would let you measure calibration, discrimination, and spread ratio with much tighter confidence intervals.

#### Blend shape comps into headline (carefully)

Shape comps, zone overlay, and bat-tracking are computed but purely narrative — they don't influence the projection number. If the batter has 0 comps against a pitcher's primary pitch shape (confidence = "no comps"), that's a real information gap that should **widen** the uncertainty or pull the projection toward league. Consider at minimum using comp confidence as a shrinkage multiplier on the per-pitch projection.

#### Count-state xwOBA blend may be aggressive

`COUNT_XWOBA_BLEND_ALPHA = 0.5` is a strong weight on the count-distribution interaction. With the spread diagnostic showing hedging, it might be worth testing lower values (0.3, 0.2) to see if they reduce RMSE without sacrificing calibration.

#### Joint outcome model

Instead of projecting K/BB/HBP separately via log5, then deriving hits from xBA, then reconciling with a clamp — consider building a single Dirichlet-multinomial or softmax outcome model that enforces the constraint that outcomes sum to 1 throughout, not as a post-hoc fix.

### Lower Priority / Polish

- **Umpire zone**: Not modeled. Umpire assignments are public pre-game; even a 2-tier split (wide/tight zone) could adjust K%/BB% by 1-2%.
- **Weather/temperature**: A seasonal temperature curve applied to HR% (cold = ~-20% HR rate) would be simple and directionally correct.
- **Prior-season weight tuning**: `[1.0, 0.5, 0.25]` is reasonable but arbitrary. Could optimize these weights on the backtest harness alongside shrinkage `k`.
- **Recency half-life**: 30 days is a common choice but untested against alternatives (14, 45, 60). Worth sweeping if the backtest harness is built.

---

## Summary Table

| Category | Issue/Opportunity | Estimated Difficulty | Expected Impact |
|----------|------------------|---------------------|-----------------|
| Hedging on K%/HH% | Spread ratio 0.59–0.68 | Moderate (tune α/k) | Medium |
| Thin proper-score skill | 0.5–0.7% over prior | Structural | — (awareness) |
| Asymmetric shrinkage | Batter shrunk, pitcher not | Low (add pitcher shrink) | Medium |
| Adaptive sample-size shrinkage | Fixed k for all players | Moderate | High |
| Park factors by hand | Single scalar | Moderate | Medium-High |
| Catcher framing | Not modeled | Moderate | Medium |
| Backtest harness | Only 16 dates evaluated | Moderate (infra) | High (enables tuning) |
| Shape comps → headline blend | Diagnostic only | High (careful design) | Medium |
| Joint outcome model | Reconciliation clamp | High (refactor) | Medium |
| Umpire zone | Not modeled | Low-Moderate | Low-Medium |
| Weather/temperature | Not modeled | Low | Low |
| Season weight tuning | Arbitrary [1, 0.5, 0.25] | Low (sweep) | Low-Medium |
| Recency half-life | Untested alternatives | Low (sweep) | Low-Medium |

---

## Reference Data

**Dashboard (2026-05-17 window):** 780 PAs, 202 hitter-games, 2 slates  
**α sweep (2026-05-17):** 13,732 PAs, 3,567 hitters, 16 dates  
**Spearman ρ (xwOBA):** 0.204 (single qualifying date, 193 hitters)  

| α | logloss | brier | xwOBA RMSE (pts) | pass30 | ext pass |
|---|---------|-------|------------------|--------|----------|
| 0.00 | 1.4897 | 0.6957 | 204.25 | 0.115 | 0.100 |
| 0.25 | 1.4888 | 0.6951 | 204.76 | 0.116 | 0.110 |
| 0.50 | 1.4888 | 0.6951 | 204.73 | 0.116 | 0.109 |
| 1.00 | 1.4888 | 0.6951 | 204.69 | 0.116 | 0.105 |
| 2.00 | 1.4889 | 0.6951 | 204.63 | 0.116 | 0.108 |
