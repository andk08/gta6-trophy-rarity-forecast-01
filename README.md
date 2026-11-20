[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/andk08/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/andk08/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# GTA VI trophy rarity forecast

Entry for make.fun's **gta6-trophy-rarity-forecast-01** bounty: forecast, for
every PlayStation 5 trophy of Grand Theft Auto VI, the percent of the players
Exophase tracks who will have earned it at **2026-11-26T00:00:00Z**, as five
quantiles (0.05 / 0.25 / 0.5 / 0.75 / 0.95) per trophy, scored by mean pinball
loss on the logit scale.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` (Python 3.13, standard library only, no network) reads every
`<id>.in` in the input directory and writes one `<id>.out` per case. It runs
in ~0.1 s per case, is fully deterministic (same bytes on every run), and
never deletes the output directory — it only creates it and writes files
into it.

## Why this model

Exophase's page is **frozen between its own refreshes**: the house saw it
unchanged, 638 players tracked and every rate the same, from launch night
until the first refresh, which jumped players tracked to 3,868 and pushed
every rate but the 100% one *down*. A refresh dilutes rates — the tracked
player count jumps by more than the earners do. So:

1. **Plateaus, not reads.** Consecutive identical reads are collapsed into
   one observation per refresh. Each plateau's value is the true earn-rate
   ratio as of that refresh. Fitting the raw reads would treat one frozen
   page as hundreds of agreeing observations.
2. **Effective target.** The answer is the page as of its last refresh
   before the target, so the trend is evaluated at the target minus half
   the measured refresh cadence (the expected lag since the last refresh),
   not at the target itself.
3. **Two shapes, blended.** Per trophy, the median blends half-and-half in
   logit space:
   - a recency-weighted least-squares line of `logit(rate)` against
     `ln(1 + hours since release)`, shrunk by the slope's significance,
     which tracks continued growth; and
   - a saturating completion curve `r(h) = A · (1 − exp(−h/τ))` (τ by grid
     search, A closed-form), which is the shape a share of players
     completing a task actually follows and stops the line from
     overshooting once a rate is high.
4. **Honest spread.** The logit spread combines a random-walk term over
   the horizon, the fit's prediction variance, refresh-timing uncertainty,
   and a base model-error term. With only one plateau in the history (the
   practice cases) there is no trend to see, so the forecast persists the
   last rate with a wide spread (σ ≈ 1.1 logit) that admits a refresh
   diluting rates several-fold.
5. **Game structure.** Medians obey theorems about who can hold which
   trophy: chapter trophies are non-increasing along the story
   (Chapter 1 ≥ Chapter 2 ≥ … ≥ final mission), the platinum is at most
   the final mission and at most 100% Completion, and 100% Completion is
   at most every other trophy. Trophies nobody has earned yet get a
   tier-based prior (platinum 0.005%, gold 0.01%, silver 0.03%,
   bronze 0.05%) with a wide spread.

## Validation

No local grade exists for this program/v1 competition (the server grades
it), so the model was validated on synthetic final-like cases: a simulated
launch week with 8-hour refreshes, frozen pages between them, and trophy
progression under three different truth shapes (saturating exponential,
logistic in log-time, logit-linear in log-time). Against the house
baseline (persistence) and the house curve baseline (logit line on the
last 24 h), this entry scored:

| truth shape | this entry | house baseline | house curve |
|---|---|---|---|
| saturating exp | **0.131** | 0.397 | 0.270 |
| logistic | **0.134** | 0.378 | 0.260 |
| logit-linear | **0.152** | 0.484 | 0.230 |

(mean pinball loss on the logit scale; lower is better; typical miss
0.2–2.3 points vs 6–13 for the baselines). Edge cases (empty history,
all-zero trophies, null rates, single read, 6,510-minute horizon) all
produce valid output, and the grader's run was reproduced as user
`nobody` on a read-only copy: exit 0, byte-identical outputs across two
runs.
