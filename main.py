"""GTA VI trophy rarity forecast (gta6-trophy-rarity-forecast-01).

A program/v1 entry in Python 3.13 (standard library only). For each case it
reads every <id>.in in $MAKEFUN_INPUT_DIR (or /input) and writes one
<id>.out to $MAKEFUN_OUTPUT_DIR (or /output):

    {"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}

five quantiles (percent) of each trophy's Exophase earn rate at the case's
target, scored by mean pinball loss on the logit scale.

How Exophase's page behaves (from the kit's RULES.txt) drives the model:

* The page is FROZEN between Exophase's own refreshes: the house saw it
  unchanged, 638 players tracked and every rate the same, from launch night
  until the first refresh, which jumped players tracked to 3,868 and pushed
  every rate but the 100% one DOWN. A refresh dilutes rates: the tracked
  player count jumps by more than the earners do. So consecutive identical
  reads are one observation, not many: the program collapses the history
  into plateaus, one per refresh, each plateau's value being the true
  earn-rate ratio as of that refresh.
* The answer is the page as of its last refresh before the target, so the
  trend is evaluated not at the target but at the target minus half the
  measured refresh cadence (the expected lag since the last refresh).

Per trophy the model fits two shapes to the plateau values and blends
them half-and-half in logit space: (a) a recency-weighted least-squares
line of logit(rate) against ln(1 + hours since release), which tracks
continued growth; and (b) a saturating completion curve
r(h) = A * (1 - exp(-h / tau)), which is the shape a share of players
completing a task actually follows and stops the line from overshooting
once a rate is high. The median is the last plateau's rate moved along
the fitted slope, shrunk by how significant the slope is;
trophies with fewer than 3 plateaus, and trophies nobody has earned yet,
fall back to persistence (or a tier prior) with a wide spread. The spread
combines a random-walk term over the horizon, the fit's prediction
variance, refresh-timing uncertainty and a base model-error term.

Game structure is enforced on the medians: chapter trophies are
non-increasing along the story (Chapter 1 >= Chapter 2 >= ... >= final
mission), the platinum is at most the final mission and at most 100%
Completion, and 100% Completion is at most every other trophy -- each is
a theorem about who can hold which trophy, so it can only remove fit
noise.

Deterministic: pure float arithmetic, fixed file order, no randomness, no
network, no files besides the standard library. The output directory is
created with makedirs(exist_ok=True); the program never removes it or
anything in it.
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# --- model constants (fixed in advance, not tuned on the practice) ---
SIGMA_RW = 0.24       # logit random-walk spread per square-root day
SIGMA_MODEL = 0.28    # base model-misspecification spread, in logit units
SIGMA_FLOOR = 0.40    # narrowest spread we will ever claim (logit)
SIGMA_CAP = 1.40      # widest spread we will ever claim (logit)
SIGMA_ONE_PLATEAU = 1.10  # history is one frozen page: a refresh can
                          # dilute rates several-fold; stay wide
SIGMA_TWO_PLATEAU = 0.90
RESID_FLOOR = 0.15    # floor on the residual sd of the logit fit
HALF_LIFE_H = 24.0    # recency half-life for plateau weights, in hours
MAX_SLOPE = 3.0       # cap on the fitted logit slope per log-hour
MAX_RISE = 3.0        # cap on the upward logit move from the last read
MAX_FALL = 1.5        # cap on the downward logit move from the last read
# Prior median (percent) for a trophy nobody tracked has earned yet.
TIER_PRIOR = {"platinum": 0.005, "gold": 0.01, "silver": 0.03, "bronze": 0.05}
SIGMA_ZERO = 1.20     # spread for a trophy with zero earners so far

CHAPTER_RE = re.compile(r"^Complete Chapter (\d+)\.")


def parse_time(value):
    """ISO 8601 timestamp (or plain date) -> aware UTC datetime."""
    if value is None:
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid(lg):
    if lg >= 0:
        return 1.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return e / (1.0 + e)


def round4(x):
    """x rounded to 4 decimals, half up (as the starter rounds it)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def collapse_plateaus(history):
    """Collapse consecutive identical reads into one plateau each.

    Exophase's page does not change between refreshes, so a run of
    identical reads is a single observation of the true ratio as of that
    refresh. Returns [(slot_datetime, read), ...] in time order.
    """
    plateaus = []
    prev_key = None
    for read in history:
        rates = read.get("rates") or []
        key = (read.get("players_tracked"), tuple(rates))
        if key == prev_key:
            continue
        prev_key = key
        plateaus.append((parse_time(read.get("slot") or read.get("at")), read))
    return plateaus


def median_gap_hours(times):
    """Median gap between consecutive plateau starts, in hours (or None)."""
    gaps = []
    for a, b in zip(times, times[1:]):
        if a is not None and b is not None:
            gaps.append((b - a).total_seconds() / 3600.0)
    if not gaps:
        return None
    gaps.sort()
    mid = len(gaps) // 2
    if len(gaps) % 2:
        return gaps[mid]
    return 0.5 * (gaps[mid - 1] + gaps[mid])


def wls_line(xs, ys, ws):
    """Weighted least squares y ~ a + b*x. Returns (a, b, s, xbar, sxx, n_eff)."""
    w_sum = sum(ws)
    xbar = sum(w * x for w, x in zip(ws, xs)) / w_sum
    ybar = sum(w * y for w, y in zip(ws, ys)) / w_sum
    sxx = sum(w * (x - xbar) ** 2 for w, x in zip(ws, xs))
    sxy = sum(w * (x - xbar) * (y - ybar) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx if sxx > 0 else 0.0
    a = ybar - b * xbar
    m = len(xs)
    if m > 2:
        sse = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
        s = math.sqrt(sse / (m - 2))
    else:
        s = RESID_FLOOR
    s = max(s, RESID_FLOOR)
    n_eff = w_sum * w_sum / sum(w * w for w in ws) if ws else 0.0
    return a, b, s, xbar, sxx, n_eff


def saturating_median(obs, ws, h_eff):
    """Median logit from a saturating completion curve r(h) = A*(1-exp(-h/tau)).

    obs: [(x_logtime, y_logit, hours, rate_pct)] per plateau. tau is found
    by grid search, A closed-form by weighted least squares for that tau.
    Returns None when the fit cannot be trusted (too few distinct hours).
    """
    hours = [o[2] for o in obs]
    rates = [o[3] for o in obs]
    if len(set(hours)) < 3 or max(rates) <= 0:
        return None
    best = None
    # tau from 4 hours to 3000 hours, log-spaced.
    for j in range(40):
        tau = 4.0 * (750.0 ** (j / 39.0))
        svals = [1.0 - math.exp(-h / tau) for h in hours]
        den = sum(w * sv * sv for w, sv in zip(ws, svals))
        if den <= 0:
            continue
        num = sum(w * sv * r for w, sv, r in zip(ws, svals, rates))
        amp = num / den
        amp = max(max(rates), min(99.9, amp))
        sse = sum(w * (r - amp * sv) ** 2 for w, sv, r in zip(ws, svals, rates))
        if best is None or sse < best[0]:
            best = (sse, amp, tau)
    if best is None:
        return None
    _, amp, tau = best
    pred = amp * (1.0 - math.exp(-h_eff / tau))
    return logit(pred)


def forecast_case(case, name):
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    history = case.get("history") or []
    trophies = case.get("trophies") or []
    if not history:
        # No history at all: tier priors with a wide spread.
        rows = []
        for trophy in trophies:
            med = TIER_PRIOR.get(trophy.get("tier"), 0.05)
            rows.append((trophy["id"], logit(med), SIGMA_ZERO))
        return quantiles_from(rows)

    release = parse_time((case.get("game") or {}).get("release_date"))
    if release is None:
        release = parse_time(history[0].get("slot") or history[0].get("at"))
    target = parse_time(case.get("target"))
    horizon_days = (case.get("horizon_minutes") or 0) / 1440.0

    plateaus = collapse_plateaus(history)
    p_times = [t for t, _ in plateaus]
    cadence_h = median_gap_hours(p_times)
    hours_to_target = None
    if release is not None and target is not None:
        hours_to_target = (target - release).total_seconds() / 3600.0

    # Effective target: the page at the target shows the last refresh
    # before it, expected half a cadence earlier.
    eff_target = target
    if target is not None and cadence_h:
        lag_h = 0.5 * cadence_h
        last_t = p_times[-1] if p_times else None
        if last_t is not None:
            from datetime import timedelta

            eff = target - timedelta(hours=lag_h)
            if eff < last_t:
                eff = last_t
            eff_target = eff

    def x_of(moment):
        if release is None or moment is None:
            return 0.0
        hours = (moment - release).total_seconds() / 3600.0
        return math.log1p(max(0.0, hours))

    x_last = x_of(p_times[-1]) if p_times else 0.0
    x_eff = x_of(eff_target)

    # Per-trophy observations: (x, y_fit, hours) per plateau, where y_fit
    # is the continuity-corrected logit used for the regression. The
    # forecast itself is anchored on the last read's RAW rate (the
    # correction would otherwise bias small rates upward: 4/638 = 0.63%
    # becomes 4.5/639 = 0.70%).
    medians = []  # (id, median_logit, sigma)
    for i, trophy in enumerate(trophies):
        obs = []  # (x, y_fit, hours_since_release)
        last_raw = None  # last non-null raw rate (percent)
        all_zero = True
        for (pt, read) in plateaus:
            rates = read.get("rates") or []
            if i >= len(rates) or rates[i] is None:
                continue
            rate = rates[i]
            last_raw = rate
            n = read.get("players_tracked")
            if not isinstance(n, (int, float)) or n <= 0:
                n = 1000.0
            k = int(round(rate / 100.0 * n))
            if k > 0:
                all_zero = False
            y = math.log(((k + 0.5) / (n + 1.0)) / (1.0 - (k + 0.5) / (n + 1.0)))
            hours = (pt - release).total_seconds() / 3600.0 if (pt and release) else 0.0
            obs.append((x_of(pt), y, hours, rate))
        tier = trophy.get("tier") or "bronze"
        if not obs or last_raw is None:
            medians.append((trophy["id"], logit(TIER_PRIOR.get(tier, 0.05)), SIGMA_ZERO))
            continue
        if all_zero:
            medians.append((trophy["id"], logit(TIER_PRIOR.get(tier, 0.05)), SIGMA_ZERO))
            continue
        last_y = logit(last_raw)
        if len(obs) < 3:
            sigma = SIGMA_ONE_PLATEAU if len(obs) == 1 else SIGMA_TWO_PLATEAU
            medians.append((trophy["id"], last_y, sigma))
            continue
        t_last_h = obs[-1][2]
        xs = [o[0] for o in obs]
        ys = [o[1] for o in obs]
        ws = [0.5 ** ((t_last_h - o[2]) / HALF_LIFE_H) for o in obs]
        a, b, s, xbar, sxx, n_eff = wls_line(xs, ys, ws)
        b = max(-MAX_SLOPE, min(MAX_SLOPE, b))
        if sxx <= 1e-12:
            se_b = float("inf")
        else:
            se_b = s / math.sqrt(sxx)
        sig = abs(b) / max(se_b, 0.05)
        gain = sig / (sig + 1.5)
        move = gain * b * (x_eff - xs[-1])
        move = max(-MAX_FALL, min(MAX_RISE, move))
        median_lin = last_y + move
        # Saturating fit: a share of players completing a task follows a
        # saturating curve, r(h) = A * (1 - exp(-h / tau)); a straight line
        # in logit-vs-log-time keeps rising and overshoots once the rate
        # is high. Fit A and tau on the plateau rates (recency-weighted,
        # tau by grid search, A closed-form), read it at the effective
        # target, and blend the two medians half-and-half in logit space:
        # each model is right about a different shape of progress, and the
        # blend is wrong by at most half of either one's error.
        h_eff = max(0.0, (eff_target - release).total_seconds() / 3600.0) if (eff_target and release) else 0.0
        median_sat = saturating_median(obs, ws, h_eff)
        if median_sat is None:
            median_logit = median_lin
        else:
            median_logit = 0.5 * median_lin + 0.5 * median_sat
        # Spread: random walk over the horizon + fit prediction variance
        # + refresh-timing uncertainty + base model error.
        pred_var = s * s * (1.0 / max(n_eff, 1.0) + (x_eff - xbar) ** 2 / sxx) if sxx > 1e-12 else s * s
        refresh_var = 0.0
        if cadence_h and hours_to_target:
            # sd of the lag since the last refresh, in log-time units.
            sd_x = (cadence_h / math.sqrt(12.0)) / (1.0 + hours_to_target)
            refresh_var = (gain * abs(b) * sd_x) ** 2
        var = (SIGMA_RW ** 2) * horizon_days + pred_var + refresh_var + SIGMA_MODEL ** 2
        sigma = math.sqrt(var)
        sigma = max(SIGMA_FLOOR, min(SIGMA_CAP, sigma))
        medians.append((trophy["id"], median_logit, sigma))

    medians = apply_structure(medians, trophies)
    return quantiles_from(medians)


def apply_structure(medians, trophies):
    """Enforce game-structure inequalities on the medians (percent scale).

    Chapter trophies are non-increasing along the story; the platinum is
    at most the final mission and at most 100% Completion; 100% Completion
    is at most every other trophy. Operates on (id, median_logit, sigma)
    triples and returns them with medians adjusted.
    """
    med = {i: m for i, m, _ in medians}
    sig = {i: s for i, _, s in medians}

    def pct(i):
        return sigmoid(med[i]) * 100.0

    def set_pct(i, p):
        med[i] = logit(p)

    # Chapter chain: Chapter 1..N then the final mission.
    chain = []
    for t in trophies:
        m = CHAPTER_RE.match((t.get("description") or "").strip())
        if m:
            chain.append((int(m.group(1)), t["id"]))
    chain.sort()
    final_ids = [t["id"] for t in trophies
                 if (t.get("description") or "").strip() == "Complete the final mission."]
    chain_ids = [i for _, i in chain] + final_ids
    for prev_i, next_i in zip(chain_ids, chain_ids[1:]):
        if prev_i in med and next_i in med and pct(next_i) > pct(prev_i):
            set_pct(next_i, pct(prev_i))

    # 100% Completion is at most every other trophy.
    comp_ids = [t["id"] for t in trophies if "100% Completion" in (t.get("description") or "")]
    others = [i for i in med if i not in comp_ids]
    for c in comp_ids:
        if c in med and others:
            cap = min(pct(o) for o in others)
            if pct(c) > cap:
                set_pct(c, cap)

    # Platinum is at most the final mission and at most 100% Completion.
    plat_ids = [t["id"] for t in trophies if t.get("tier") == "platinum"]
    caps = [i for i in final_ids + comp_ids if i in med]
    for p in plat_ids:
        if p in med and caps:
            cap = min(pct(c) for c in caps)
            if pct(p) > cap:
                set_pct(p, cap)

    return [(i, med[i], sig[i]) for i, _, _ in medians]


def quantiles_from(medians):
    """(id, median_logit, sigma) -> {"trophies": [{"id", "q": [...]}]}."""
    rows = []
    for tid, m, sigma in medians:
        qs = []
        for z in Z:
            v = round4(clamp_pct(sigmoid(m + z * sigma) * 100.0))
            qs.append(v)
        # Rounding can only break monotonicity by a hair; repair it.
        for j in range(1, len(qs)):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": tid, "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
