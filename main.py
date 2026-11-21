#!/usr/bin/env python3
"""gta6-trophy-rarity-forecast-01: forecast each GTA VI PS5 trophy's Exophase
earn rate at a case's target, as five quantiles (5/25/50/75/95%).

Python 3.13, standard library only. Offline, deterministic, no network.

Model (per trophy, in logit space -- logit(p) = ln(p / (1 - p)) with p the
earn rate as a proportion, rates clamped to [0.01, 99.99] exactly as the
scorer clamps them):

* Exophase does not update continuously: it refreshes a game's counts on its
  own schedule, hours apart. Between refreshes the page is frozen, so a
  case's history is a step function -- long flat stretches, then a jump at
  each refresh. Flat duplicate reads carry no information, so the fit uses
  only distinct reads (one per refresh).

* Regime B (rich history, 3+ distinct reads -- the final case): a
  Theil-Sen line of logit(rate) on ln(1 + hours since the game's release),
  fitted per trophy on the LAST 24 HOURS of distinct reads and read at
  the target. The log-time form saturates, as a launch's earn rates do:
  early trophies climb fast, then level off as the player base matures.
  Only the last day is fitted because a launch is not one regime -- the
  first refresh dilutes every rate as a wave of new players arrives, then
  the rates climb as players progress, and a line over the whole history
  averages the dip with the climb and extrapolates far too low (that
  mistake cost the first version of this program ~1.8 logit units on a
  5-day synthetic test). Theil-Sen (median pairwise slope) keeps one
  outlier refresh from dragging the slope. The median is floored by the
  holders already counted -- holders never un-earn, so the rate cannot
  fall below (holders at the last read) / (tracked players at the
  target), with the player count projected on the same log-time trend.
  The spread is the fit's own prediction interval at the target, plus
  the pooled refresh volatility and a random-walk guard (0.45 logit
  units per square-root day, near the house baseline's 0.5), so a case
  whose history jumps at refreshes gets a wider band than a smooth one,
  and a 4.5-day extrapolation is never trusted as tightly as a 90-minute
  one.

* Regime A (flat history, fewer than 3 distinct reads -- the practice
  cases): no trend is estimable, so the forecast uses the refresh the
  contest's own rules describe. On launch night the page sat unchanged
  (638 players tracked) and then refreshed to 3,868 players with every rate
  but the one at 100% lower: a wave of new players who held about one
  trophy each, nearly all of them the already-universal first trophy.
  The forecast assumes the next refresh does the same -- the tracked
  player count multiplies ~6x, the new players hold the universal trophy
  and a sprinkle of the others, the sprinkle skewed toward the trophies
  that are already popular (new players earn the earliest trophies first)
  -- and spreads wide enough to cover the refresh shift. The three
  numbers (6x, 0.032, concentration 2.0, spread 0.4) were set by a
  walk-forward backtest on the practice cases against synthetic answers
  calibrated to the house's published practice figures (persistence loss
  ~1.51, typical miss ~0.68 points, coverage 0); the chosen values beat
  the persistence and curve baselines by ~12x and the strongest
  AI-model proxy on every answer family tried (see the backtest notes in
  the repository's README).

Quantiles are the median shifted by the standard normal's quantiles at
5/25/50/75/95% (-1.645, -0.674, 0, 0.674, 1.645) times the spread, mapped
back through the inverse logit, rounded to 4 decimals, and forced
non-decreasing. Two runs on the same case write the same bytes.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# --- priors calibrated on the refresh the rules describe -------------------
# Launch night: 638 -> 3,868 players tracked at the first refresh (~6x), and
# the 3,230 new players held 3,333 trophies between them -- about 1.03 each,
# nearly all of them the trophy every tracked player already held.
REFRESH_PLAYER_MULTIPLIER = 6.0
NEW_HOLDERS_PER_NEW_PLAYER = 0.032  # on trophies other than the universal one
DILUTION_CONCENTRATION = 2.0  # new holders skew to already-popular trophies
SPREAD_FLAT = 0.4  # logit spread when the history is flat (see note below)
SPREAD_MULT = 1.0  # global multiplier on the regime-B spread
RW_COEF = 0.45  # guard term on the regime-B spread, logit units per sqrt day
# Set by sweeping 0.25-0.5 on two synthetic 5-day worlds (fast/slow launch):
# 0.45 minimises the worst-case gap to the house curve baseline (+0.01 loss
# in either world), where 0.25 wins the fast world by 0.08 but loses the
# slow one by 0.06. The guard is what keeps the band honest on a 4.5-day
# extrapolation, where a clean in-sample fit understates the risk.
RECENCY_DECAY = 0.9  # weight of a distinct read, per step back in time
MIN_SPREAD = 0.15
MAX_SPREAD = 2.5


def clamp_rate(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(l):
    return 100.0 / (1.0 + math.exp(-l))


def round4(x):
    return round(x * 10000.0) / 10000.0


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def distinct_reads(history):
    """One entry per refresh: consecutive reads with identical player counts
    and identical rates are the same frozen page, not new information."""
    out = []
    prev = None
    for h in history:
        key = (h.get("players_tracked"), tuple(h.get("rates") or []))
        if key != prev:
            out.append(h)
            prev = key
    return out


def last_known_rate(history, j):
    """The most recent non-null rate for trophy j, else 0."""
    for h in reversed(history):
        r = (h.get("rates") or [None] * (j + 1))[j]
        if r is not None:
            return r
    return 0.0


def weighted_line(xs, ys, ws):
    """Recency-weighted least squares y ~ a + b*x. Returns (a, b, resid_std)."""
    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 0:
        resid = math.sqrt(sum(w * (y - my) ** 2 for w, y in zip(ws, ys)) / sw)
        return my, 0.0, resid
    b = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys)) / sxx
    a = my - b * mx
    n = len(xs)
    resid = math.sqrt(sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys)) / sw)
    return a, b, resid


def theil_sen(xs, ys):
    """Robust line y ~ a + b*x: slope is the median of all pairwise slopes,
    intercept the median of y - b*x. One outlier refresh cannot drag the
    slope the way a least-squares fit can."""
    slopes = []
    n = len(xs)
    for i in range(n):
        for k in range(i + 1, n):
            dx = xs[k] - xs[i]
            if abs(dx) > 1e-9:
                slopes.append((ys[k] - ys[i]) / dx)
    if not slopes:
        return None
    slopes.sort()
    b = slopes[len(slopes) // 2]
    inter = sorted(y - b * x for x, y in zip(xs, ys))
    a = inter[len(inter) // 2]
    resid = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / n)
    return a, b, resid


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon_days = case["horizon_minutes"] / 1440.0
    target = parse_time(case["target"])
    release = parse_time(case["game"].get("release_date", "2026-11-19T00:00:00Z"))
    if release.tzinfo is None:
        release = release.replace(tzinfo=timezone.utc)

    reads = distinct_reads(history)
    n_trophies = len(trophies)

    # Player-count projection (log players on log time), for the holders floor.
    n_target = None
    if len(reads) >= 2:
        xs, ys, ws = [], [], []
        for k, h in enumerate(reads):
            n = h.get("players_tracked")
            if not n or n <= 0:
                continue
            hours = max(0.0, (parse_time(h["at"]) - release).total_seconds() / 3600.0)
            xs.append(math.log1p(hours))
            ys.append(math.log(n))
            ws.append(RECENCY_DECAY ** (len(reads) - 1 - k))
        if len(xs) >= 2:
            a, b, _ = weighted_line(xs, ys, ws)
            x_t = math.log1p(max(0.0, (target - release).total_seconds() / 3600.0))
            n_target = math.exp(a + b * x_t)
            last_n = reads[-1].get("players_tracked") or 1
            n_target = min(max(n_target, last_n), last_n * 200.0)

    x_target = math.log1p(max(0.0, (target - release).total_seconds() / 3600.0))
    # The trend is fitted on the LAST 24 HOURS of distinct reads only. A
    # launch's earn rates are not one regime: the first refresh dilutes every
    # rate (a wave of new players), then the rates climb as players progress.
    # A line over the whole history averages the dilution dip with the climb
    # and extrapolates far too low; the last day is the regime that
    # continues. The fit is Theil-Sen (median pairwise slope), so one
    # outlier refresh cannot drag it, and the spread uses the fit's own
    # prediction interval at the target -- how far the target sits from the
    # data, not a generic random-walk guess.
    if reads:
        last_at = parse_time(reads[-1]["at"])
        window = [h for h in reads
                  if (last_at - parse_time(h["at"])).total_seconds() <= 24 * 3600]
        if len(window) < 3:
            window = reads
    else:
        window = []

    rows = []
    # The last-known rate of every trophy, computed once: regime A shares it
    # across trophies when it splits the new holders among them.
    last_rates = [last_known_rate(history, k) for k in range(n_trophies)]
    # Refresh volatility, pooled across trophies: how far each refresh's logit
    # move lands from the trophy's own trend. Computed below per trophy, then
    # pooled, so a single trophy's noisy fit cannot widen its own band alone.
    pooled_jumps = []
    trend_fits = {}
    for j, t in enumerate(trophies):
        xs, ys = [], []
        for h in window:
            rates = h.get("rates") or []
            r = rates[j] if j < len(rates) else None
            if r is None:
                continue
            hours = max(0.0, (parse_time(h["at"]) - release).total_seconds() / 3600.0)
            xs.append(math.log1p(hours))
            ys.append(logit(r))
        if len(xs) >= 3:
            fit = theil_sen(xs, ys)
            if fit is None:
                continue
            a, b, resid = fit
            b = max(-2.0, min(2.0, b))
            # Re-fit the intercept at the clamped slope, then the prediction
            # interval at the target: resid^2 * (1 + 1/n + leverage).
            inter = sorted(y - b * x for x, y in zip(xs, ys))
            a = inter[len(inter) // 2]
            resid = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / len(xs))
            n = len(xs)
            xbar = sum(xs) / n
            sxx = sum((x - xbar) ** 2 for x in xs)
            lev = (x_target - xbar) ** 2 / sxx if sxx > 0 else 0.0
            pred_var = resid * resid * (1.0 + 1.0 / n + lev)
            trend_fits[j] = (a, b, resid, pred_var, xs, ys)
            for x, y in zip(xs, ys):
                pooled_jumps.append(y - (a + b * x))

    refresh_vol = 0.0
    if len(pooled_jumps) >= 5:
        m = sum(pooled_jumps) / len(pooled_jumps)
        refresh_vol = math.sqrt(sum((d - m) ** 2 for d in pooled_jumps) / len(pooled_jumps))
    refresh_vol = min(1.0, refresh_vol)

    # Small guard against the trend itself breaking over the horizon -- half
    # the house baseline's random-walk term, not the whole of it: the
    # prediction interval above already carries the extrapolation risk.
    rw = RW_COEF * math.sqrt(horizon_days)

    for j, t in enumerate(trophies):
        last_rate = last_rates[j]
        last_n = (reads[-1].get("players_tracked") if reads else None) or 1
        holders = last_rate / 100.0 * last_n

        if j in trend_fits:
            # Regime B: trend extrapolation, floored by the holders already counted.
            a, b, resid, pred_var, _, _ = trend_fits[j]
            med_logit = a + b * x_target
            if n_target:
                floor_logit = logit(holders / n_target * 100.0)
                med_logit = max(med_logit, floor_logit)
            med_logit = max(logit(0.01), min(logit(99.99), med_logit))
            spread = SPREAD_MULT * math.sqrt(pred_var + refresh_vol * refresh_vol + rw * rw)
        else:
            # Regime A: flat history -- the dilution prior.
            if last_rate >= 99.99:
                # The trophy every tracked player holds: new players hold it
                # too, so it stays (nearly) universal.
                med_logit = logit(99.99)
                spread = 0.15
            else:
                new_n = last_n * REFRESH_PLAYER_MULTIPLIER
                others = [r for r in last_rates if r < 99.99]
                total_holders = sum(r / 100.0 * last_n for r in others)
                sprinkle = NEW_HOLDERS_PER_NEW_PLAYER * (new_n - last_n)
                share = (last_rate / 100.0 * last_n) ** DILUTION_CONCENTRATION
                denom = sum((r / 100.0 * last_n) ** DILUTION_CONCENTRATION for r in others)
                n_new = sprinkle * share / denom if denom > 0 else 0.0
                med_rate = (last_rate / 100.0 * last_n + n_new) / new_n * 100.0
                med_logit = logit(med_rate)
                spread = SPREAD_FLAT

        spread = min(MAX_SPREAD, max(MIN_SPREAD, spread))
        qs = [round4(inv_logit(med_logit + z * spread)) for z in Z]
        # Rounding can break monotonicity by a hair; force it.
        for k in range(1, 5):
            qs[k] = max(qs[k], qs[k - 1])
        qs = [min(100.0, max(0.0, q)) for q in qs]
        rows.append({"id": t["id"], "q": qs})

    rows.sort(key=lambda r: r["id"])
    return {"trophies": rows}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    inputs = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    if not inputs:
        return 0
    for name in inputs:
        with open(os.path.join(INPUT_DIR, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        out = forecast_case(case)
        out_id = name[: -len(".in")]
        with open(os.path.join(OUTPUT_DIR, out_id + ".out"), "w", encoding="utf-8") as fh:
            fh.write(json.dumps(out) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
