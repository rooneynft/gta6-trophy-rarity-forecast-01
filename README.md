# GTA VI Trophy Rarity Forecast

[![make.fun](https://img.shields.io/badge/make.fun-gta6--trophy--rarity--forecast--01-blue)](https://make.fun/c/gta6-trophy-rarity-forecast-01)
[![rank](https://img.shields.io/badge/rank-not%20yet%20scored-lightgrey)](https://make.fun/c/gta6-trophy-rarity-forecast-01)
[![reward](https://img.shields.io/badge/reward-%24250%20%7C%2020%20places-green)](https://make.fun/c/gta6-trophy-rarity-forecast-01)

A program that forecasts every Grand Theft Auto VI PlayStation 5 trophy's
Exophase earn rate at 2026-11-26T00:00:00Z — a week after launch — from the
rates the house recorded up to the deadline, as five quantiles a trophy
(5/25/50/75/95%), for the lowest pinball loss on the logit scale.

## How it works

Python 3.13, standard library only. Offline, deterministic, no network.
Reads every `*.in` case in the input directory, writes one `<id>.out` per
case with exactly one key, `trophies`, one row per trophy, five
non-decreasing quantiles each, rounded to 4 decimals.

The model works per trophy on the **logit scale** (logit(p) = ln(p/(1-p))),
because an earn rate saturates below 100% — a linear forecast would happily
predict rates above it. Two regimes:

* **Rich history (3+ distinct reads — the final case).** A Theil–Sen line
  of logit(rate) on ln(1 + hours since release), fitted on the **last 24
  hours** of distinct reads and read at the target. Only the last day is
  fitted because a launch is not one regime: the first refresh dilutes
  every rate as a wave of new players arrives, then the rates climb as
  players progress, and a line over the whole history averages the dip
  with the climb and extrapolates far too low (that mistake cost the first
  version ~1.8 logit units on a 5-day synthetic test). Theil–Sen (median
  pairwise slope) keeps one outlier refresh from dragging the slope. The
  median is floored by the holders already counted — holders never
  un-earn — with the player count projected on the same log-time trend.
  The spread is the fit's own prediction interval at the target, plus the
  pooled refresh volatility and a random-walk guard (0.45 logit units per
  square-root day).
* **Flat history (fewer than 3 distinct reads — the practice cases).**
  No trend is estimable, so the forecast uses the refresh the contest's
  own rules describe: on launch night the page sat unchanged (638 players
  tracked) and then refreshed to 3,868 players with every rate but the one
  at 100% lower — a wave of new players who held about one trophy each,
  nearly all of them the already-universal first trophy. The forecast
  assumes the next refresh does the same: the player count multiplies
  ~6x, the new players hold the universal trophy and a sprinkle of the
  others, the sprinkle skewed toward the trophies that are already
  popular (concentration 2.0, spread 0.4).

Exophase refreshes hours apart, so between refreshes the page is frozen:
flat duplicate reads carry no information, and the fit uses only distinct
reads (one per refresh).

## Backtest

Walk-forward backtest on the practice cases (90-minute horizon), against
synthetic answers calibrated to the house's published practice figures
(persistence loss ~1.51, typical miss ~0.68 points, coverage 0), across
four answer families (proportional / early-trophy / uniform / late-trophy
skew):

| family | this entry | persistence | curve | AI proxy (tightest) |
|--------|-----------:|------------:|------:|--------------------:|
| prop   | 0.089 | 1.401 | 1.401 | 0.100 |
| early  | 0.081 | 1.423 | 1.423 | 0.110 |
| unif   | 0.101 | 1.387 | 1.387 | 0.110 |
| late   | 0.112 | 1.379 | 1.379 | 0.119 |

Mean loss 0.096, worst family 0.112 — it beats the persistence and curve
baselines by ~12x and the strongest AI-model proxy on every family. (The
AI proxy is the house model forecast as the rules describe it: median =
last rate diluted by the refresh, tight spread 0.5; the real house model
may differ.)

5-day horizon check (synthetic worlds with known ground truth, fast and
slow launches, checkpoint at the deadline, target 5 days out):

| world | this entry | persistence | curve | AI proxy |
|-------|-----------:|------------:|------:|---------:|
| fast  | 0.261 | 1.098 | 0.268 | 2.907 |
| slow  | 0.462 | 1.027 | 0.452 | 2.785 |

On the 5-day horizon it beats persistence and the AI proxy by a wide
margin and is at parity with the curve baseline (±0.01 in either world):
both share the log-time form, so the contest there is the spread, and
the spread here is calibrated to be competitive in either world rather
than tuned to one.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Runs in ~0.07 s for the two practice cases (limit 8 s). Two runs on the
same case write the same bytes.

## License

MIT. See [LICENSE](LICENSE).
