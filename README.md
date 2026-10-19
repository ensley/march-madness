# March Madness

[![PyPI - Version](https://img.shields.io/pypi/v/march-madness.svg)](https://pypi.org/project/march-madness)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/march-madness.svg)](https://pypi.org/project/march-madness)

-----

## Table of Contents

- [Installation](#installation)
- [Preparing game data](#preparing-game-data)
- [Historical backtesting](#historical-backtesting)
- [Model parameterization](#model-parameterization)
- [License](#license)

## Installation

```console
pip install march-madness
```

## Preparing game data

Convert raw scraped JSONL reports to one Parquet observation per game:

```console
march-madness prepare-games gamelogs.jsonl --division NCAAM --year 2026 --output data/games2026.parquet
march-madness prepare-games gamelogs_w.jsonl --division NCAAW --year 2026 --output data/games2026w.parquet
```

Game identity uses division, season, date, and the unordered pair of school IDs.
Names and scores are not identifiers. Reversed reports are merged after checking
scores, overtime, venue, and game type. Conflicts stop conversion and display both
reports; resolve them in the source data before retrying. Same-day rematches need
explicit game identities and are not supported by this date-based key.

School names prefer each school's own report, with whitespace and trailing NCAA
annotations normalized. Raw logs remain unchanged for auditing. Do not use
`set(gamelogs)` to deduplicate games: object equality represents identical reports,
not matching reports from opposing schools.

`fit-season` rejects repeated game identities in its selected input, including
old Parquet files. Rebuild affected datasets, then refit and rerun simulations;
previous posterior draws are not corrected by cleaning the input. Data and model
outputs are local files excluded from version control.

## Historical backtesting

Evaluate predictions against recorded NCAA tournament outcomes:

```console
march-madness backtest data/games.parquet --year 2022 --year 2023 --year 2024 --legacy-home-first --outdir output/backtest-2022-2024
```

Each season is trained on regular-season and conference-tournament games strictly
before its first NCAA tournament game. All ratings remain frozen during the
holdout, including Elo. No tournament scores enter training or tune baseline
constants. First Four games count toward evaluation.

Use `--legacy-home-first` only for older inputs where team 1 is home or the game
is neutral and `team2_home_adv` is absent. It sets that missing flag to zero,
not the complement of team 1's flag. Modern inputs already carry both flags.

The command compares Stan with fixed-parameter Elo and a 50/50 baseline. It saves
per-game forecasts, season and pooled log loss, Brier score, accuracy, calibration
bins with counts, and Stan margin errors per 40 minutes. Lower log loss and Brier
scores are better. Sampling warnings are included in the report; increase
`--samples` or `--warmup` if diagnostics are poor.

`--metric` records the Stan sampler metric (`diag_e` by default; `dense_e` is
available for diagnostic experiments). Changing it affects sampling, not the
model equations or holdout split.

Default sampling uses four chains, 1,000 warmup iterations and 4,000 retained
draws per chain. `--jobs 2` fits two seasons concurrently (eight chain processes).
Seeds, cutoffs, source/model hashes, training and holdout rows, and posterior
outputs are preserved. Use a new output directory for each run.

Choose years with recorded tournament results: 2020 has none, and the local 2025
file ends before the tournament finishes. Reports show actual game counts and
round coverage; an incomplete source year is not a complete-tournament evaluation.
Seed-based baselines require seed data, which the historical game file lacks.

## Model parameterization

In the centered parameterization, `mu_scoring` represents
`mean(mu_off) - mean(mu_def)`. The current independent standard-normal priors
give each mean variance `1 / T`, where `T` is the number of teams. Their
difference therefore has variance `2 / T`, so its standard deviation is
`sqrt(2.0 / T)`. Using `mu_scoring ~ normal(0, sqrt(2.0 / T))` preserves the
existing induced prior on overall scoring when removing the shared offense/defense
shift that the likelihood cannot identify.

The season, prediction, and bracket models share this parameterization. Refit
posterior CSVs from the earlier parameterization before using them with the
current prediction or bracket model; CmdStan requires matching parameter layouts.
Season fits save 18 significant digits so rounding does not break the sum-to-zero
constraints when posterior CSVs are reloaded.

## License

`march-madness` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
