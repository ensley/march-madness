"""Evaluate frozen pre-tournament forecasts on unseen NCAA tournament games."""

import hashlib
import json
import math
import pathlib
import shutil
from collections import defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import pairwise

import cmdstanpy
import numpy as np
import pandas as pd

from march_madness.ingest import validate_unique_games
from march_madness.transform import collect_season_outputs, prep_season_frame

TRAIN_TYPES = {"REG (Conf)", "REG (Non-Conf)", "CTOURN"}
TOURNAMENT_TYPES = {"FIRST-FOUR", "ROUND-64", "ROUND-32", "ROUND-16", "ROUND-8", "NATIONAL-SEMI", "NATIONAL-FINAL"}
ELO_START = 1500.0
ELO_K = 20.0
ELO_HOME = 100.0
ELO_SCALE = 400.0
RHAT_LIMIT = 1.01
ESS_LIMIT = 400
MAX_TREEDEPTH = 10
PROBABILITY_FLOOR = 1e-15
COIN_FLIP = 0.5
MIN_CHAINS = 2


@dataclass(frozen=True)
class TournamentSplit:
    year: int
    cutoff: pd.Timestamp
    train: pd.DataFrame
    test: pd.DataFrame


def tournament_split(
    games: pd.DataFrame, year: int, division: str, *, legacy_home_first: bool = False
) -> TournamentSplit:
    """Select one season and require every training game to precede the holdout."""
    required = {
        "division",
        "year",
        "date",
        "type",
        "team1_link",
        "team2_link",
        "team1_name",
        "team2_name",
        "team1_score",
        "team2_score",
        "team1_home_adv",
        "minutes_played",
    }
    missing = required - set(games.columns)
    if missing:
        msg = f"Missing game columns: {', '.join(sorted(missing))}."
        raise ValueError(msg)
    season = games.loc[
        (games.year == year) & (games.division == division) & games.type.isin(TRAIN_TYPES | TOURNAMENT_TYPES)
    ].copy()
    if season.empty:
        msg = f"No regular-season or NCAA tournament games found for {division} {year}."
        raise ValueError(msg)
    season["date"] = pd.to_datetime(season.date, errors="coerce")
    if season.date.isna().any():
        msg = f"Missing or invalid game dates in {division} {year}; chronological evaluation requires dates."
        raise ValueError(msg)
    if season.team1_link.isna().any():
        msg = f"Missing team1 identifiers in {division} {year}."
        raise ValueError(msg)
    season["team2_link"] = season.team2_link.fillna("other")
    if "team2_home_adv" not in season:
        if not legacy_home_first:
            msg = "Missing team2_home_adv; use --legacy-home-first only for home-first/neutral historical data."
            raise ValueError(msg)
        # In that legacy convention team 1 hosts, or both teams play neutrally.
        # A zero team-1 flag must not be interpreted as a team-2 home game.
        season["team2_home_adv"] = 0
    home = season[["team1_home_adv", "team2_home_adv"]]
    if not home.isin([0, 1]).all().all() or (home.sum(axis=1) > 1).any():
        msg = f"Invalid home/neutral flags in {division} {year}."
        raise ValueError(msg)
    scores = season[["team1_score", "team2_score"]]
    if scores.isna().any().any() or (scores < 0).any().any() or (season.team1_score == season.team2_score).any():
        msg = f"Invalid or tied final scores in {division} {year}."
        raise ValueError(msg)
    if season.minutes_played.isna().any() or (season.minutes_played <= 0).any():
        msg = f"Invalid game durations in {division} {year}."
        raise ValueError(msg)
    validate_unique_games(season)
    test = season.loc[season.type.isin(TOURNAMENT_TYPES)].sort_values(["date", "team1_link", "team2_link"])
    if test.empty:
        msg = f"No NCAA tournament games recorded for {division} {year}."
        raise ValueError(msg)
    cutoff = test.date.min().normalize()
    train = season.loc[season.type.isin(TRAIN_TYPES) & (season.date < cutoff)].sort_values(
        ["date", "team1_link", "team2_link"]
    )
    if train.empty:
        msg = f"No training games precede the {division} {year} tournament cutoff {cutoff.date()}."
        raise ValueError(msg)
    known = set(train.team1_link) | set(train.team2_link)
    unknown = (set(test.team1_link) | set(test.team2_link)) - known
    if unknown:
        msg = f"Tournament teams without pre-cutoff observations: {', '.join(sorted(unknown))}."
        raise ValueError(msg)
    return TournamentSplit(year, cutoff, train.reset_index(drop=True), test.reset_index(drop=True))


def elo_probabilities(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Fit a fixed K=20, home=100 Elo baseline, frozen before the tournament."""
    ratings: dict[str, float] = defaultdict(lambda: ELO_START)

    def probability(row):
        difference = ratings[row.team1_link] - ratings[row.team2_link]
        difference += ELO_HOME * (row.team1_home_adv - row.team2_home_adv)
        return 1 / (1 + 10 ** (-difference / ELO_SCALE))

    for row in train.sort_values(["date", "team1_link", "team2_link"]).itertuples():
        change = ELO_K * (int(row.team1_score > row.team2_score) - probability(row))
        ratings[row.team1_link] += change
        ratings[row.team2_link] -= change
    return np.array([probability(row) for row in test.itertuples()])


def posterior_predictions(fit, teams: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Integrate Normal win probabilities over draws, without simulating outcomes."""
    index = teams.set_index("link").tid - 1
    first = index.loc[test.team1_link].to_numpy(dtype=int)
    second = index.loc[test.team2_link].to_numpy(dtype=int)
    offense, defense, home = (fit.stan_variable(name) for name in ("mu_off", "mu_def", "mu_home"))
    sigma = fit.stan_variable("sigma")
    if not np.isfinite(sigma).all() or (sigma <= 0).any():
        msg = "Posterior residual standard deviations must be finite and positive."
        raise ValueError(msg)
    probabilities, margins = [], []
    for i, row in enumerate(test.itertuples()):
        a, b = first[i], second[i]
        difference = offense[:, a] + defense[:, a] - offense[:, b] - defense[:, b]
        difference += home[:, a] * row.team1_home_adv - home[:, b] * row.team2_home_adv
        # Difference of two independent Normal scores has SD sqrt(2)*sigma.
        z = difference / (2 * sigma)
        probability = np.fromiter((0.5 * math.erfc(-float(value)) for value in z), dtype=float)
        probabilities.append(probability.mean())
        margins.append(40 * difference.mean())
    return np.array(probabilities), np.array(margins)


def score_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Score the same held-out games for each model; lower proper scores are better."""
    rows = []
    for model, group in predictions.groupby("model", sort=True):
        probability = group.p_team1.to_numpy()
        outcome = group.team1_won.to_numpy(dtype=float)
        if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
            msg = f"Invalid probabilities from {model}."
            raise ValueError(msg)
        clipped = np.clip(probability, PROBABILITY_FLOOR, 1 - PROBABILITY_FLOOR)
        correct = np.where(probability == COIN_FLIP, COIN_FLIP, (probability > COIN_FLIP) == outcome)
        row = {
            "model": model,
            "games": len(group),
            "log_loss": float(-(outcome * np.log(clipped) + (1 - outcome) * np.log1p(-clipped)).mean()),
            "brier": float(((probability - outcome) ** 2).mean()),
            "accuracy": float(correct.mean()),
            "margin_mae_per40": None,
            "margin_rmse_per40": None,
        }
        if group.predicted_margin_per40.notna().all():
            residual = group.predicted_margin_per40 - group.actual_margin_per40
            row.update(
                margin_mae_per40=float(residual.abs().mean()), margin_rmse_per40=float(np.sqrt((residual**2).mean()))
            )
        rows.append(row)
    return pd.DataFrame(rows)


def calibration_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Keep counts with fixed-width reliability bins, including empty bins."""
    rows = []
    edges = np.linspace(0, 1, 11)
    for model, group in predictions.groupby("model", sort=True):
        buckets = np.minimum((group.p_team1.to_numpy() * 10).astype(int), 9)
        for bucket, (lower, upper) in enumerate(pairwise(edges)):
            selected = group.loc[buckets == bucket]
            rows.append(
                {
                    "model": model,
                    "lower": lower,
                    "upper": upper,
                    "games": len(selected),
                    "predicted": float(selected.p_team1.mean()) if len(selected) else None,
                    "observed": float(selected.team1_won.mean()) if len(selected) else None,
                }
            )
    return pd.DataFrame(rows)


def fit_season(
    data: dict, destination: pathlib.Path, *, seed: int, chains: int, samples: int, warmup: int, metric: str
):
    # Import only when fitting, so split/metric tests do not need a C++ toolchain.
    from march_madness.stan import Season

    cmdstanpy.write_stan_json(str(destination / "stan_data.json"), data)
    return Season.sample(
        data=str(destination / "stan_data.json"),
        seed=seed,
        chains=chains,
        parallel_chains=chains,
        iter_sampling=samples,
        iter_warmup=warmup,
        metric=metric,
        # Preserve sum-to-zero constraints when generated quantities reload the CSVs.
        sig_figs=18,
        output_dir=str(destination / "fit"),
        show_progress=False,
        refresh=500,
    )


def _evaluate_fold(
    split: TournamentSplit, destination: pathlib.Path, *, seed: int, chains: int, samples: int, warmup: int, metric: str
):
    destination.mkdir()
    split.train.to_parquet(destination / "training.parquet", index=False)
    split.test.to_parquet(destination / "holdout.parquet", index=False)
    _, teams, data = prep_season_frame(split.train, split.year)
    fit = fit_season(data, destination, seed=seed, chains=chains, samples=samples, warmup=warmup, metric=metric)
    collect_season_outputs(fit, teams).to_csv(destination / "results.csv", index_label="tid")
    posterior, margin = posterior_predictions(fit, teams, split.test)
    base = split.test[
        ["division", "year", "date", "type", "team1_link", "team2_link", "team1_name", "team2_name"]
    ].copy()
    base["cutoff"] = split.cutoff
    base["team1_won"] = (split.test.team1_score > split.test.team2_score).astype(int)
    base["actual_margin_per40"] = 40 * (split.test.team1_score - split.test.team2_score) / split.test.minutes_played
    parts = []
    for model, probability in [
        ("stan", posterior),
        ("elo", elo_probabilities(split.train, split.test)),
        ("coin_flip", np.full(len(base), 0.5)),
    ]:
        part = base.copy()
        part["model"] = model
        part["p_team1"] = probability
        part["predicted_margin_per40"] = margin if model == "stan" else np.nan
        parts.append(part)
    predictions = pd.concat(parts, ignore_index=True)
    predictions.to_csv(destination / "predictions.csv", index=False)
    summary = fit.summary()
    summary.to_csv(destination / "posterior_summary.csv")
    parameters = summary.loc[~summary.index.str.endswith("__")]
    methods = fit.method_variables()
    diagnostics = {
        "divergences": int(methods["divergent__"].sum()),
        "max_treedepth_hits": int((methods["treedepth__"] >= MAX_TREEDEPTH).sum()),
        "max_rhat": float(parameters.R_hat.max()),
        "min_bulk_ess": float(parameters.ESS_bulk.min()),
    }
    warnings = []
    if diagnostics["divergences"] or diagnostics["max_treedepth_hits"]:
        warnings.append("Sampler divergences or maximum tree-depth hits; inspect diagnostics.")
    if diagnostics["max_rhat"] > RHAT_LIMIT or diagnostics["min_bulk_ess"] < ESS_LIMIT:
        warnings.append("R-hat exceeds 1.01 or bulk ESS is below 400; increase sampling before trusting comparisons.")
    metadata = {
        "year": split.year,
        "cutoff": str(split.cutoff.date()),
        "training_games": len(split.train),
        "test_games": len(split.test),
        "last_test_date": str(split.test.date.max().date()),
        "rounds": split.test.type.value_counts().to_dict(),
        "seed": seed,
        "diagnostics": diagnostics,
        "warnings": warnings,
    }
    (destination / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return predictions, metadata


def run_backtest(
    source: str | pathlib.Path,
    destination: str | pathlib.Path,
    *,
    years: Sequence[int],
    division: str = "NCAAM",
    legacy_home_first: bool = False,
    samples: int = 4000,
    warmup: int = 1000,
    chains: int = 4,
    seed: int = 20261018,
    jobs: int = 1,
    metric: str = "diag_e",
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Refit separate seasons, preserving every cutoff, prediction, and diagnostic."""
    if not years or samples < 1 or warmup < 1 or chains < MIN_CHAINS or jobs < 1:
        msg = "Supply years, positive sampling/warmup/jobs, and at least two chains."
        raise ValueError(msg)
    if metric not in ("diag_e", "dense_e"):
        msg = "Sampler metric must be 'diag_e' or 'dense_e'."
        raise ValueError(msg)
    source, destination = pathlib.Path(source), pathlib.Path(destination)
    games = pd.read_parquet(source)
    splits = [
        tournament_split(games, year, division, legacy_home_first=legacy_home_first) for year in sorted(set(years))
    ]
    if destination.exists() and any(destination.iterdir()):
        msg = f"Output directory is not empty: {destination}. Choose a new directory to preserve previous runs."
        raise ValueError(msg)
    destination.mkdir(parents=True, exist_ok=True)
    model_source = destination / "model_source"
    model_source.mkdir()
    for filename in ["season.stan", "functions.stan"]:
        shutil.copyfile(pathlib.Path(__file__).parent / "stan" / filename, model_source / filename)
    manifest = {
        "status": "running",
        "source": str(source.resolve()),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "division": division,
        "years": [split.year for split in splits],
        "legacy_home_first": legacy_home_first,
        "samples": samples,
        "warmup": warmup,
        "chains": chains,
        "seed": seed,
        "jobs": jobs,
        "metric": metric,
        "sig_figs": 18,
        "model_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in model_source.iterdir()},
        "elo": {"initial": ELO_START, "k": ELO_K, "home": ELO_HOME, "scale": ELO_SCALE},
        "folds": [],
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    predictions = []
    try:
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = {}
            for split in splits:
                if progress:
                    progress(
                        f"{split.year}: {len(split.train)} training games before {split.cutoff.date()}; {len(split.test)} held-out games."
                    )
                future = executor.submit(
                    _evaluate_fold,
                    split,
                    destination / str(split.year),
                    seed=(seed + split.year) % (2**32 - 1) + 1,
                    chains=chains,
                    samples=samples,
                    warmup=warmup,
                    metric=metric,
                )
                futures[future] = split.year
            for future in as_completed(futures):
                frame, metadata = future.result()
                predictions.append(frame)
                manifest["folds"].append(metadata)
                manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
                if progress:
                    progress(f"{metadata['year']}: complete. {' '.join(metadata['warnings'])}")
        combined = pd.concat(predictions, ignore_index=True).sort_values(
            ["year", "model", "date", "team1_link", "team2_link"]
        )
        combined.to_csv(destination / "predictions.csv", index=False)
        rows = [score_predictions(group).assign(year=year) for year, group in combined.groupby("year")]
        rows.append(score_predictions(combined).assign(year="pooled"))
        metrics = pd.concat(rows, ignore_index=True)
        metrics.to_csv(destination / "metrics.csv", index=False)
        calibration_table(combined).to_csv(destination / "calibration.csv", index=False)
        manifest["folds"].sort(key=lambda fold: fold["year"])
        manifest["status"] = "complete"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        _write_report(destination, manifest, metrics)
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        raise
    else:
        return metrics


def _write_report(destination: pathlib.Path, manifest: dict, metrics: pd.DataFrame) -> None:
    lines = [
        "# Historical NCAA tournament backtest",
        "",
        "Each season is fit only on regular-season and conference-tournament games strictly before the first recorded NCAA tournament game. Ratings stay frozen throughout the tournament; test outcomes never update any model.",
        "",
        "## Accuracy",
        "",
        "Lower log loss and Brier scores are better. Log loss uses natural logarithms. An exact 50/50 forecast receives half credit for accuracy.",
        "",
        "| Season | Model | Games | Log loss | Brier | Accuracy | Margin MAE / 40 min |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in metrics.itertuples():
        margin = f"{row.margin_mae_per40:.2f}" if pd.notna(row.margin_mae_per40) else "—"
        lines.append(
            f"| {row.year} | {row.model} | {row.games} | {row.log_loss:.4f} | {row.brier:.4f} | {row.accuracy:.1%} | {margin} |"
        )
    lines += ["", "## Data and diagnostics", ""]
    for fold in manifest["folds"]:
        d = fold["diagnostics"]
        lines.append(
            f"- {fold['year']}: cutoff {fold['cutoff']}; {fold['training_games']} training games, {fold['test_games']} tournament games recorded through {fold['last_test_date']}. "
            f"Divergences: {d['divergences']}; tree-depth hits: {d['max_treedepth_hits']}; max R-hat: {d['max_rhat']:.4f}; min bulk ESS: {d['min_bulk_ess']:.0f}."
        )
        lines.extend(f"  - **Sampling warning:** {warning}" for warning in fold["warnings"])
    lines += [
        "",
        "## Interpretation and reproducibility",
        "",
        "- Elo starts every team at 1500 each season, with K=20, a 100-point home adjustment, and the usual 400-point logistic scale. Constants are fixed in advance, not tuned against these holdouts.",
        "- No seed baseline is available because the game dataset does not contain tournament seeds.",
        "- Probabilities average the Normal score-difference CDF over posterior draws. Margin errors compare predicted and observed points per 40 minutes, including duration adjustment for overtime.",
        "- Calibration bins retain their sample counts in calibration.csv; sparse bins are noisy. A few tournaments do not establish universal calibration or superiority.",
        "- The evaluation includes every recorded NCAA tournament game, including First Four. Round counts and the last recorded date are saved in each manifest; incomplete source seasons are not presented as complete tournaments.",
        "- Legacy home-first mode sets the missing team-2 home flag to zero: team 1 hosts or the game is neutral. It must not be used on arbitrary away-first inputs.",
        "- Unknown lower-division opponents in the historical file retain the existing pooled 'other' identity.",
        "- predictions.csv contains all game-level probabilities and outcomes; metrics.csv contains season and pooled metrics. Per-season directories preserve training/holdout data, Stan input JSON, posterior CSVs, summaries, ratings, seeds, and diagnostics.",
        "- manifest.json records sampling settings, baseline constants, source hashes, and cutoffs; model_source/ preserves the model equations. Data and outputs remain local and ignored by Git.",
        "",
    ]
    (destination / "report.md").write_text("\n".join(lines))
