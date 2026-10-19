import csv
import logging
import os
import pathlib
from datetime import datetime, timezone

import click
import cmdstanpy
import numpy as np
import pandas as pd
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from march_madness import tournament, transform
from march_madness.__about__ import __version__
from march_madness.backtest import run_backtest
from march_madness.bracket import draw_bracket
from march_madness.ingest import prepare_games_file
from march_madness.stan import Bracket, Season

console = Console()

logger = logging.getLogger(__name__)
stan_logger = logging.getLogger("cmdstanpy")

rich_handler = RichHandler(
    level=logging.INFO,
    console=console,
    rich_tracebacks=True,
    tracebacks_suppress=[click],
)

logger.addHandler(rich_handler)
stan_logger.handlers = []
stan_logger.addHandler(rich_handler)


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    invoke_without_command=True,
)
@click.version_option(version=__version__, prog_name="March Madness")
def march_madness():
    pass


@march_madness.command()
@click.argument("raw_file", type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path))
@click.option("--division", type=click.Choice(["NCAAM", "NCAAW"]), required=True)
@click.option("--year", type=int, required=True)
@click.option("--output", type=click.Path(dir_okay=False, path_type=pathlib.Path), required=True)
def prepare_games(raw_file: pathlib.Path, division: str, year: int, output: pathlib.Path):
    """Reconcile raw JSONL reports and write one Parquet row per game."""
    try:
        reports, games = prepare_games_file(raw_file, output, division=division, year=year)
    except (ValueError, OSError) as exc:
        raise click.ClickException(str(exc)) from exc
    console.print(f"Reconciled {reports} reports into {games} games; merged {reports - games} repeated reports.")
    console.print(f"Games saved to {output}")


@march_madness.command()
@click.argument("games_file", type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path))
@click.option("--year", "years", type=click.IntRange(1900, 2100), multiple=True, required=True)
@click.option("--division", type=click.Choice(["NCAAM", "NCAAW"]), default="NCAAM", show_default=True)
@click.option("--outdir", type=click.Path(file_okay=False, path_type=pathlib.Path), required=True)
@click.option(
    "--legacy-home-first",
    is_flag=True,
    help="Interpret a missing team-2 home flag as zero in legacy home-first/neutral data.",
)
@click.option("--samples", type=click.IntRange(min=1), default=4000, show_default=True)
@click.option("--warmup", type=click.IntRange(min=1), default=1000, show_default=True)
@click.option("--chains", type=click.IntRange(2, 4), default=4, show_default=True)
@click.option("--metric", type=click.Choice(["diag_e", "dense_e"]), default="diag_e", show_default=True)
@click.option("--seed", type=click.IntRange(1, 2**32 - 1), default=20261018, show_default=True)
@click.option(
    "--jobs",
    type=click.IntRange(1, 4),
    default=1,
    show_default=True,
    help="Seasons to fit concurrently; each uses --chains processes.",
)
def backtest(
    games_file: pathlib.Path,
    years: tuple[int, ...],
    division: str,
    outdir: pathlib.Path,
    samples: int,
    warmup: int,
    chains: int,
    seed: int,
    jobs: int,
    metric: str,
    *,
    legacy_home_first: bool,
):
    """Evaluate frozen pre-tournament forecasts against held-out NCAA games."""
    try:
        metrics = run_backtest(
            games_file,
            outdir,
            years=years,
            division=division,
            legacy_home_first=legacy_home_first,
            samples=samples,
            warmup=warmup,
            chains=chains,
            seed=seed,
            jobs=jobs,
            metric=metric,
            progress=click.echo,
        )
    except (ValueError, OSError, RuntimeError) as exc:
        raise click.ClickException(str(exc)) from exc
    table = Table(title="Pooled held-out tournament accuracy")
    for label in ["Model", "Games", "Log loss ↓", "Brier ↓", "Accuracy"]:
        table.add_column(label)
    for row in metrics.loc[metrics.year == "pooled"].itertuples():
        table.add_row(row.model, str(row.games), f"{row.log_loss:.4f}", f"{row.brier:.4f}", f"{row.accuracy:.1%}")
    console.print(table)
    console.print(f"Report saved to {outdir / 'report.md'}")


@march_madness.command()
@click.argument("games_file", type=click.Path(exists=True, path_type=pathlib.Path))
@click.option("--year", type=int, required=True)
@click.option(
    "--outdir",
    type=click.Path(dir_okay=True, exists=False, path_type=pathlib.Path),
    default=pathlib.Path("."),
)
def fit_season(games_file: pathlib.Path, year: int, outdir: pathlib.Path):
    outdir = outdir / f"march-madness-season-{datetime.now(tz=timezone.utc).strftime('%Y%m%d%H%M%S')}"

    try:
        _, teams, data = transform.prep_season_data(games_file, year, games_through="ctourn")
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    # Preserve sum-to-zero constraints when generated quantities reload the CSVs.
    season = Season.sample(data=data, iter_sampling=10000, sig_figs=18)

    outdir.mkdir(parents=True, exist_ok=True)
    season.save_csvfiles(dir=os.fspath(outdir))

    results = transform.collect_season_outputs(season, teams)
    results.to_csv(outdir / "results.csv", index_label="tid")
    console.print(f"Results saved to {outdir}")


@march_madness.command()
@click.option("--model-dir", type=click.Path(exists=True, path_type=pathlib.Path), required=True)
@click.option(
    "--tourney-file",
    type=click.Path(exists=True, path_type=pathlib.Path),
    required=True,
)
@click.option(
    "--outdir",
    type=click.Path(dir_okay=True, exists=False, path_type=pathlib.Path),
    default=pathlib.Path("."),
)
def simulate_tournament(model_dir: pathlib.Path, tourney_file: pathlib.Path, outdir):
    season = cmdstanpy.from_csv(os.fspath(model_dir) + "/season-*.csv", method="sample")
    if not isinstance(season, cmdstanpy.CmdStanMCMC):
        msg = "Model unable to be read from disk."
        raise TypeError(msg)
    season_results = pd.read_csv(model_dir / "results.csv")

    tourney = tournament.read_tournament_file(tourney_file)
    data = {
        "N": len(tourney),
        "T": season.stan_variable("mu_off").shape[1],
        "tid": tourney,
    }
    bracket = Bracket.generate_quantities(data=data, previous_fit=season)
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(os.fspath(outdir / "bracket.npy"), bracket.bracket)
    rounds = bracket.stan_variable("bracket").shape[1] - 1
    outcomes = (
        pd.DataFrame(bracket.bracket.mean(axis=0).T, index=pd.Index(tourney))
        .join(season_results.set_index("tid"))
        .sort_values(rounds, ascending=False)
    )
    outcomes.to_csv(model_dir / "tourney_outcomes.csv")
    console.print(f"Results saved to {model_dir / 'tourney_outcomes.csv'}")

    outcomes_styled = outcomes.set_index("name").drop(columns="link")
    table = Table(title="Tournament results")
    table.add_column("Team")
    for col in outcomes_styled.columns:
        table.add_column(str(col), justify="right")
    for row in outcomes_styled.itertuples():
        pcts = [f"{r:.1%}" for r in row[1 : (rounds + 2)]]
        coefs = [f"{r:.2f}" for r in row[(rounds + 2) :]]
        table.add_row(row[0], *pcts, *coefs)

    console.print(table)


@march_madness.command()
@click.option(
    "--tourney-file",
    type=click.Path(exists=True, path_type=pathlib.Path),
    required=True,
    help="Tournament CSV file with team names",
)
@click.option(
    "--bracket-dir",
    type=click.Path(exists=True, path_type=pathlib.Path),
    required=True,
    help="Directory containing bracket CSV files",
)
@click.option(
    "--compact",
    is_flag=True,
    default=False,
    help="Use compact bracket display",
)
@click.option(
    "--n",
    type=int,
    required=True,
    help="Which draw to use",
)
def print_simulation(tourney_file: pathlib.Path, bracket_dir: pathlib.Path, n: int, *, compact: bool):
    """Display the tournament bracket with predicted winners."""
    # Extract team names from tourney file (second column)
    with open(tourney_file, newline="") as f:
        reader = csv.reader(f)
        tourney = pd.DataFrame(
            [(int(row[0]), row[1].strip(), int(row[2])) for row in reader if not row[0].startswith("#")],
            columns=["tid", "name", "seed"],
        )

    bracket = np.load(bracket_dir / "bracket.npy", allow_pickle=True)

    if compact:
        from march_madness.bracket import draw_bracket_compact

        draw_bracket_compact(bracket[n].T, tourney["name"].values, console)
    else:
        draw_bracket(bracket[n].T, tourney["name"].values, console)
