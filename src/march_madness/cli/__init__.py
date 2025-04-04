import logging
import os
import pathlib
from datetime import datetime, timezone

import click
import cmdstanpy
import pandas as pd
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from march_madness import tournament, transform
from march_madness.__about__ import __version__
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


@click.group(context_settings={"help_option_names": ["-h", "--help"]}, invoke_without_command=True)
@click.version_option(version=__version__, prog_name="March Madness")
def march_madness():
    pass


@march_madness.command()
@click.argument("games_file", type=click.Path(exists=True, path_type=pathlib.Path))
@click.option("--year", type=int, required=True)
@click.option(
    "--outdir", type=click.Path(dir_okay=True, exists=False, path_type=pathlib.Path), default=pathlib.Path(".")
)
def fit_season(games_file: pathlib.Path, year: int, outdir: pathlib.Path):
    outdir = outdir / f"march-madness-season-{datetime.now(tz=timezone.utc).strftime('%Y%m%d%H%M%S')}"

    _, teams, data = transform.prep_season_data(games_file, year, games_through="ctourn")
    season = Season.sample(data=data)

    outdir.mkdir(parents=True, exist_ok=True)
    season.save_csvfiles(dir=os.fspath(outdir))

    results = transform.collect_season_outputs(season, teams)
    results.to_csv(outdir / "results.csv")
    console.print(f"Results saved to {outdir}")


@march_madness.command()
@click.option("--model-dir", type=click.Path(exists=True, path_type=pathlib.Path), required=True)
@click.option("--tourney-file", type=click.Path(exists=True, path_type=pathlib.Path), required=True)
def simulate_tournament(model_dir: pathlib.Path, tourney_file: pathlib.Path):
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
