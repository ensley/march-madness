import pathlib
from typing import Any

import cmdstanpy
import pandas as pd

from march_madness.ingest import validate_unique_games

DATADIR = pathlib.Path(__file__).parent.parent.parent / "data"


def prep_season_data(
    file: str | pathlib.Path, year: int, games_through: str = "all"
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    return prep_season_frame(pd.read_parquet(file), year, games_through)


def prep_season_frame(
    games: pd.DataFrame, year: int, games_through: str = "all"
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Prepare a caller-selected frame, so temporal holdouts stay out of the fit."""
    if games_through not in ("ctourn", "all"):
        msg = f"{games_through} must be 'ctourn' or 'all'"
        raise ValueError(msg)

    df = games.loc[games["year"] == year, :].copy()
    if games_through == "ctourn":
        df = df.loc[df["type"].isin(("REG (Conf)", "REG (Non-Conf)", "CTOURN"))]

    if df.empty:
        msg = f"No training games found for {year}."
        raise ValueError(msg)

    df = df.fillna(value={"team2_link": "other"})
    validate_unique_games(df)

    teams = pd.concat(
        [
            df[["team1_link", "team1_name"]].rename(columns={"team1_link": "link", "team1_name": "name"}),
            df[["team2_link", "team2_name"]].rename(columns={"team2_link": "link", "team2_name": "name"}),
        ],
    ).drop_duplicates(ignore_index=True)

    tids = (
        pd.DataFrame({"link": teams["link"].unique()}).sort_values("link", ignore_index=True).reset_index(names="tid")
    )
    tids["tid"] += 1

    teams = (
        teams.merge(tids, on="link")
        .sort_values(["tid", "name"], ignore_index=True)
        .drop_duplicates(subset=["link", "tid"])
    )
    teams.loc[teams["link"] == "other", "name"] = "Other"

    df = (
        df.merge(teams, left_on="team1_link", right_on="link")
        .merge(teams, left_on="team2_link", right_on="link", suffixes=("_team1", "_team2"))
        .drop(columns=["link_team1", "link_team2", "name_team1", "name_team2"])
    )

    data = {
        "N": len(df),
        "T": len(teams),
        "Y": df[["team1_score", "team2_score"]].values.T,
        "tid": df[["tid_team1", "tid_team2"]].values.T,
        "minutes": df["minutes_played"].values,
        "H": df[["team1_home_adv", "team2_home_adv"]].values.T,
    }

    return df, teams, data


def collect_season_outputs(fit: cmdstanpy.CmdStanMCMC, teams: pd.DataFrame) -> pd.DataFrame:
    results = pd.DataFrame(
        {v: fit.stan_variable(v).mean(axis=0) for v in ("mu_off", "mu_def", "mu_home")},
        index=teams["tid"].values,
    )
    return results.join(teams.set_index("tid"))
