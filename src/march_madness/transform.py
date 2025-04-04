import pathlib
from typing import Any

import cmdstanpy
import pandas as pd

DATADIR = pathlib.Path(__file__).parent.parent.parent / "data"


def prep_season_data(
    file: str | pathlib.Path, year: int, games_through: str = "all"
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if games_through not in ("ctourn", "all"):
        msg = f"{games_through} must be 'ctourn' or 'all'"
        raise ValueError(msg)

    df = pd.read_parquet(file)
    df = df.loc[df["year"] == year, :]
    if games_through == "ctourn":
        df = df.loc[df["type"].isin(("REG (Conf)", "REG (Non-Conf)", "CTOURN"))]

    df = df.fillna(value={"team2_link": "other"})

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
        "V": df["team1_home_adv"].values,
    }

    return df, teams, data


def collect_season_outputs(fit: cmdstanpy.CmdStanMCMC, teams: pd.DataFrame) -> pd.DataFrame:
    results = pd.DataFrame(
        {v: fit.stan_variable(v).mean(axis=0) for v in ("mu_off", "mu_def", "mu_home")},
        index=teams["tid"].values,
    )
    return results.join(teams.set_index("tid"))
