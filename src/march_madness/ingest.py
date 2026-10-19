"""Reconcile scraped school reports into one observation per basketball game."""

import datetime
import pathlib
import re
from collections.abc import Iterable

import pandas as pd

from march_madness.scrape import Game, GameLog

GameKey = tuple[str, int, datetime.date, str, str]


def game_key(division: str, year: int, date: datetime.date, school: str, opponent: str) -> GameKey:
    """Identify a game independently of report orientation, names, or scores."""
    first, second = sorted((school, opponent))
    return division, year, date, first, second


def _school_name(name: str) -> str:
    # NCAA is a source annotation, not a set of trailing characters to strip.
    return re.sub(r"\s+NCAA$", "", " ".join(name.split()))


def reconcile_gamelogs(gamelogs: Iterable[GameLog], *, division: str, year: int) -> list[Game]:
    """Merge agreeing reports; reject ambiguous games with the original reports."""
    if division not in ("NCAAM", "NCAAW"):
        msg = "division must be 'NCAAM' or 'NCAAW'"
        raise ValueError(msg)
    category = "men" if division == "NCAAM" else "women"
    logs = list(gamelogs)

    # Prefer each school's own directory name over opponent abbreviations.
    names: dict[str, str] = {}
    for log in logs:
        name = _school_name(log.school.name)
        slug = log.school.shortname
        names[slug] = min(names.get(slug, name), name)
    fallback_names: dict[str, str] = {}
    for log in logs:
        name = _school_name(log.opponent.name)
        slug = log.opponent.shortname
        fallback_names[slug] = min(fallback_names.get(slug, name), name)
    names = fallback_names | names

    games: dict[GameKey, Game] = {}
    sources: dict[GameKey, tuple[int, GameLog]] = {}
    for number, log in enumerate(logs, start=1):
        key = game_key(division, year, log.date, log.school.shortname, log.opponent.shortname)
        if key[3] == key[4]:
            msg = f"Report {number} has the same school on both sides: {log.model_dump_json()}"
            raise ValueError(msg)
        if log.is_home and log.is_neutral_site:
            msg = f"Report {number} is both home and neutral: {log.model_dump_json()}"
            raise ValueError(msg)
        sides = sorted(
            [
                (log.school.shortname, log.score, int(log.is_home)),
                (log.opponent.shortname, log.opp_score, int(not log.is_home and not log.is_neutral_site)),
            ]
        )
        first, second = sides
        game = Game(
            division=division,
            year=year,
            date=log.date,
            type=log.game_type,
            team1_link=f"/cbb/schools/{first[0]}/{category}/",
            team2_link=f"/cbb/schools/{second[0]}/{category}/",
            team1_name=names[first[0]],
            team2_name=names[second[0]],
            team1_score=first[1],
            team2_score=second[1],
            team1_home_adv=first[2],
            team2_home_adv=second[2],
            minutes_played=40 + 5 * log.ot_periods,
        )
        previous = games.get(key)
        if previous is not None:
            conflicts = [field for field, value in game.model_dump().items() if value != getattr(previous, field)]
            if conflicts:
                original_number, original = sources[key]
                msg = (
                    f"Conflicting reports for {division} {year}, {log.date}, {key[3]} vs {key[4]} "
                    f"({', '.join(conflicts)}). Resolve the reports or supply distinct game identities "
                    "for a same-day rematch. "
                    f"Report {original_number}: {original.model_dump_json()}; "
                    f"report {number}: {log.model_dump_json()}"
                )
                raise ValueError(msg)
        else:
            games[key] = game
            sources[key] = number, log

    return [games[key] for key in sorted(games)]


def prepare_games_file(
    source: str | pathlib.Path, destination: str | pathlib.Path, *, division: str, year: int
) -> tuple[int, int]:
    """Validate raw JSONL reports before writing the reconciled Parquet dataset."""
    source, destination = pathlib.Path(source), pathlib.Path(destination)
    if source.resolve() == destination.resolve():
        msg = "The output must differ from the raw JSONL input."
        raise ValueError(msg)
    logs = []
    with source.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                logs.append(GameLog.model_validate_json(line))
            except ValueError as exc:
                msg = f"Invalid game report at {source}:{number}: {exc}"
                raise ValueError(msg) from exc
    if not logs:
        msg = f"No game reports found in {source}."
        raise ValueError(msg)
    games = reconcile_gamelogs(logs, division=division, year=year)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([game.model_dump() for game in games]).to_parquet(destination, index=False)
    return len(logs), len(games)


def validate_unique_games(games: pd.DataFrame) -> None:
    """Reject repeated game identities before they become Stan observations."""
    seen: set[GameKey] = set()
    for row in games.itertuples():
        key = game_key(row.division, row.year, pd.Timestamp(row.date).date(), row.team1_link, row.team2_link)
        if key in seen:
            msg = (
                f"Repeated game in model input: {key[0]} {key[1]}, {key[2]}, {key[3]} vs {key[4]}. "
                "Rebuild the dataset from raw reports with 'march-madness prepare-games'. "
                "Same-day rematches require distinct game identities."
            )
            raise ValueError(msg)
        seen.add(key)
