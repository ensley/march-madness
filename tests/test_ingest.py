import datetime

import pandas as pd
import pytest

from march_madness.ingest import prepare_games_file, reconcile_gamelogs, validate_unique_games
from march_madness.scrape import GameLog, School
from march_madness.transform import prep_season_data


@pytest.fixture
def report():
    return GameLog(
        school=School(shortname="alpha", name="Alpha", is_valid=True),
        opponent=School(shortname="beta", name="Beta", is_valid=True),
        date=datetime.date(2026, 2, 10),
        game_type="REG (Conf)",
        is_home=True,
        is_neutral_site=False,
        score=83,
        opp_score=69,
        ot_periods=0,
    )


def reverse(report):
    return report.model_copy(
        update={
            "school": report.opponent,
            "opponent": report.school,
            "score": report.opp_score,
            "opp_score": report.score,
            "is_home": not report.is_home and not report.is_neutral_site,
        }
    )


def test_gamelog_equality_is_reflexive_and_hash_consistent(report):
    same_report = report
    assert report == same_report
    assert report == report.model_copy()
    assert len({report, report.model_copy()}) == 1
    assert report != reverse(report)


@pytest.mark.parametrize("neutral", [False, True])
def test_reconcile_uses_school_ids_and_preserves_scores_and_venue(report, neutral):
    report = report.model_copy(update={"is_home": not neutral, "is_neutral_site": neutral})
    other = reverse(report).model_copy(
        update={
            "school": School(shortname="beta", name="Beta\xa0NCAA", is_valid=True),
            "opponent": School(shortname="alpha", name="A.", is_valid=True),
        }
    )
    games = reconcile_gamelogs([report, other, report.model_copy()], division="NCAAW", year=2026)
    assert len(games) == 1
    game = games[0]
    assert game.team1_link == "/cbb/schools/alpha/women/"
    assert game.team2_link == "/cbb/schools/beta/women/"
    assert (game.team1_name, game.team2_name) == ("Alpha", "Beta")
    assert (game.team1_score, game.team2_score) == (83, 69)
    assert (game.team1_home_adv, game.team2_home_adv) == (int(not neutral), 0)
    assert game.minutes_played == 40
    assert games == reconcile_gamelogs([other, report], division="NCAAW", year=2026)


@pytest.mark.parametrize(
    ("update", "field"),
    [
        ({"score": 70}, "team2_score"),
        ({"is_home": True}, "team2_home_adv"),
        ({"is_neutral_site": True}, "team1_home_adv"),
        ({"ot_periods": 1}, "minutes_played"),
        ({"game_type": "CTOURN"}, "type"),
    ],
)
def test_conflicting_reports_are_rejected_with_both_sources(report, update, field):
    other = reverse(report).model_copy(update=update)
    with pytest.raises(ValueError, match=f"Conflicting reports.*{field}") as error:
        reconcile_gamelogs([report, other], division="NCAAM", year=2026)
    assert report.model_dump_json() in str(error.value)
    assert other.model_dump_json() in str(error.value)


def test_different_dates_remain_separate_games(report):
    rematch = report.model_copy(update={"date": datetime.date(2026, 3, 1)})
    games = reconcile_gamelogs([rematch, reverse(report), report], division="NCAAM", year=2026)
    assert [game.date for game in games] == [report.date, rematch.date]
    assert games[0].team1_link.endswith("/men/")


def test_name_cleanup_does_not_strip_school_name_characters(report):
    report = report.model_copy(
        update={
            "school": report.school.model_copy(update={"name": "UCLA\xa0NCAA"}),
            "opponent": report.opponent.model_copy(update={"name": "UTSA"}),
        }
    )
    game = reconcile_gamelogs([report], division="NCAAW", year=2026)[0]
    assert (game.team1_name, game.team2_name) == ("UCLA", "UTSA")


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"is_neutral_site": True}, "both home and neutral"),
        ({"opponent": School(shortname="alpha", name="Alpha", is_valid=True)}, "same school"),
    ],
)
def test_invalid_game_reports_are_rejected(report, update, message):
    with pytest.raises(ValueError, match=message):
        reconcile_gamelogs([report.model_copy(update=update)], division="NCAAM", year=2026)


def test_reject_unknown_division(report):
    with pytest.raises(ValueError, match="division must be"):
        reconcile_gamelogs([report], division="unknown", year=2026)


def test_prepare_games_file_keeps_raw_logs_and_writes_one_game(report, tmp_path):
    source = tmp_path / "logs.jsonl"
    raw = report.model_dump_json() + "\n\n" + reverse(report).model_dump_json() + "\n"
    source.write_text(raw, encoding="utf-8")
    destination = tmp_path / "data" / "games.parquet"
    assert prepare_games_file(source, destination, division="NCAAM", year=2026) == (2, 1)
    assert source.read_text(encoding="utf-8") == raw
    frame, teams, data = prep_season_data(destination, 2026, games_through="ctourn")
    assert len(frame) == data["N"] == 1
    assert len(teams) == 2
    assert data["tid"].shape == (2, 1)
    assert data["Y"].tolist() == [[83], [69]]
    assert data["H"].tolist() == [[1], [0]]


def test_conflict_leaves_existing_output_untouched(report, tmp_path):
    source = tmp_path / "logs.jsonl"
    other = reverse(report).model_copy(update={"score": 70})
    source.write_text(report.model_dump_json() + "\n" + other.model_dump_json(), encoding="utf-8")
    destination = tmp_path / "games.parquet"
    destination.write_bytes(b"existing output")
    with pytest.raises(ValueError, match="Conflicting reports"):
        prepare_games_file(source, destination, division="NCAAM", year=2026)
    assert destination.read_bytes() == b"existing output"


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("\n", "No game reports"),
        ("\ninvalid json", r"logs.jsonl:2"),
    ],
)
def test_invalid_input_does_not_create_output(tmp_path, content, message):
    source = tmp_path / "logs.jsonl"
    source.write_text(content, encoding="utf-8")
    destination = tmp_path / "games.parquet"
    with pytest.raises(ValueError, match=message):
        prepare_games_file(source, destination, division="NCAAM", year=2026)
    assert not destination.exists()


def test_prepare_games_file_cannot_overwrite_raw_input(report, tmp_path):
    source = tmp_path / "logs.jsonl"
    source.write_text(report.model_dump_json(), encoding="utf-8")
    with pytest.raises(ValueError, match="output must differ"):
        prepare_games_file(source, source, division="NCAAM", year=2026)
    assert source.read_text(encoding="utf-8") == report.model_dump_json()


@pytest.mark.parametrize("changed_score", [False, True])
def test_model_preparation_rejects_reversed_duplicate_games(report, tmp_path, changed_score):
    row = reconcile_gamelogs([report], division="NCAAM", year=2026)[0].model_dump()
    other = dict(row)
    for suffix in ["link", "name", "score", "home_adv"]:
        other[f"team1_{suffix}"], other[f"team2_{suffix}"] = row[f"team2_{suffix}"], row[f"team1_{suffix}"]
    if changed_score:
        other["team1_score"] += 1
    path = tmp_path / "games.parquet"
    pd.DataFrame([row, other]).to_parquet(path)
    with pytest.raises(ValueError, match="Repeated game.*prepare-games"):
        prep_season_data(path, 2026)


def test_uniqueness_check_distinguishes_seasons_and_divisions(report):
    row = reconcile_gamelogs([report], division="NCAAM", year=2026)[0].model_dump()
    validate_unique_games(pd.DataFrame([row, {**row, "year": 2025}, {**row, "division": "NCAAW"}]))
