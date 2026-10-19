import json
import math

import cmdstanpy
import numpy as np
import pandas as pd
import pytest

from march_madness import backtest
from march_madness.backtest import (
    calibration_table,
    elo_probabilities,
    posterior_predictions,
    run_backtest,
    score_predictions,
    tournament_split,
)


@pytest.fixture
def games():
    rows = []
    for date, kind, first, second in [
        ("2024-03-01", "REG (Conf)", 75, 60),
        ("2024-03-10", "CTOURN", 80, 70),
        ("2024-03-19", "FIRST-FOUR", 50, 99),
        ("2024-03-21", "ROUND-64", 90, 60),
        ("2024-03-19", "CTOURN", 200, 1),
        ("2024-03-23", "REG (Conf)", 150, 1),
        ("2024-03-12", "NIT", 160, 1),
    ]:
        rows.append(
            {
                "division": "NCAAM",
                "year": 2024,
                "date": date,
                "type": kind,
                "team1_link": "alpha",
                "team2_link": "beta",
                "team1_name": "Alpha",
                "team2_name": "Beta",
                "team1_score": first,
                "team2_score": second,
                "team1_home_adv": 0,
                "team2_home_adv": 0,
                "minutes_played": 40,
            }
        )
    frame = pd.DataFrame(rows)
    frame.loc[4, ["team2_link", "team2_name"]] = ["gamma", "Gamma"]
    return frame


def test_split_is_strictly_before_first_tournament_day(games):
    split = tournament_split(games, 2024, "NCAAM")
    assert split.cutoff == pd.Timestamp("2024-03-19")
    assert split.train.date.max() < split.cutoff
    assert len(split.train) == len(split.test) == 2
    assert set(split.test.type) == {"FIRST-FOUR", "ROUND-64"}
    assert split.train.team1_score.max() == 80
    assert "NIT" not in set(split.train.type)


def test_legacy_venue_requires_opt_in_and_keeps_neutral_games_neutral(games):
    legacy = games.drop(columns="team2_home_adv")
    with pytest.raises(ValueError, match="legacy-home-first"):
        tournament_split(legacy, 2024, "NCAAM")
    split = tournament_split(legacy, 2024, "NCAAM", legacy_home_first=True)
    assert not split.train.team2_home_adv.any()
    assert not split.test.team2_home_adv.any()


def test_modern_away_first_venue_is_preserved(games):
    games.loc[0, "team2_home_adv"] = 1
    split = tournament_split(games, 2024, "NCAAM")
    assert split.train.iloc[0].team2_home_adv == 1


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("date", None, "game dates"),
        ("team1_link", None, "identifiers"),
        ("minutes_played", 0, "durations"),
        ("team1_home_adv", -1, "flags"),
        ("team1_score", -1, "scores"),
        ("team1_score", 60, "tied"),
    ],
)
def test_invalid_data_fails_before_fitting(games, column, value, message):
    games.loc[0, column] = value
    with pytest.raises(ValueError, match=message):
        tournament_split(games, 2024, "NCAAM")


def test_unknown_tournament_team_is_not_silently_dropped(games):
    games.loc[2, "team2_link"] = "never-observed"
    with pytest.raises(ValueError, match="without pre-cutoff observations"):
        tournament_split(games, 2024, "NCAAM")


def test_no_tournament_no_training_and_missing_columns_are_errors(games):
    with pytest.raises(ValueError, match="No NCAA tournament"):
        tournament_split(games[games.type.str.startswith("REG")], 2024, "NCAAM")
    with pytest.raises(ValueError, match="No training games precede"):
        tournament_split(games[games.type == "ROUND-64"], 2024, "NCAAM")
    with pytest.raises(ValueError, match="Missing game columns"):
        tournament_split(games.drop(columns="date"), 2024, "NCAAM")


def test_split_rejects_duplicate_holdout_observations(games):
    duplicated = pd.concat([games, games.iloc[[2]]], ignore_index=True)
    with pytest.raises(ValueError, match="Repeated game"):
        tournament_split(duplicated, 2024, "NCAAM")


def test_elo_is_frozen_and_never_reads_holdout_scores(games):
    split = tournament_split(games, 2024, "NCAAM")
    probability = elo_probabilities(split.train.iloc[:1], split.test)
    expected = 1 / (1 + 10 ** (-20 / 400))
    np.testing.assert_allclose(probability, expected)
    changed = split.test.copy()
    changed["team1_score"] = [999, 0]
    changed["team2_score"] = [0, 999]
    np.testing.assert_array_equal(probability, elo_probabilities(split.train.iloc[:1], changed))


class FakeFit:
    def __init__(self, teams=2, *, rhat=1.0):
        self.variables = {name: np.zeros((8, teams)) for name in ("mu_off", "mu_def", "mu_home")}
        self.variables["sigma"] = np.full(8, 0.3)
        self.rhat = rhat

    def stan_variable(self, name):
        return self.variables[name]

    def summary(self):
        return pd.DataFrame({"R_hat": [self.rhat], "ESS_bulk": [1000.0]}, index=["sigma"])

    def method_variables(self):
        return {name: np.zeros((4, 2)) for name in ["divergent__", "treedepth__"]}


def test_posterior_win_probability_integrates_draws_with_correct_variance(games):
    split = tournament_split(games, 2024, "NCAAM")
    teams = pd.DataFrame({"tid": [2, 1], "link": ["beta", "alpha"]})
    fit = FakeFit()
    fit.variables["mu_off"][:, 0] = 0.2
    probability, margin = posterior_predictions(fit, teams, split.test)
    expected = 0.5 * math.erfc(-0.2 / (2 * 0.3))
    np.testing.assert_allclose(probability, expected)
    np.testing.assert_allclose(margin, 8)


def test_posterior_predictions_use_venue_and_are_symmetric(games):
    test = tournament_split(games, 2024, "NCAAM").test.copy()
    teams = pd.DataFrame({"tid": [1, 2], "link": ["alpha", "beta"]})
    fit = FakeFit()
    fit.variables["mu_home"][:, 1] = 0.1
    test["team2_home_adv"] = 1
    probability, margin = posterior_predictions(fit, teams, test)
    swapped = test.rename(
        columns={
            "team1_link": "team2_link",
            "team2_link": "team1_link",
            "team1_home_adv": "team2_home_adv",
            "team2_home_adv": "team1_home_adv",
        }
    )
    reversed_probability, reversed_margin = posterior_predictions(fit, teams, swapped)
    np.testing.assert_allclose(probability + reversed_probability, 1)
    np.testing.assert_allclose(margin, -4)
    np.testing.assert_allclose(margin, -reversed_margin)


def test_invalid_posterior_scale_is_rejected(games):
    fit = FakeFit()
    fit.variables["sigma"][:] = 0
    teams = pd.DataFrame({"tid": [1, 2], "link": ["alpha", "beta"]})
    with pytest.raises(ValueError, match="finite and positive"):
        posterior_predictions(fit, teams, tournament_split(games, 2024, "NCAAM").test)


@pytest.fixture
def predictions():
    return pd.DataFrame(
        {
            "model": ["stan", "stan"],
            "p_team1": [0.8, 0.2],
            "team1_won": [1, 0],
            "predicted_margin_per40": [8.0, -8.0],
            "actual_margin_per40": [10.0, -5.0],
        }
    )


def test_proper_scores_and_margin_units(predictions):
    row = score_predictions(predictions).iloc[0]
    assert row.log_loss == pytest.approx(-math.log(0.8))
    assert row.brier == pytest.approx(0.04)
    assert row.accuracy == 1
    assert row.margin_mae_per40 == pytest.approx(2.5)
    assert row.margin_rmse_per40 == pytest.approx(math.sqrt(6.5))


def test_coin_flip_accuracy_is_half_even_if_team1_always_wins(predictions):
    predictions["p_team1"] = 0.5
    predictions["team1_won"] = 1
    row = score_predictions(predictions).iloc[0]
    assert row.accuracy == 0.5
    assert row.brier == 0.25
    assert row.log_loss == pytest.approx(math.log(2))


@pytest.mark.parametrize("value", [-0.1, 1.1, np.nan])
def test_invalid_probabilities_are_rejected(predictions, value):
    predictions.loc[0, "p_team1"] = value
    with pytest.raises(ValueError, match="Invalid probabilities"):
        score_predictions(predictions)


def test_calibration_includes_probability_endpoints_and_empty_bins(predictions):
    predictions["p_team1"] = [0.0, 1.0]
    table = calibration_table(predictions)
    assert len(table) == 10
    assert table.games.sum() == 2
    assert table.iloc[0].games == table.iloc[-1].games == 1
    assert table.loc[table.games == 0, "observed"].isna().all()


def test_full_backtest_saves_auditable_holdouts_without_leakage(games, tmp_path, monkeypatch):
    source = tmp_path / "games.parquet"
    games.to_parquet(source)
    captured = []

    def fake_fit(data, destination, **kwargs):
        captured.append((data, kwargs))
        cmdstanpy.write_stan_json(str(destination / "stan_data.json"), data)
        return FakeFit(data["T"])

    monkeypatch.setattr(backtest, "fit_season", fake_fit)
    destination = tmp_path / "backtest"
    metrics = run_backtest(source, destination, years=[2024], samples=10, warmup=10)
    assert captured[0][0]["N"] == 2
    assert int(captured[0][0]["Y"].max()) == 80
    assert captured[0][1]["seed"] == 20263043
    assert set(metrics.model) == {"stan", "elo", "coin_flip"}
    assert set(metrics.year) == {2024, "pooled"}
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["metric"] == "diag_e"
    assert manifest["folds"][0]["cutoff"] == "2024-03-19"
    assert manifest["folds"][0]["test_games"] == 2
    assert manifest["folds"][0]["warnings"] == []
    assert (destination / "model_source" / "season.stan").exists()
    assert (destination / "2024" / "stan_data.json").exists()
    assert manifest["folds"][0]["rounds"]["FIRST-FOUR"] == 1
    saved = pd.read_csv(destination / "predictions.csv")
    assert len(saved) == 6
    assert (saved.groupby("model").size() == 2).all()
    assert pd.read_csv(destination / "calibration.csv").games.sum() == 6
    with pytest.raises(ValueError, match="not empty"):
        run_backtest(source, destination, years=[2024])


def test_backtest_records_sampling_warnings(games, tmp_path, monkeypatch):
    source = tmp_path / "games.parquet"
    games.to_parquet(source)
    monkeypatch.setattr(backtest, "fit_season", lambda data, _destination, **_kwargs: FakeFit(data["T"], rhat=1.1))
    destination = tmp_path / "run"
    run_backtest(source, destination, years=[2024])
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["folds"][0]["warnings"]
    assert "Sampling warning" in (destination / "report.md").read_text()


def test_preflight_errors_do_not_create_run_directory(games, tmp_path):
    source = tmp_path / "games.parquet"
    games.to_parquet(source)
    destination = tmp_path / "run"
    with pytest.raises(ValueError, match="No regular-season"):
        run_backtest(source, destination, years=[2024, 2020])
    assert not destination.exists()


def test_multiple_seasons_are_evaluated_separately_and_pooled(games, tmp_path, monkeypatch):
    previous = games.copy()
    previous["year"] = 2023
    previous["date"] = previous.date.str.replace("2024", "2023")
    source = tmp_path / "games.parquet"
    pd.concat([games, previous]).to_parquet(source)
    monkeypatch.setattr(backtest, "fit_season", lambda data, _destination, **_kwargs: FakeFit(data["T"]))
    destination = tmp_path / "run"
    metrics = run_backtest(source, destination, years=[2024, 2023], jobs=2)
    assert len(metrics) == 9
    assert (metrics.loc[metrics.year == "pooled", "games"] == 4).all()
    manifest = json.loads((destination / "manifest.json").read_text())
    assert [fold["year"] for fold in manifest["folds"]] == [2023, 2024]
    assert manifest["folds"][0]["seed"] != manifest["folds"][1]["seed"]
    for year in [2023, 2024]:
        training = pd.read_parquet(destination / str(year) / "training.parquet")
        holdout = pd.read_parquet(destination / str(year) / "holdout.parquet")
        assert training.date.max() < holdout.date.min()
        assert set(training.year) == set(holdout.year) == {year}


def test_failed_fit_is_recorded_in_manifest(games, tmp_path, monkeypatch):
    source = tmp_path / "games.parquet"
    games.to_parquet(source)

    def fail_fit(_data, _destination, **_kwargs):
        msg = "Sampling failed"
        raise RuntimeError(msg)

    monkeypatch.setattr(backtest, "fit_season", fail_fit)
    destination = tmp_path / "run"
    with pytest.raises(RuntimeError, match="Sampling failed"):
        run_backtest(source, destination, years=[2024])
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["error"] == "Sampling failed"
