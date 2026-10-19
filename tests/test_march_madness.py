from unittest.mock import patch

import pandas as pd
from click.testing import CliRunner

from march_madness.cli import march_madness
from march_madness.scrape import GameLog


def test_import_succeeds():
    runner = CliRunner()
    result = runner.invoke(march_madness)
    assert result.exit_code == 0


def test_prepare_games_cli(tmp_path):
    log = GameLog.model_validate(
        {
            "school": {"shortname": "alpha", "name": "Alpha", "is_valid": True},
            "opponent": {"shortname": "beta", "name": "Beta", "is_valid": True},
            "date": "2026-02-10",
            "game_type": "REG (Conf)",
            "is_home": True,
            "is_neutral_site": False,
            "score": 83,
            "opp_score": 69,
            "ot_periods": 0,
        }
    )
    source = tmp_path / "logs.jsonl"
    source.write_text((log.model_dump_json() + "\n") * 2, encoding="utf-8")
    output = tmp_path / "games.parquet"
    runner = CliRunner()
    result = runner.invoke(
        march_madness,
        [
            "prepare-games",
            str(source),
            "--division",
            "NCAAM",
            "--year",
            "2026",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "2 reports into 1 games" in result.output
    assert output.exists()

    # Even a stale file with changed scores must fail before sampling.
    games = pd.read_parquet(output)
    duplicate = games.copy()
    duplicate["team1_score"] += 1
    pd.concat([games, duplicate]).to_parquet(output)
    with patch("march_madness.stan.Season.sample") as sample:
        result = runner.invoke(march_madness, ["fit-season", str(output), "--year", "2026"])
        sample.assert_not_called()
    assert result.exit_code != 0
    assert "Repeated game" in result.output
    assert "prepare-games" in result.output


def test_backtest_cli_passes_cutoff_and_sampling_options(tmp_path):
    source = tmp_path / "games.parquet"
    pd.DataFrame().to_parquet(source)
    metrics = pd.DataFrame(
        {
            "year": ["pooled"],
            "model": ["stan"],
            "games": [67],
            "log_loss": [0.5],
            "brier": [0.17],
            "accuracy": [0.7],
        }
    )
    with patch("march_madness.cli.run_backtest", return_value=metrics) as run:
        result = CliRunner().invoke(
            march_madness,
            [
                "backtest",
                str(source),
                "--year",
                "2023",
                "--year",
                "2024",
                "--legacy-home-first",
                "--samples",
                "2000",
                "--warmup",
                "1000",
                "--jobs",
                "2",
                "--metric",
                "dense_e",
                "--outdir",
                str(tmp_path / "run"),
            ],
        )
    assert result.exit_code == 0, result.output
    assert "Pooled held-out tournament accuracy" in result.output
    assert run.call_args.kwargs["years"] == (2023, 2024)
    assert run.call_args.kwargs["legacy_home_first"] is True
    assert run.call_args.kwargs["samples"] == 2000
    assert run.call_args.kwargs["jobs"] == 2
    assert run.call_args.kwargs["metric"] == "dense_e"


def test_backtest_cli_rejects_legacy_schema_without_opt_in(tmp_path):
    source = tmp_path / "games.parquet"
    pd.DataFrame(
        [
            {
                "division": "NCAAM",
                "year": 2024,
                "date": "2024-03-01",
                "type": "REG (Conf)",
                "team1_link": "alpha",
                "team2_link": "beta",
                "team1_name": "Alpha",
                "team2_name": "Beta",
                "team1_score": 70,
                "team2_score": 60,
                "team1_home_adv": 0,
                "minutes_played": 40,
            }
        ]
    ).to_parquet(source)
    with patch("march_madness.backtest.fit_season") as sample:
        result = CliRunner().invoke(
            march_madness,
            [
                "backtest",
                str(source),
                "--year",
                "2024",
                "--outdir",
                str(tmp_path / "run"),
            ],
        )
        sample.assert_not_called()
    assert result.exit_code != 0
    assert "legacy-home-first" in result.output
