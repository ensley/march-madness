"""Exercise the browser API with small local artifacts and deterministic jobs."""

import json
import threading

import numpy as np
import pandas as pd
import pytest

from march_madness.web import create_app
from march_madness.web.service import Jobs, Workspace


@pytest.fixture
def workspace(tmp_path):
    directory = tmp_path / "output" / "season"
    directory.mkdir(parents=True)
    pd.DataFrame(
        {
            "tid": [1, 2, 3, 4],
            "name": ["Alpha", "Beta", "Gamma", "Delta"],
            "link": ["a", "b", "c", "d"],
            "mu_off": [0.2, 0, -0.1, -0.1],
            "mu_def": [0.1, 0.1, -0.1, -0.1],
            "mu_home": [0.1] * 4,
        }
    ).to_csv(directory / "results.csv", index=False)
    (directory / "manifest.json").write_text(json.dumps({"year": 2024, "warnings": ["Review this test fit."]}))
    (tmp_path / "data").mkdir()
    return Workspace(tmp_path)


@pytest.fixture
def client(workspace):
    app = create_app(workspace.root)
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client
    app.extensions["jobs"].executor.shutdown(wait=True)


def token(client):
    from bs4 import BeautifulSoup

    page = BeautifulSoup(client.get("/").data, "html.parser")
    return {"X-Workspace-Token": page.select_one('meta[name="workspace-token"]')["content"]}


def test_catalog_and_centered_ratings(client):
    catalog = client.get("/api/catalog").get_json()
    assert catalog["fits"][0]["id"] == "output/season"
    ratings = client.get("/api/ratings?run=output/season").get_json()
    assert ratings["teams"][0]["name"] == "Alpha"
    assert ratings["teams"][0]["rating"] == pytest.approx(12)
    assert sum(t["offense"] for t in ratings["teams"]) == pytest.approx(0)
    assert ratings["warnings"] == ["Review this test fit."]
    assert ratings["has_draws"] is False


@pytest.mark.parametrize("value", ["../outside", ".git/config", ".", "/etc/passwd", "README.md"])
def test_workspace_rejects_non_artifact_paths(workspace, value):
    with pytest.raises(ValueError, match="workspace|data/|output"):
        workspace.path(value)


def test_workspace_rejects_symlink_escape(workspace, tmp_path):
    outside = tmp_path.parent / "outside-web-test"
    try:
        (workspace.root / "data" / "escape").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Symlinks are unavailable on this platform.")
    with pytest.raises(ValueError, match="workspace|data/|output"):
        workspace.path("data/escape/file.csv")


def test_local_actions_require_token_and_trusted_host(client):
    assert client.post("/api/fit", json={}).status_code == 403
    assert client.get("/api/catalog", headers={"Host": "untrusted.example"}).status_code == 400
    assert client.get("/").headers["Content-Security-Policy"].startswith("default-src 'self'")
    response = client.post("/api/fit", json={"dataset": "../outside"}, headers=token(client))
    assert response.status_code == 400


@pytest.mark.parametrize("year", [True, 2024.5, "invalid", 2200])
def test_fit_rejects_invalid_sampling_inputs(client, year):
    response = client.post("/api/fit", json={"dataset": "data/games.parquet", "year": year}, headers=token(client))
    assert response.status_code == 400
    assert "year must" in response.get_json()["error"]


def test_download_does_not_expose_arbitrary_files(client):
    assert client.get("/api/download?run=output/season&file=results.csv").status_code == 200
    assert client.get("/api/download?run=output/season&file=../../pyproject.toml").status_code == 400
    assert client.get("/api/download?run=../outside&file=results.csv").status_code == 400


def test_matchup_preserves_posterior_and_home_advantage(workspace, monkeypatch):
    class Fit:
        def stan_variable(self, name):
            if name == "sigma":
                return np.full(20, 0.2)
            return np.tile(
                {"mu_off": [0.2, 0, -0.1, -0.1], "mu_def": [0.1, 0.1, -0.1, -0.1], "mu_home": [0.1] * 4}[name], (20, 1)
            )

    monkeypatch.setattr(workspace, "fit", lambda _value: Fit())
    neutral = workspace.matchup("output/season", 1, 2, "neutral")
    home = workspace.matchup("output/season", 1, 2, "first")
    reverse = workspace.matchup("output/season", 2, 1, "neutral")
    assert home["probability"] > neutral["probability"] > 0.5
    assert neutral["probability"] + reverse["probability"] == pytest.approx(1)
    assert home["margin"] - neutral["margin"] == pytest.approx(4)
    with pytest.raises(ValueError, match="different"):
        workspace.matchup("output/season", 1, 1, "neutral")


def test_simulation_rounds_use_joint_draws(workspace):
    directory = workspace.root / "output" / "tournament"
    directory.mkdir()
    pd.DataFrame(
        {
            "tid": [1, 2, 3, 4],
            "name": ["Alpha", "Beta", "Gamma", "Delta"],
            "seed": [1, 4, 2, 3],
            "0": [1, 0, 1, 0],
            "1": [0, 0, 1, 0],
        }
    ).to_csv(directory / "outcomes.csv", index=False)
    np.save(directory / "bracket.npy", np.array([[[1, 0, 1, 0], [0, 0, 1, 0]]], dtype=np.int8))
    result = workspace.simulation("output/tournament")
    assert result["rounds"][0][0] == {"first": "Alpha", "second": "Beta", "winner": "Alpha"}
    assert result["rounds"][1][0] == {"first": "Alpha", "second": "Gamma", "winner": "Gamma"}
    assert workspace.catalog()["simulations"][0]["id"] == "output/tournament"
    with pytest.raises(ValueError, match="between"):
        workspace.simulation("output/tournament", 1)


def test_job_failures_and_restarts_are_visible(workspace):
    jobs = Jobs(workspace)

    def fail(progress):
        progress("Checking data")
        msg = "Resolve duplicate team IDs."
        raise ValueError(msg)

    key = jobs.submit("simulation", fail)
    jobs.executor.shutdown(wait=True)
    assert jobs.all()[0]["status"] == "failed"
    assert "duplicate" in jobs.all()[0]["message"]
    file = workspace.root / "output" / "web" / "jobs" / f"{key}.json"
    saved = json.loads(file.read_text())
    saved["status"] = "running"
    file.write_text(json.dumps(saved))
    restarted = Jobs(workspace)
    assert restarted.all()[0]["status"] == "interrupted"
    restarted.executor.shutdown(wait=True)


def test_jobs_reject_overlapping_work(workspace):
    jobs = Jobs(workspace)
    gate = threading.Event()
    key = jobs.submit("fit", lambda _progress: gate.wait(5))
    try:
        with pytest.raises(ValueError, match="already running"):
            jobs.submit("fit", lambda _progress: {})
    finally:
        gate.set()
        jobs.executor.shutdown(wait=True)
    assert jobs.all()[0]["id"] == key


def test_fit_api_submits_validated_background_job(client, monkeypatch):
    jobs = client.application.extensions["jobs"]
    captured = []
    monkeypatch.setattr(jobs, "submit", lambda kind, _operation: captured.append(kind) or "job-1")
    response = client.post("/api/fit", json={"dataset": "data/games.parquet", "year": 2024}, headers=token(client))
    assert response.status_code == 202
    assert response.get_json() == {"id": "job-1"}
    assert captured == ["fit"]


def test_simulation_accepts_known_alias_but_rejects_wrong_identity(workspace, monkeypatch):
    from march_madness.stan import Bracket

    directory = workspace.root / "output" / "season"
    teams = pd.read_csv(directory / "results.csv")
    teams.loc[0, "name"] = "Virginia Commonwealth"
    teams.to_csv(directory / "results.csv", index=False)
    source = workspace.root / "data" / "tournament.csv"
    source.write_text("1,VCU,1\n2,Beta,2\n")

    class Fit:
        column_names = ("mu_scoring",)

    class Draws:
        def stan_variable(self, _name):
            return np.array([[[1, 0]], [[0, 1]]])

    monkeypatch.setattr(workspace, "fit", lambda _value: Fit())
    monkeypatch.setattr(Bracket, "generate_quantities", lambda **_kwargs: Draws())
    result = workspace.simulate("output/season", "data/tournament.csv", 1, lambda _message: None)
    saved = pd.read_csv(workspace.path(result["id"]) / "outcomes.csv")
    assert saved.name.tolist() == ["Virginia Commonwealth", "Beta"]
    assert saved["0"].tolist() == [0.5, 0.5]
    source.write_text("1,Gamma,1\n2,Beta,2\n")
    with pytest.raises(ValueError, match="does not match"):
        workspace.simulate("output/season", "data/tournament.csv", 1, lambda _message: None)


@pytest.mark.parametrize("body", [None, [], "text"])
def test_actions_reject_non_object_json(client, body):
    response = client.post("/api/fit", data=json.dumps(body), content_type="application/json", headers=token(client))
    assert response.status_code == 400


def test_restarted_job_history_keeps_chronological_order(workspace):
    import os

    directory = workspace.root / "output" / "web" / "jobs"
    directory.mkdir(parents=True)
    for key, timestamp in [("z-older", 1000), ("a-newer", 2000)]:
        file = directory / f"{key}.json"
        file.write_text(
            json.dumps({"id": key, "status": "complete", "message": "Saved", "kind": "fit", "result": None})
        )
        os.utime(file, (timestamp, timestamp))
    jobs = Jobs(workspace)
    assert [job["id"] for job in jobs.all()] == ["z-older", "a-newer"]
    jobs.executor.shutdown(wait=True)
