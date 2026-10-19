"""A local browser interface for the existing Python models and saved artifacts."""

import secrets
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.exceptions import HTTPException

from march_madness.web.service import Jobs, Workspace


def create_app(root: str | Path = ".") -> Flask:
    app = Flask(__name__)
    app.config.update(MAX_CONTENT_LENGTH=16_384, TRUSTED_HOSTS=["localhost", "127.0.0.1", "[::1]"])
    workspace = Workspace(Path(root))
    jobs = Jobs(workspace)
    token = secrets.token_urlsafe(32)
    app.extensions["workspace"] = workspace
    app.extensions["jobs"] = jobs

    @app.before_request
    def protect_local_actions():
        if request.method == "POST" and not secrets.compare_digest(request.headers.get("X-Workspace-Token", ""), token):
            return jsonify(error="Reload this page before starting an operation."), 403
        if request.method == "POST" and not isinstance(request.get_json(), dict):
            return jsonify(error="Send a JSON object with the operation settings."), 400
        return None

    @app.after_request
    def headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
            "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
        )
        return response

    @app.errorhandler(Exception)
    def error(exc):
        if isinstance(exc, HTTPException):
            return jsonify(error=exc.description), exc.code
        if isinstance(exc, ValueError | KeyError | OSError):
            return jsonify(error=str(exc)), 400
        app.logger.exception("Web operation failed")
        return jsonify(error="The operation failed. Inspect the server log, then try another saved run."), 500

    @app.get("/")
    def index():
        return render_template("index.html", token=token)

    @app.get("/api/catalog")
    def catalog():
        return jsonify(workspace.catalog())

    @app.get("/api/ratings")
    def ratings():
        return jsonify(workspace.ratings(request.args.get("run", "")))

    @app.get("/api/backtest")
    def backtest():
        return jsonify(workspace.backtest(request.args.get("run", "")))

    @app.get("/api/simulation")
    def simulation():
        return jsonify(
            workspace.simulation(request.args.get("run", ""), integer(request.args, "draw", 0, 1_000_000, 0))
        )

    @app.get("/api/jobs")
    def job_status():
        return jsonify(jobs.all())

    @app.post("/api/matchup")
    def matchup():
        body = request.get_json()
        return jsonify(
            workspace.matchup(
                body.get("model", ""),
                integer(body, "first", 1, 10000),
                integer(body, "second", 1, 10000),
                body.get("venue", "neutral"),
            )
        )

    @app.post("/api/fit")
    def fit():
        body = request.get_json()
        dataset = body.get("dataset", "")
        workspace.path(dataset)
        division = body.get("division", "NCAAM")
        if division not in {"NCAAM", "NCAAW"}:
            msg = "Choose men's or women's basketball."
            raise ValueError(msg)
        legacy = body.get("legacy_home_first", False)
        if not isinstance(legacy, bool):
            msg = "The legacy home-first option must be true or false."
            raise ValueError(msg)  # noqa: TRY004 - Client settings use validation errors.
        year = integer(body, "year", 1900, 2100)
        samples = integer(body, "samples", 100, 10000, 4000)
        warmup = integer(body, "warmup", 100, 10000, 1000)
        seed = integer(body, "seed", 1, 2**32 - 1, 20261018)
        key = jobs.submit(
            "fit",
            lambda progress: workspace.fit_season(
                dataset, division, year, samples, warmup, seed, progress, legacy_home_first=legacy
            ),
        )
        return jsonify(id=key), 202

    @app.post("/api/simulate")
    def simulate():
        body = request.get_json()
        model, tournament = body.get("model", ""), body.get("tournament", "")
        workspace.path(model)
        workspace.path(tournament)
        seed = integer(body, "seed", 1, 2**32 - 1, 20261018)
        key = jobs.submit("simulation", lambda progress: workspace.simulate(model, tournament, seed, progress))
        return jsonify(id=key), 202

    @app.get("/api/download")
    def download():
        filename = request.args.get("file", "")
        if filename not in {
            "results.csv",
            "metrics.csv",
            "predictions.csv",
            "calibration.csv",
            "outcomes.csv",
            "posterior_summary.csv",
            "teams.csv",
        }:
            msg = "Choose a results CSV to download."
            raise ValueError(msg)
        path = workspace.path(request.args.get("run", "") + "/" + filename)
        return send_file(path, as_attachment=True)

    return app


def integer(values, key: str, minimum: int, maximum: int, default=None) -> int:
    raw = values.get(key, default)
    if isinstance(raw, bool):
        msg = f"{key} must be a whole number between {minimum} and {maximum}."
        raise ValueError(msg)  # noqa: TRY004 - Invalid client input is reported as a validation error.
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        msg = f"{key} must be a whole number between {minimum} and {maximum}."
        raise ValueError(msg) from exc
    if str(value) != str(raw) or not minimum <= value <= maximum:
        msg = f"{key} must be a whole number between {minimum} and {maximum}."
        raise ValueError(msg)
    return value
