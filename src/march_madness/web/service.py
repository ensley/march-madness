"""Local artifacts and modeling operations used by the browser interface."""

import csv
import json
import math
import threading
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import cmdstanpy
import numpy as np
import pandas as pd

from march_madness.backtest import ESS_LIMIT, MAX_TREEDEPTH, RHAT_LIMIT, posterior_predictions
from march_madness.transform import collect_season_outputs, prep_season_frame

MAX_CACHED_FITS = 2
MIN_BRACKET_TEAMS = 2
MAX_BRACKET_TEAMS = 128
TEAM_NAME_ALIASES = {"vcu": "virginia commonwealth", "liu": "long island university", "smu": "southern methodist"}


def normalized_team_name(name: str) -> str:
    name = " ".join(name.split()).casefold()
    return TEAM_NAME_ALIASES.get(name, name)


class Workspace:
    """Keep browser-selected inputs and generated artifacts inside one workspace."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self._fits: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def path(self, value: str) -> Path:
        if not isinstance(value, str) or not value:
            msg = "Choose a workspace file or saved run."
            raise ValueError(msg)
        candidate = (self.root / value).resolve()
        if not candidate.is_relative_to(self.root) or any(p.startswith(".") for p in Path(value).parts):
            msg = "Choose a file inside the workspace's data or output folders."
            raise ValueError(msg)
        parts = candidate.relative_to(self.root).parts
        if not parts or parts[0] not in {"data", "output", "outputs"}:
            msg = "Choose a file inside data/, output/, or outputs/."
            raise ValueError(msg)
        return candidate

    def relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def catalog(self) -> dict:
        fits, backtests, simulations, problems = [], [], [], []
        for directory in (self.root / "output", self.root / "outputs"):
            if not directory.exists():
                continue
            for file in sorted(directory.rglob("results.csv")):
                try:
                    if file.is_symlink() or not file.resolve().is_relative_to(self.root):
                        continue
                    teams = pd.read_csv(file)
                    if not {"tid", "name", "link", "mu_off", "mu_def", "mu_home"} <= set(teams):
                        continue
                    chains = self.chain_files(file.parent)
                    fits.append(
                        {
                            "id": self.relative(file.parent),
                            "name": self.relative(file.parent),
                            "teams": len(teams),
                            "modified": file.stat().st_mtime,
                            "has_draws": bool(chains),
                        }
                    )
                except (ValueError, OSError, pd.errors.ParserError) as exc:
                    problems.append(f"Could not read {self.relative(file)}: {exc}")
            for file in sorted(directory.rglob("metrics.csv")):
                if file.resolve().is_relative_to(self.root):
                    backtests.append(  # noqa: PERF401 - Check file containment before exposing a saved run.
                        {
                            "id": self.relative(file.parent),
                            "name": self.relative(file.parent),
                            "modified": file.stat().st_mtime,
                        }
                    )
            for file in sorted(directory.rglob("outcomes.csv")):
                if file.resolve().is_relative_to(self.root) and (file.parent / "bracket.npy").exists():
                    simulations.append(  # noqa: PERF401 - Also require a saved draw array.
                        {
                            "id": self.relative(file.parent),
                            "name": self.relative(file.parent),
                            "modified": file.stat().st_mtime,
                        }
                    )
        datasets = []
        data_dir = self.root / "data"
        for file in sorted(data_dir.glob("*.parquet")):
            try:
                frame = pd.read_parquet(file, columns=["year", "division"])
                seasons = frame.groupby(["division", "year"]).size()
                datasets.append(
                    {
                        "id": self.relative(file),
                        "name": file.name,
                        "seasons": [
                            {"division": str(d), "year": int(y), "games": int(n)} for (d, y), n in seasons.items()
                        ],
                    }
                )
            except (ValueError, OSError, KeyError) as exc:
                problems.append(f"Could not read {file.name}: {exc}")
        tournaments = [{"id": self.relative(p), "name": p.name} for p in sorted(data_dir.glob("*.csv"))]
        return {
            "fits": sorted(fits, key=lambda f: f["modified"], reverse=True),
            "backtests": sorted(backtests, key=lambda f: f["modified"], reverse=True),
            "datasets": datasets,
            "tournaments": tournaments,
            "problems": problems,
            "simulations": sorted(simulations, key=lambda f: f["modified"], reverse=True),
        }

    @staticmethod
    def chain_files(directory: Path) -> list[Path]:
        return sorted(directory.glob("season-*.csv")) or sorted((directory / "fit").glob("season-*.csv"))

    def ratings(self, value: str) -> dict:
        directory = self.path(value)
        teams = pd.read_csv(directory / "results.csv")
        required = {"tid", "name", "link", "mu_off", "mu_def", "mu_home"}
        if not required <= set(teams) or teams.tid.duplicated().any():
            msg = "This run has an invalid results.csv; choose another fit or refit the season."
            raise ValueError(msg)
        for column in ("mu_off", "mu_def", "mu_home"):
            if not np.isfinite(teams[column]).all():
                msg = "The saved ratings contain missing or invalid values. Refit this season."
                raise ValueError(msg)
        teams["offense"] = 40 * (teams.mu_off - teams.mu_off.mean())
        teams["defense"] = 40 * (teams.mu_def - teams.mu_def.mean())
        teams["rating"] = teams.offense + teams.defense
        teams["home"] = 40 * teams.mu_home
        teams = teams.sort_values("rating", ascending=False)
        metadata = {}
        if (directory / "manifest.json").exists():
            metadata = json.loads((directory / "manifest.json").read_text())
        diagnostics = metadata.get("diagnostics")
        if diagnostics is None and (directory / "posterior_summary.csv").exists():
            summary = pd.read_csv(directory / "posterior_summary.csv", index_col=0)
            selected = summary.loc[~summary.index.str.endswith("__")]
            diagnostics = {"max_rhat": float(selected.R_hat.max()), "min_bulk_ess": float(selected.ESS_bulk.min())}
        return {
            "id": value,
            "teams": teams[["tid", "name", "rating", "offense", "defense", "home"]].to_dict("records"),
            "diagnostics": diagnostics,
            "warnings": metadata.get("warnings", []),
            "year": metadata.get("year"),
            "cutoff": metadata.get("cutoff"),
            "has_draws": bool(self.chain_files(directory)),
        }

    def fit(self, value: str):
        files = self.chain_files(self.path(value))
        if not files:
            msg = "This run has ratings but no posterior CSVs. Choose a fit with saved draws."
            raise ValueError(msg)
        signature = tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in files)
        with self._lock:
            if signature not in self._fits:
                fitted = cmdstanpy.from_csv([str(p) for p in files], method="sample")
                if not isinstance(fitted, cmdstanpy.CmdStanMCMC):
                    msg = "These files are not a sampled season fit."
                    raise ValueError(msg)
                self._fits[signature] = fitted
                while len(self._fits) > MAX_CACHED_FITS:
                    self._fits.popitem(last=False)
            self._fits.move_to_end(signature)
            return self._fits[signature]

    def matchup(self, value: str, first: int, second: int, venue: str) -> dict:
        if first == second:
            msg = "Choose two different teams."
            raise ValueError(msg)
        if venue not in {"neutral", "first", "second"}:
            msg = "Choose a neutral court or one of the teams' home courts."
            raise ValueError(msg)
        teams = pd.read_csv(self.path(value) / "results.csv")
        selected = teams.set_index("tid")
        if first not in selected.index or second not in selected.index:
            msg = "A selected team is missing from this fit. Choose the teams again."
            raise ValueError(msg)
        games = pd.DataFrame(
            [
                {
                    "team1_link": selected.loc[first, "link"],
                    "team2_link": selected.loc[second, "link"],
                    "team1_home_adv": int(venue == "first"),
                    "team2_home_adv": int(venue == "second"),
                }
            ]
        )
        probability, margin = posterior_predictions(self.fit(value), teams, games)
        return {
            "first": selected.loc[first, "name"],
            "second": selected.loc[second, "name"],
            "probability": float(probability[0]),
            "margin": float(margin[0]),
            "venue": venue,
        }

    def backtest(self, value: str) -> dict:
        directory = self.path(value)
        metrics = pd.read_csv(directory / "metrics.csv")
        calibration = pd.read_csv(directory / "calibration.csv")
        manifest = json.loads((directory / "manifest.json").read_text())
        # pandas serializes absent baseline margin metrics as JSON null.
        return {
            "id": value,
            "metrics": json.loads(metrics.to_json(orient="records")),
            "calibration": json.loads(calibration.to_json(orient="records")),
            "folds": manifest.get("folds", []),
            "status": manifest.get("status", "unknown"),
        }

    def new_output(self, kind: str) -> Path:
        stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d-%H%M%S")
        path = self.path("output/web") / f"{kind}-{stamp}-{uuid.uuid4().hex[:6]}"
        path.mkdir(parents=True)
        return path

    def fit_season(
        self,
        dataset: str,
        division: str,
        year: int,
        samples: int,
        warmup: int,
        seed: int,
        progress,
        *,
        legacy_home_first: bool = False,
    ):
        from march_madness.stan import Season

        games = pd.read_parquet(self.path(dataset))
        games = games.loc[games.division == division].copy()
        if "team2_home_adv" not in games:
            if not legacy_home_first:
                msg = "This dataset lacks the team-2 home flag. Confirm the legacy home-first setting or prepare a modern dataset."
                raise ValueError(msg)
            games["team2_home_adv"] = 0
        _, teams, data = prep_season_frame(games, year, games_through="ctourn")
        directory = self.new_output("season")
        progress(f"Fitting {len(teams)} teams from {data['N']:,} games. Four chains are running.")
        teams.to_csv(directory / "teams.csv", index=False)
        cmdstanpy.write_stan_json(str(directory / "stan_data.json"), data)
        fit = Season.sample(
            data=data,
            chains=4,
            parallel_chains=4,
            iter_sampling=samples,
            iter_warmup=warmup,
            seed=seed,
            sig_figs=18,
            output_dir=str(directory / "fit"),
            show_progress=False,
        )
        collect_season_outputs(fit, teams).to_csv(directory / "results.csv", index_label="tid")
        summary = fit.summary()
        summary.to_csv(directory / "posterior_summary.csv")
        selected = summary.loc[~summary.index.str.endswith("__")]
        methods = fit.method_variables()
        diagnostics = {
            "max_rhat": float(selected.R_hat.max()),
            "min_bulk_ess": float(selected.ESS_bulk.min()),
            "divergences": int(methods["divergent__"].sum()),
            "max_treedepth_hits": int((methods["treedepth__"] >= MAX_TREEDEPTH).sum()),
        }
        warnings = []
        if (
            diagnostics["max_rhat"] > RHAT_LIMIT
            or diagnostics["min_bulk_ess"] < ESS_LIMIT
            or diagnostics["divergences"]
            or diagnostics["max_treedepth_hits"]
        ):
            warnings.append("Sampling diagnostics need attention. Increase warmup/draws and inspect the saved summary.")
        metadata = {
            "year": year,
            "division": division,
            "dataset": dataset,
            "seed": seed,
            "samples": samples,
            "warmup": warmup,
            "chains": 4,
            "sig_figs": 18,
            "training_games": data["N"],
            "legacy_home_first": legacy_home_first,
            "diagnostics": diagnostics,
            "warnings": warnings,
        }
        (directory / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
        return {"kind": "fit", "id": self.relative(directory)}

    def simulate(self, model: str, tournament: str, seed: int, progress):
        from march_madness.stan import Bracket

        teams = pd.read_csv(self.path(model) / "results.csv").set_index("tid")
        with self.path(tournament).open(newline="", encoding="utf-8") as stream:
            rows = [r for r in csv.reader(stream) if r and not r[0].lstrip().startswith("#")]
        if len(rows) < MIN_BRACKET_TEAMS or len(rows) > MAX_BRACKET_TEAMS or len(rows) & (len(rows) - 1):
            msg = "Use a tournament CSV with 2, 4, 8, 16, 32, 64, or 128 teams in bracket order."
            raise ValueError(msg)
        ids = [int(r[0]) for r in rows]
        if len(set(ids)) != len(ids):
            msg = "The tournament repeats team IDs. Resolve placeholders or duplicate teams first."
            raise ValueError(msg)
        for tid, row in zip(ids, rows, strict=True):
            if tid not in teams.index:
                msg = f"Team {tid} is absent from this fit. Choose the matching season fit."
                raise ValueError(msg)
            if len(row) <= 1 or normalized_team_name(row[1]) != normalized_team_name(teams.loc[tid, "name"]):
                msg = f"Tournament team {tid} does not match {teams.loc[tid, 'name']}. Use the team IDs and names from this fit."
                raise ValueError(msg)
        fit = self.fit(model)
        if "mu_scoring" not in fit.column_names:
            msg = "This fit uses the earlier Stan parameter layout. Refit the season before running brackets."
            raise ValueError(msg)
        directory = self.new_output("tournament")
        progress(f"Simulating {len(ids)} teams from saved posterior draws.")
        try:
            generated = Bracket.generate_quantities(
                data={"N": len(ids), "T": len(teams), "tid": ids},
                previous_fit=fit,
                seed=seed,
                gq_output_dir=str(directory / "draws"),
            )
        except RuntimeError as exc:
            if "does not sum to zero" in str(exc):
                msg = "This fit's CSVs have insufficient precision for sum-to-zero effects. Refit the season using the current app."
                raise ValueError(msg) from exc
            raise
        draws = generated.stan_variable("bracket").astype(np.int8)
        if draws.shape[1:] != (int(math.log2(len(ids))), len(ids)):
            msg = "The bracket model returned an unexpected shape. Inspect the generated quantities logs."
            raise ValueError(msg)
        np.save(directory / "bracket.npy", draws)
        tourney = pd.DataFrame(
            {
                "tid": ids,
                "name": [teams.loc[i, "name"] for i in ids],
                "seed": [r[2] if len(r) > MIN_BRACKET_TEAMS else "" for r in rows],
            }
        )
        tourney.to_csv(directory / "teams.csv", index=False)
        outcomes = pd.DataFrame(draws.mean(axis=0).T, columns=[str(i) for i in range(draws.shape[1])])
        pd.concat([tourney, outcomes], axis=1).to_csv(directory / "outcomes.csv", index=False)
        (directory / "manifest.json").write_text(
            json.dumps({"model": model, "tournament": tournament, "seed": seed, "draws": len(draws)}, indent=2) + "\n"
        )
        return {"kind": "simulation", "id": self.relative(directory)}

    def simulation(self, value: str, draw: int = 0) -> dict:
        directory = self.path(value)
        outcomes = pd.read_csv(directory / "outcomes.csv").fillna("")
        draws = np.load(directory / "bracket.npy", mmap_mode="r", allow_pickle=False)
        if draw < 0 or draw >= len(draws):
            msg = f"Choose a draw between 0 and {len(draws) - 1}."
            raise ValueError(msg)
        winners = draws[draw]
        active = list(range(len(outcomes)))
        rounds = []
        for r in range(winners.shape[0]):
            games = []
            following = []
            for a, b in zip(active[::2], active[1::2], strict=True):
                if winners[r, a] + winners[r, b] != 1:
                    msg = "This saved draw is not a valid bracket. Rerun this simulation."
                    raise ValueError(msg)
                winner = a if winners[r, a] else b
                games.append(
                    {
                        "first": str(outcomes.iloc[a]["name"]),
                        "second": str(outcomes.iloc[b]["name"]),
                        "winner": str(outcomes.iloc[winner]["name"]),
                    }
                )
                following.append(winner)
            rounds.append(games)
            active = following
        return {
            "id": value,
            "draw": draw,
            "draws": len(draws),
            "rounds": rounds,
            "outcomes": outcomes.to_dict("records"),
        }


class Jobs:
    """Serialize expensive work; record status independently of browser polling."""

    def __init__(self, workspace: Workspace):
        self.workspace = workspace
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="march-madness")
        self._jobs: dict = {}
        self._lock = threading.Lock()
        for file in sorted(workspace.path("output/web/jobs").glob("*.json"), key=lambda p: p.stat().st_mtime_ns):
            try:
                saved = json.loads(file.read_text())
                if saved["status"] in {"queued", "running"}:
                    saved.update(
                        status="interrupted",
                        message="The server stopped before this job finished. Start a new job to retry.",
                    )
                self._jobs[saved["id"]] = saved
            except (ValueError, OSError, KeyError):
                continue

    def all(self):
        with self._lock:
            return list(self._jobs.values())

    def submit(self, kind: str, operation) -> str:
        with self._lock:
            if any(j["status"] in {"queued", "running"} for j in self._jobs.values()):
                msg = "A job is already running. Let it finish before starting another."
                raise ValueError(msg)
            key = uuid.uuid4().hex
            self._jobs[key] = {
                "id": key,
                "kind": kind,
                "status": "queued",
                "message": "Waiting to start.",
                "result": None,
            }
        self.executor.submit(self._run, key, operation)
        return key

    def _update(self, key: str, **changes):
        with self._lock:
            self._jobs[key] = self._jobs[key] | changes
            snapshot = self._jobs[key].copy()
        directory = self.workspace.path("output/web/jobs")
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{key}.json").write_text(json.dumps(snapshot, indent=2) + "\n")

    def _run(self, key: str, operation):
        try:
            self._update(key, status="running", message="Starting…")
            result = operation(lambda message: self._update(key, message=message))
            self._update(key, status="complete", message="Complete. Results are saved.", result=result)
        except Exception as exc:  # noqa: BLE001 - Every worker failure must become a visible terminal job state.
            self._update(key, status="failed", message=str(exc) or type(exc).__name__)
