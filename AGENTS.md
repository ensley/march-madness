# Repository Guidelines

## Project Structure & Module Organization

`src/march_madness/` contains the Python package. CLI commands live in `cli/__init__.py`; `transform.py` prepares game data and collects model outputs; `tournament.py` reads tournament inputs. Bracket rendering and scraping live in `bracket.py` and `scrape.py`. The `stan/` directory contains season, prediction, and bracket models plus shared Stan functions.

`ingest.py` reconciles reports by school IDs; `backtest.py` evaluates frozen pre-tournament forecasts. `tests/` holds automated tests. `data/` contains local Parquet and tournament CSV inputs. Use ignored `output/` for generated results. Compiled Stan executables without extensions are also ignored; keep model sources in `.stan` files.

## Build, Test, and Development Commands

Use Python 3.10 or newer and install Hatch. Run commands from the repository root:

- `hatch run march-madness --help`: inspect the local CLI and available commands.
- `hatch run march-madness prepare-games --help`: inspect JSONL-to-Parquet reconciliation options.
- `hatch build`: build source and wheel distributions using Hatchling.
- `hatch fmt --check`: run the formatting and lint checks used in CI.
- `hatch fmt`: apply formatting and lint fixes.
- `hatch run types:check`: run the configured mypy checks over source and tests.
- `hatch test --cover`: run tests with coverage, matching CI.

Install CmdStan before importing the CLI or running tests: `python -m cmdstanpy.install_cmdstan --version 2.36.0 --cores 2`. CI uses this version; Windows also requires the installer’s `--compiler` option. Importing the Stan wrapper can compile models and requires a working C++ toolchain.

## Coding Style & Naming Conventions

Use four-space indentation, snake_case for functions and variables, PascalCase for classes, and uppercase constants. Follow existing double-quoted strings and type annotations on data-processing interfaces. Keep CLI handling in `cli/` and reusable transformations in package modules. Let Hatch’s formatter and linter determine formatting details.

## Testing Guidelines

Tests use pytest conventions: `tests/test_*.py` files and `test_*` functions. Use Click’s `CliRunner` for CLI tests. Add focused tests for changed parsing, transformations, and command behavior. Coverage includes branch measurement; no minimum percentage is configured. CI tests Python 3.10–3.13 on Linux and Windows.

## Commit & Pull Request Guidelines

Recent commits use short subjects with prefixes such as `ci:`, `test:`, and `style:`, alongside plain descriptions. Prefer a concise, action-oriented subject. PRs should explain the behavior change, identify affected inputs or models, link relevant issues, and report validation commands. Include representative output when changing bracket rendering. Exclude generated sampling results and compiled executables.
