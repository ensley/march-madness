from pathlib import Path
from cmdstanpy import CmdStanModel

STAN_DIR = Path(__file__).parent.resolve()

Season = CmdStanModel(stan_file=STAN_DIR / "season.stan")
Predict = CmdStanModel(stan_file=STAN_DIR / "predict.stan")
Bracket = CmdStanModel(stan_file=STAN_DIR / "bracket.stan")
