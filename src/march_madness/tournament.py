import csv
import pathlib


def read_tournament_file(tourney_file: pathlib.Path) -> list[int]:
    with open(tourney_file, newline="") as f:
        reader = csv.reader(f)
        return [int(row[0]) for row in reader if not row[0].startswith("#")]
