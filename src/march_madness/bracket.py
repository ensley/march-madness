"""NCAA Tournament Bracket Visualization using Rich."""

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

ROUND_32_INDEX = 0
SWEET_16_INDEX = 1
ELITE_8_INDEX = 2
FINAL_FOUR_INDEX = 3
FINAL_INDEX = 4
CHAMPION_INDEX = 5


def _team_or_tbd(teams: list[str], index: int) -> str:
    return teams[index] if index < len(teams) else "TBD"


def get_bracket_winners(predictions: np.ndarray, team_names: np.ndarray) -> list[list[str]]:
    """
    Convert prediction matrix to winners for each round.

    Args:
        predictions: 64x6 numpy array where 1 = team advances, 0 = team eliminated
        team_names: 64x1 array of team names

    Returns:
        List of lists, where each inner list contains winners for that round
    """
    rounds = []

    # Round 1: 64 teams -> 32 winners
    # Teams are paired: 0v1, 2v3, 4v5, etc.
    round1_winners = []
    for i in range(0, 64, 2):
        if predictions[i, 0] == 1:
            round1_winners.append(team_names[i])
        else:
            round1_winners.append(team_names[i + 1])
    rounds.append(round1_winners)

    # Subsequent rounds: use the prediction matrix
    # Round 2: 32 -> 16, Round 3: 16 -> 8, etc.
    current_teams = list(range(64))
    for round_idx in range(1, 6):
        # Find teams that advanced to this round (have 1s in previous columns)
        advancing = [team_idx for team_idx in current_teams if predictions[team_idx, round_idx - 1] == 1]

        # Now find winners of this round
        round_winners = []
        for i in range(0, len(advancing), 2):
            team_a = advancing[i]
            team_b = advancing[i + 1] if i + 1 < len(advancing) else advancing[i]
            if predictions[team_a, round_idx] == 1:
                round_winners.append(team_names[team_a])
            else:
                round_winners.append(team_names[team_b])
        rounds.append(round_winners)
        current_teams = advancing

    return rounds


def draw_bracket(predictions: np.ndarray, team_names: np.ndarray, console: Console | None = None) -> None:
    """
    Draw an NCAA tournament bracket using Rich.

    Args:
        predictions: 64x6 numpy array of 1s and 0s indicating wins
        team_names: 64-element array of team name strings
    """
    if console is None:
        console = Console()

    # Flatten team_names if needed
    if team_names.ndim > 1:
        team_names = team_names.flatten()

    # Get winners for each round
    rounds = get_bracket_winners(predictions, team_names)

    # Build the bracket display
    # Left side: teams 0-31 (regions 1 & 2)
    # Right side: teams 32-63 (regions 3 & 4)

    # Calculate max team name width for alignment
    max_name_len = max(len(str(name)) for name in team_names)
    name_width = min(max_name_len, 16)  # Cap at 16 chars

    def truncate(name: str, width: int = name_width) -> str:
        name = str(name)
        return name[:width] if len(name) > width else name.ljust(width)

    # Create simplified bracket display using tables
    table = Table(title="NCAA Tournament Bracket", show_header=True, header_style="bold cyan")

    # Add columns for each round
    table.add_column("R64", justify="left", width=name_width + 2)
    table.add_column("R32", justify="left", width=name_width + 2)
    table.add_column("S16", justify="left", width=name_width + 2)
    table.add_column("E8", justify="left", width=name_width + 2)
    table.add_column("F4", justify="center", width=name_width + 2)
    table.add_column("Final", justify="center", width=name_width + 4)
    table.add_column("F4", justify="center", width=name_width + 2)
    table.add_column("E8", justify="right", width=name_width + 2)
    table.add_column("S16", justify="right", width=name_width + 2)
    table.add_column("R32", justify="right", width=name_width + 2)
    table.add_column("R64", justify="right", width=name_width + 2)

    # Build rows showing bracket progression
    # Left side: teams 0-31, Right side: teams 32-63

    # Display champion at top
    champion = rounds[CHAMPION_INDEX][0] if len(rounds) > CHAMPION_INDEX and len(rounds[CHAMPION_INDEX]) > 0 else "TBD"
    console.print()
    console.print(Panel(f"[bold yellow]🏆 CHAMPION: {champion} 🏆[/bold yellow]", expand=False), justify="center")
    console.print()

    # Final Four display
    ff_table = Table(show_header=True, header_style="bold magenta", title="Final Four")
    ff_table.add_column("Left Semifinal", justify="center", width=20)
    ff_table.add_column("Championship", justify="center", width=20)
    ff_table.add_column("Right Semifinal", justify="center", width=20)

    f4_teams = rounds[FINAL_FOUR_INDEX] if len(rounds) > FINAL_FOUR_INDEX else ["TBD"] * 4
    finals = rounds[FINAL_INDEX] if len(rounds) > FINAL_INDEX else ["TBD"] * 2

    ff_table.add_row(f"{_team_or_tbd(f4_teams, 0)}", "", f"{_team_or_tbd(f4_teams, 2)}")
    ff_table.add_row("vs", f"{_team_or_tbd(finals, 0)} vs {_team_or_tbd(finals, 1)}", "vs")
    ff_table.add_row(f"{_team_or_tbd(f4_teams, 1)}", "", f"{_team_or_tbd(f4_teams, 3)}")

    console.print(ff_table, justify="center")
    console.print()

    # Display regions
    region_names = ["EAST", "WEST", "SOUTH", "MIDWEST"]

    for region_idx in range(4):
        region_start = region_idx * 16
        r1_start = region_idx * 8
        r2_start = region_idx * 4
        r3_start = region_idx * 2
        e8_idx = region_idx

        region_table = Table(
            title=f"[bold]{region_names[region_idx]} REGION[/bold]", show_header=True, header_style="bold green"
        )
        region_table.add_column("Seed", justify="center", width=4)
        region_table.add_column("Team", justify="left", width=name_width)
        region_table.add_column("R32", justify="left", width=name_width)
        region_table.add_column("S16", justify="left", width=name_width)
        region_table.add_column("E8", justify="left", width=name_width)

        # Standard bracket seeding order
        seed_order = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]

        for i, seed in enumerate(seed_order):
            team_idx = region_start + i
            team = truncate(team_names[team_idx])

            # Determine which matchups this team is part of
            r1_match = r1_start + i // 2
            r1_winner = rounds[ROUND_32_INDEX][r1_match] if len(rounds) > ROUND_32_INDEX else ""

            r2_match = r2_start + i // 4
            r2_winner = rounds[SWEET_16_INDEX][r2_match] if len(rounds) > SWEET_16_INDEX else ""

            r3_match = r3_start + i // 8
            r3_winner = rounds[ELITE_8_INDEX][r3_match] if len(rounds) > ELITE_8_INDEX else ""

            rounds[FINAL_FOUR_INDEX][e8_idx] if len(rounds) > FINAL_FOUR_INDEX else ""

            # Only show round winners at appropriate row positions
            r32_display = truncate(r1_winner) if i % 2 == 0 else ""
            s16_display = truncate(r2_winner) if i % 4 == 0 else ""
            e8_display = truncate(r3_winner) if i % 8 == 0 else ""

            region_table.add_row(str(seed), team, r32_display, s16_display, e8_display)

            # Add separator after each pair of teams
            if i % 2 == 1 and i < len(seed_order) - 1:
                region_table.add_row("", "", "", "", "")

        console.print(region_table)
        console.print()


def draw_bracket_compact(predictions: np.ndarray, team_names: np.ndarray, console: Console | None = None) -> None:
    """
    Draw a compact single-view bracket using box drawing characters.

    Args:
        predictions: 64x6 numpy array of 1s and 0s indicating wins
        team_names: 64-element array of team name strings
    """
    if console is None:
        console = Console()

    if team_names.ndim > 1:
        team_names = team_names.flatten()

    rounds = get_bracket_winners(predictions, team_names)

    # Print the bracket
    console.print("[bold cyan]NCAA Tournament Bracket[/bold cyan]")
    console.print()

    # Champion
    champion = rounds[CHAMPION_INDEX][0] if len(rounds) > CHAMPION_INDEX and rounds[CHAMPION_INDEX] else "TBD"
    console.print(f"[bold yellow]🏆 Champion: {champion}[/bold yellow]", justify="center")
    console.print()

    # Final Four
    f4 = rounds[FINAL_FOUR_INDEX] if len(rounds) > FINAL_FOUR_INDEX else ["TBD"] * 4
    finals = rounds[FINAL_INDEX] if len(rounds) > FINAL_INDEX else ["TBD"] * 2

    console.print("[bold magenta]Final Four:[/bold magenta]")
    console.print(f"  {_team_or_tbd(f4, 0)} vs {_team_or_tbd(f4, 1)} → {finals[0] if finals else 'TBD'}")
    console.print(f"  {_team_or_tbd(f4, 2)} vs {_team_or_tbd(f4, 3)} → {_team_or_tbd(finals, 1)}")
    console.print()

    # Elite 8
    e8 = rounds[FINAL_FOUR_INDEX] if len(rounds) > FINAL_FOUR_INDEX else ["TBD"] * 4
    console.print("[bold green]Elite Eight:[/bold green]")
    for i, team in enumerate(e8):
        console.print(f"  {['East', 'West', 'South', 'Midwest'][i]}: {team}")
    console.print()

    # Full bracket by region
    region_names = ["EAST", "WEST", "SOUTH", "MIDWEST"]
    for region_idx in range(4):
        console.print(f"[bold]{region_names[region_idx]}[/bold]")

        region_start = region_idx * 16
        r1_start = region_idx * 8
        region_idx * 4
        region_idx * 2

        for game in range(8):
            t1_idx = region_start + game * 2
            t2_idx = region_start + game * 2 + 1
            r1_idx = r1_start + game

            t1 = team_names[t1_idx]
            t2 = team_names[t2_idx]
            winner = rounds[ROUND_32_INDEX][r1_idx] if rounds else "TBD"

            console.print(f"  {t1} vs {t2} → [cyan]{winner}[/cyan]")
        console.print()


if __name__ == "__main__":
    # Demo with random data
    np.random.seed(42)

    # Create sample team names
    teams = np.array([f"Team_{i+1}" for i in range(64)])

    # Create random predictions (ensuring valid bracket)
    predictions = np.zeros((64, 6), dtype=int)

    # Round 1: pick winners randomly
    for i in range(0, 64, 2):
        winner = np.random.choice([i, i + 1])
        predictions[winner, 0] = 1

    # Subsequent rounds
    for round_idx in range(1, 6):
        advancing = np.where(predictions[:, round_idx - 1] == 1)[0]
        for i in range(0, len(advancing), 2):
            winner = np.random.choice([advancing[i], advancing[i + 1]]) if i + 1 < len(advancing) else advancing[i]
            predictions[winner, round_idx] = 1

    console = Console()
    draw_bracket(predictions, teams, console)
