import datetime
import re
import time
from typing import Any

import pydantic
import requests
from bs4 import BeautifulSoup
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_fixed,
)
from tenacity.wait import wait_base

console = Console()
progress = Progress(
    SpinnerColumn(),
    TextColumn("[cyan]{task.description}", justify="right"),
    BarColumn(bar_width=None),
    "[progress.percentage]{task.percentage:>3.1f}%",
    "•",
    TimeElapsedColumn(),
    "•",
    TimeRemainingColumn(),
    console=console,
)


class HTTP429Error(BaseException):
    def __init__(self, response) -> None:
        super().__init__("429 Too Many Requests")
        self.headers = response.headers


class retry_if_http_429_error(retry_if_exception):
    def __init__(self):
        def is_http_429_error(exception):
            return isinstance(exception, HTTP429Error)

        super().__init__(predicate=is_http_429_error)


class wait_for_retry_after_header(wait_base):
    def __init__(self, fallback) -> None:
        self.fallback = fallback

    def __call__(self, retry_state: RetryCallState):
        if retry_state.outcome:
            exc = retry_state.outcome.exception()
            if isinstance(exc, HTTP429Error):
                retry_after = exc.headers.get("Retry-After")
                try:
                    return int(retry_after)
                except (TypeError, ValueError):
                    pass

        return self.fallback(retry_state)


def before_sleep(retry_state):
    console.log(
        f"[yellow]⚠ Attempt {retry_state.attempt_number} failed:[/yellow] "
        f"[red]{retry_state.outcome.exception()}[/red] "
        f"— retrying in {retry_state.next_action.sleep:.1f}s..."
    )


@retry(
    retry=retry_if_http_429_error(),
    wait=wait_for_retry_after_header(fallback=wait_fixed(5)),
    stop=stop_after_attempt(3),
    before_sleep=before_sleep,
)
def get_page_respectfully(url):
    response = requests.get(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:148.0) Gecko/20100101 Firefox/148.0"
        },
    )
    if response.status_code == requests.codes["too_many_requests"]:
        raise HTTP429Error(response)

    return response.text


class School(pydantic.BaseModel, frozen=True):
    shortname: str
    name: str
    is_valid: bool


class GameLog(pydantic.BaseModel, frozen=True):
    school: School
    opponent: School
    date: datetime.date
    game_type: str
    is_home: bool
    is_neutral_site: bool
    score: int
    opp_score: int
    ot_periods: int

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, GameLog):
            return (
                self.school == other.opponent
                and self.opponent == other.school
                and self.date == other.date
                and self.score == other.opp_score
                and self.opp_score == other.score
            )
        else:
            return super().__eq__(other)

    def __hash__(self):
        return hash(self.date)


def get_gamelogs_url(school: School, year: int) -> str | None:
    if school.is_valid:
        return f"https://sports-reference.com/cbb/schools/{school.shortname}/women/{year}-gamelogs.html"


def find_data_stat(parent, attr: str) -> str | None:
    if tag := parent.find(attrs={"data-stat": attr}):
        return tag.text.strip()


def extract_team_id(tag) -> str:
    if link := tag.a.get("href"):
        result = re.findall(r"cbb/schools/([a-z\-]+)/", link)
        if len(result) > 0:
            return result[0]
        else:
            return "null"
    else:
        return "null"


def get_schools(year: int) -> list[School]:
    url = f"https://www.sports-reference.com/cbb/seasons/women/{year}-school-stats.html"
    soup = BeautifulSoup(get_page_respectfully(url), "html.parser")
    return [
        School(shortname=extract_team_id(tag), name=tag.text.strip(), is_valid=True)
        for tag in soup.find_all("td", attrs={"data-stat": "school_name"})
    ]


def get_gamelogs(school: School, year: int) -> list[GameLog]:
    url = get_gamelogs_url(school, year)
    soup = BeautifulSoup(get_page_respectfully(url), "html.parser")
    rows = soup.find_all(
        lambda tag: tag.name == "tr"
        and tag.has_attr("id")
        and tag.get("id", "").startswith("team_game_log.")
    )

    gamelogs = []
    for row in rows:
        opp_cell = row.find(attrs={"data-stat": "opp_name_abbr"})
        if opp_cell is None:
            continue
        if not opp_cell.find("a"):
            opp = School(
                shortname=opp_cell.text.lower().replace(" ", "-"),
                name=opp_cell.text,
                is_valid=False,
            )
        else:
            opp = School(
                shortname=re.findall(r"cbb/schools/([a-z\-]+)/", opp_cell.a["href"])[0],
                name=opp_cell.text,
                is_valid=True,
            )

        ots = re.findall(r"(\d*)?OT", find_data_stat(row, "overtimes"))
        if not ots:
            ots = 0
        elif len(ots) == 1 and not ots[0]:
            ots = 1
        else:
            ots = ots[0]

        gamelog = GameLog(
            school=school,
            opponent=opp,
            date=datetime.datetime.strptime(
                find_data_stat(row, "date"), "%Y-%m-%d"
            ).date(),
            game_type=find_data_stat(row, "game_type"),
            is_home=find_data_stat(row, "game_location") == "",
            is_neutral_site=find_data_stat(row, "game_location") == "N",
            score=find_data_stat(row, "team_game_score"),
            opp_score=find_data_stat(row, "opp_team_game_score"),
            ot_periods=ots,
        )

        gamelogs.append(gamelog)

    return gamelogs


def scrape_gamelogs(year: int):
    schools = get_schools(year)
    with progress:
        task_id = progress.add_task("Getting gamelogs", total=len(schools))
        for school in schools:
            school_gamelogs = get_gamelogs(school, year)
            yield from school_gamelogs
            time.sleep(5)
            progress.update(task_id, advance=1)
            console.log(
                f"[magenta]{school.name}[/] - [i]{len(school_gamelogs)} games found[/i]"
            )


class Game(pydantic.BaseModel):
    division: str
    year: int
    date: datetime.date
    type: str
    team1_link: str
    team2_link: str
    team1_name: str
    team2_name: str
    team1_score: int
    team2_score: int
    team1_home_adv: int
    team2_home_adv: int
    minutes_played: int


if __name__ == "__main__":
    with open("gamelogs_w.jsonl", "w", encoding="utf-8") as f:
        for gamelog in scrape_gamelogs(2026):
            f.write(gamelog.model_dump_json() + "\n")
