from click.testing import CliRunner

from march_madness.cli import march_madness


def test_import_succeeds():
    runner = CliRunner()
    result = runner.invoke(march_madness)
    assert result.exit_code == 0
