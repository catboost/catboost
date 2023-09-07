import io
import pytest
from configparser import ConfigParser

from . import util

FLAKE8_CONFIG_DATA = """
    [flake8]
    select = E, W, F
    ignore =
        E122,
"""


@pytest.mark.parametrize(
    "file_name, noqa_line_no, is_scanned",
    [
        ("__init__.py", 1, True),
        ("__init__.py", 3, True),
        ("__init__.py", 4, False),
        ("not_init.py", 1, False),
    ],
)
def test_scanned_line_count(file_name, noqa_line_no, is_scanned):
    test_file = "project/" + file_name

    test_file_data = io.StringIO()
    for lno in range(1, 10):
        test_file_data.write("pass")
        if lno == noqa_line_no:
            test_file_data.write(" # flake8 noqa: F401")
        test_file_data.write("\n")

    runner = util.LinterRunner()
    runner.create_source_file(util.FLAKE8_CONFIG_FILE, FLAKE8_CONFIG_DATA)
    runner.create_source_file(util.MIGRATIONS_CONFIG_FILE, "")
    runner.create_source_file(test_file, test_file_data.getvalue())

    run_result = runner.run_test([test_file])

    assert len(run_result.flake8_launches) == 1

    launch = run_result.flake8_launches[0]
    got_config = ConfigParser()
    got_config.read_string(launch.config_data)
    if is_scanned:
        assert "F401" in got_config["flake8"]["ignore"]
    else:
        assert "F401" not in got_config["flake8"]["ignore"]


def test_not_F401():
    test_file = "project/__init__.py"

    test_file_data = """
        pass # flake8 noqa: F777, F401
    """

    runner = util.LinterRunner()
    runner.create_source_file(util.FLAKE8_CONFIG_FILE, FLAKE8_CONFIG_DATA)
    runner.create_source_file(util.MIGRATIONS_CONFIG_FILE, "")
    runner.create_source_file(test_file, test_file_data)

    run_result = runner.run_test([test_file])

    assert len(run_result.flake8_launches) == 1

    launch = run_result.flake8_launches[0]
    got_config = ConfigParser()
    got_config.read_string(launch.config_data)
    assert "F401" in got_config["flake8"]["ignore"]
    assert "F777" not in got_config["flake8"]["ignore"]
    assert "Disabling ['F777'] checks" in run_result.linter_run_result.stdout
