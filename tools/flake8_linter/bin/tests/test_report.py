import pytest

from . import util

FLAKE8_CONFIG_DATA = """
    [flake8]
    select = E, W, F
    ignore =
        E122,
"""


def test_no_errors():
    test_file = "project/test.py"
    runner = util.LinterRunner()
    runner.create_source_tree(util.DEFAULT_CONFIGS + [test_file])

    run_result = runner.run_test([test_file])

    abs_test_file = runner.abs_source_file_path(test_file)
    file_report = run_result.report_data["report"][abs_test_file]
    assert file_report["status"] == "GOOD"
    assert file_report["message"] == ""
    assert file_report["elapsed"] >= 0.0


def test_skip_markup():
    test_file = "project/test.py"
    flake8_result = """
        [[bad]]
        [[rst]]
    """

    runner = util.LinterRunner()
    runner.create_source_tree(util.DEFAULT_CONFIGS + [test_file])

    run_result = runner.run_test([test_file], flake8_result=flake8_result)

    abs_test_file = runner.abs_source_file_path(test_file)
    file_report = run_result.report_data["report"][abs_test_file]
    assert file_report["status"] == "GOOD"
    assert file_report["message"] == ""
    assert file_report["elapsed"] >= 0.0


@pytest.mark.parametrize(
    "errors",
    [
        [("10", "F401", "Error with row number only")],
        [("10:20", "F401", "Error with row and column numbers")],
        [
            ("10", "F401", "Multiple errors: the first error"),
            ("20", "F402", "Multiple errors: the second error"),
        ],
    ],
)
def test_error_formatting(errors):
    test_file = "project/test.py"
    flake8_result = "[[bad]]\n"
    for file_pos, code, text in errors:
        flake8_result += f"{{test_dir}}/{test_file}:{file_pos}: [{code}] {text}\n"
    flake8_result += "[[rst]]\n"

    runner = util.LinterRunner()
    runner.create_source_tree(util.DEFAULT_CONFIGS + [test_file])

    run_result = runner.run_test([test_file], flake8_result=flake8_result)

    abs_test_file = runner.abs_source_file_path(test_file)
    file_report = run_result.report_data["report"][abs_test_file]
    expected_message_lines = []
    for file_pos, code, text in errors:
        if ":" in file_pos:
            row, col = file_pos.split(":")
            col_with_sep = col + ":"
        else:
            row = file_pos
            col_with_sep = ""
        line = f"[[unimp]]{abs_test_file}[[rst]]:[[alt2]]{row}[[rst]]:[[alt2]]{col_with_sep}[[rst]] [[[alt1]]{code}[[rst]]] [[bad]]{text}[[rst]]"
        expected_message_lines.append(line)

    assert file_report["status"] == "FAIL"
    assert file_report["message"] == "\n".join(expected_message_lines)
    assert file_report["elapsed"] >= 0.0


def test_fail_on_wrong_message():
    test_file = "project/test.py"
    flake8_result = """
        [[bad]]
        Unexpected error message
        [[rst]]
    """

    runner = util.LinterRunner()
    runner.create_source_tree(util.DEFAULT_CONFIGS + [test_file])

    run_result = runner.run_test([test_file], flake8_result=flake8_result)

    assert run_result.linter_run_result.returncode != 0
    assert "Cannot parse flake8 output line" in run_result.linter_run_result.stderr
