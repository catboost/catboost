import json

from library.python.testing.custom_linter_util.reporter import LintReport, LintStatus
from yatest.common import output_path, context


def dump_and_load(report):
    report_file = output_path(context.test_name)
    report.dump(report_file=report_file)
    with open(report_file) as f:
        return json.load(f)


def test_empty_report():
    report = LintReport()
    got = dump_and_load(report)
    assert got == {"report": {}}


def test_good_test():
    report = LintReport()
    report.add("file.cpp", LintStatus.GOOD)

    got = dump_and_load(report)

    assert got == {
        "report": {
            "file.cpp": {
                "status": "GOOD",
                "message": "",
                "elapsed": 0.0,
            }
        }
    }


def test_skipped_test():
    report = LintReport()
    report.add("file.cpp", LintStatus.SKIPPED, "Generated file", elapsed=1.0)

    got = dump_and_load(report)

    assert got == {
        "report": {
            "file.cpp": {
                "status": "SKIPPED",
                "message": "Generated file",
                "elapsed": 1.0,
            }
        }
    }


def test_failed_test():
    report = LintReport()
    report.add("file.cpp", LintStatus.FAIL, "Test failed", elapsed=2.0)

    got = dump_and_load(report)

    assert got == {
        "report": {
            "file.cpp": {
                "status": "FAIL",
                "message": "Test failed",
                "elapsed": 2.0,
            }
        }
    }
