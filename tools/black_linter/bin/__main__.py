import contextlib
import io
import logging
import os
import sys
import time
from pathlib import Path

import yalibrary.term.console as console
from library.python.testing.custom_linter_util import linter_params, reporter

import black
import black.files
import black.mode
import black.report


logger = logging.getLogger(__name__)

SNIPPET_LINES_LIMIT = 20


@contextlib.contextmanager
def collect_stdout(stream):
    sys.stdout.flush()
    old = sys.stdout
    sys.stdout = stream
    yield stream
    stream.flush()
    sys.stdout = old


def run_black(filename, wr_back, mode, report, fast):
    # black prints diff to stdout
    bb = io.BytesIO()
    stream = io.TextIOWrapper(
        buffer=bb,
        encoding=sys.stdout.encoding,
        write_through=True,
    )

    with collect_stdout(stream):
        black.reformat_one(
            Path(filename),
            fast=fast,
            write_back=wr_back,
            mode=mode,
            report=report,
        )

    return bb.getvalue().decode()


def run_black_safe(filename, wr_back, mode, report):
    try:
        return run_black(filename, wr_back, mode, report, fast=False)
    except Exception:
        # fast mode failed - drop report stats and retry
        report.change_count = 0
        report.same_count = 0
        report.failure_count = 0

        return run_black(filename, wr_back, mode, report, fast=True)


def process_file(filename, config):
    logger.debug("Check %s", filename)

    report = black.report.Report(
        check=True,
        quiet=True,
    )
    mode = black.Mode(
        line_length=config.get("line_length"),
        string_normalization=not config.get("skip_string_normalization"),
    )
    wr_back_without_diff = black.WriteBack.from_configuration(check=True, diff=False)
    # Fast path for runs with fix_style option or without errors.
    error_msg = run_black_safe(filename, wr_back_without_diff, mode, report)
    if report.change_count:
        # black runs 15x+ slower if diff is requested, even for files w/o actual diff.
        # Rerun black in case of found error.
        wr_back_with_diff = black.WriteBack.from_configuration(check=True, diff=True)
        error_msg = run_black_safe(filename, wr_back_with_diff, mode, report)

    if error_msg:
        sys.stdout.write(console.strip_ansi_codes(error_msg))
        lines = error_msg.split(os.linesep)
        # strip diff header with "+++" "---" lines
        lines = lines[2:]
        if len(lines) > SNIPPET_LINES_LIMIT:
            lines = lines[:SNIPPET_LINES_LIMIT]
            lines += ["[[rst]]..[truncated].. see full diff in the stdout file in the logsdir"]
        error_msg = os.linesep.join(lines)
    return error_msg


def main():
    params = linter_params.get_params()

    black_parser_logger = logging.getLogger("blib2to3.pgen2.driver")
    black_parser_logger.setLevel(logging.WARNING)

    style_config_path = Path(params.source_root, params.configs[0])

    report = reporter.LintReport()
    for file_name in params.files:
        black_config = black.parse_pyproject_toml(str(style_config_path))
        start_time = time.time()
        error = process_file(file_name, black_config)
        elapsed = time.time() - start_time

        if error:
            rel_file_name = os.path.relpath(file_name, params.source_root)
            message = "Run [[imp]]ya style {}[[rst]] to fix format\n".format(rel_file_name) + error
            status = reporter.LintStatus.FAIL
        else:
            message = ""
            status = reporter.LintStatus.GOOD
        report.add(file_name, status, message, elapsed=elapsed)

    report.dump(params.report_file)


if __name__ == "__main__":
    main()
