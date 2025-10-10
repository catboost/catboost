import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from build.plugins.lib.test_const import BLACK_RESOURCE
from library.python.testing.custom_linter_util import linter_params, reporter
from library.python.testing.style import rules


logger = logging.getLogger(__name__)

SNIPPET_LINES_LIMIT = 20


def get_black_bin(params) -> str:
    black_root = params.global_resources[BLACK_RESOURCE]
    return os.path.join(black_root, 'black')


def run_black(black_bin, filename, args):
    cmd = [black_bin, filename, *args]

    res = subprocess.run(
        cmd,
        capture_output=True,
        encoding='utf8',
        errors='replace',
    )

    return res.returncode, res.stdout if res.returncode else ''


def run_black_safe(black_bin, filename, args):
    try:
        return run_black(black_bin, filename, args)
    except Exception:
        # fast mode failed - retry
        return run_black(black_bin, filename, args + ['--fast'])


def process_file(black_bin, filename, config):
    logger.debug("Check %s", filename)
    args = ['--quiet', '--check', '--config', config]

    # Fast path for runs with fix_style option or without errors.
    rc, out = run_black_safe(black_bin, filename, args)
    if rc == 1:
        # black runs 15x+ slower if diff is requested, even for files w/o actual diff.
        # Rerun black in case of found error.
        rc, out = run_black_safe(black_bin, filename, args + ['--diff'])

        if out:
            sys.stdout.write(out)
            lines = out.splitlines(keepends=True)
            # strip diff header with "+++" "---" lines
            lines = lines[2:]
            if len(lines) > SNIPPET_LINES_LIMIT:
                lines = lines[:SNIPPET_LINES_LIMIT]
                lines += ["[[rst]]..[truncated].. see full diff in the stdout file in the logsdir"]
            out = ''.join(lines)

    return out


def main():
    params = linter_params.get_params()

    black_bin = get_black_bin(params)
    style_config_path = Path(params.source_root, params.configs[0])

    report = reporter.LintReport()
    for file_name in params.files:
        start_time = time.perf_counter()

        skip_reason = rules.get_skip_reason(file_name, Path(file_name).read_text(), skip_links=False)
        if skip_reason:
            elapsed = time.perf_counter() - start_time
            report.add(
                file_name,
                reporter.LintStatus.SKIPPED,
                f"Style check is omitted: {skip_reason}",
                elapsed=elapsed,
            )
            continue

        error = process_file(black_bin, file_name, style_config_path)
        elapsed = time.perf_counter() - start_time

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
