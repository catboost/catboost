import difflib
import os
import subprocess
import time
from pathlib import PurePath

from build.plugins.lib.test_const import CLANG_FORMAT_RESOURCE
from library.python.testing.custom_linter_util import linter_params, reporter
from library.python.testing.style import rules

# Amortize process startup without making a formatting error recheck too many files.
BATCH_SIZE = 32


def main():
    params = linter_params.get_params()

    if 'custom_clang_format' in params.extra_params:
        dep_result = next(
            (
                PurePath(params.depends[dep])
                for dep in params.depends
                if str(PurePath(dep).parent) == params.extra_params['custom_clang_format']
            ),
            None,
        )
        if dep_result is None:
            raise Exception('Could not find clang-format binary')

        if 'custom_clang_format_bin' in params.extra_params:
            # dep_result is not a clang-format binary (package etc)
            clang_format_binary = str(dep_result.parent / params.extra_params['custom_clang_format_bin'])
        else:
            # dep_result is a clang-format binary
            clang_format_binary = str(dep_result)
    else:
        clang_format_binary = os.path.join(params.global_resources[CLANG_FORMAT_RESOURCE], 'clang-format')

    style_config_path = params.configs[0]

    report = reporter.LintReport()
    results = check_files(clang_format_binary, style_config_path, params.files)
    for file_name, (status, message, elapsed) in zip(params.files, results):
        report.add(file_name, status, message, elapsed=elapsed)

    report.dump(params.report_file)


def check_files(clang_format_binary, style_config_path, filenames):
    if len(filenames) < 2:
        return [check_file_with_elapsed(clang_format_binary, style_config_path, filename) for filename in filenames]

    results = [None] * len(filenames)
    files_to_check = []
    for index, filename in enumerate(filenames):
        start_time = time.perf_counter()
        with open(filename, "rb") as f:
            actual_source = f.read()

        skip_reason = rules.get_skip_reason(filename, actual_source, skip_links=False)
        if skip_reason:
            results[index] = (
                reporter.LintStatus.SKIPPED,
                "Style check is omitted: {}".format(skip_reason),
                time.perf_counter() - start_time,
            )
        else:
            files_to_check.append((index, filename, time.perf_counter() - start_time))

    for start in range(0, len(files_to_check), BATCH_SIZE):
        batch = files_to_check[start : start + BATCH_SIZE]
        batch_filenames = [filename for _, filename, _ in batch]
        batch_is_formatted, batch_elapsed = check_batch(clang_format_binary, style_config_path, batch_filenames)
        if len(batch) > 1 and batch_is_formatted:
            for index, _, precheck_elapsed in batch:
                results[index] = (
                    reporter.LintStatus.GOOD,
                    "",
                    precheck_elapsed + batch_elapsed / len(batch),
                )
        else:
            for index, filename, precheck_elapsed in batch:
                status, message, elapsed = check_file_with_elapsed(clang_format_binary, style_config_path, filename)
                results[index] = status, message, precheck_elapsed + elapsed

    return results


def check_batch(clang_format_binary, style_config_path, filenames):
    if len(filenames) < 2:
        return False, 0.0

    command = [
        clang_format_binary,
        '-style=file:' + style_config_path,
        '--dry-run',
        '--Werror',
        *filenames,
    ]
    start_time = time.perf_counter()
    return (
        subprocess.call(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0,
        time.perf_counter() - start_time,
    )


def check_file_with_elapsed(clang_format_binary, style_config_path, filename):
    start_time = time.perf_counter()
    status, message = check_file(clang_format_binary, style_config_path, filename)
    return status, message, time.perf_counter() - start_time


def check_file(clang_format_binary, style_config_path, filename):
    with open(filename, "rb") as f:
        actual_source = f.read()

    skip_reason = rules.get_skip_reason(filename, actual_source, skip_links=False)
    if skip_reason:
        return reporter.LintStatus.SKIPPED, "Style check is omitted: {}".format(skip_reason)

    command = [clang_format_binary, '-assume-filename=' + filename, '-style=file:' + style_config_path]
    styled_source = subprocess.check_output(command, input=actual_source)

    if styled_source == actual_source:
        return reporter.LintStatus.GOOD, ""
    else:
        diff = make_diff(actual_source, styled_source)
        return reporter.LintStatus.FAIL, diff


def make_diff(left, right):
    result = ""
    for line in difflib.unified_diff(left.decode().splitlines(), right.decode().splitlines(), fromfile='L', tofile='R'):
        line = line.rstrip("\n")
        if line:
            if line[0] == "-":
                line = "[[bad]]" + line + "[[rst]]"
            elif line[0] == "+":
                line = "[[good]]" + line + "[[rst]]"
        result += line + "\n"
    return result


if __name__ == "__main__":
    main()
