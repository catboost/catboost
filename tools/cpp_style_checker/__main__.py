import difflib
import json
import os
import subprocess
import time
import yaml
from pathlib import PurePath

from build.plugins.lib.test_const import CLANG_FORMAT_RESOURCE
from library.python.testing.custom_linter_util import linter_params, reporter
from library.python.testing.style import rules


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

    with open(style_config_path) as f:
        style_config = yaml.safe_load(f)
    style_config_json = json.dumps(style_config)

    report = reporter.LintReport()
    for file_name in params.files:
        start_time = time.time()
        status, message = check_file(clang_format_binary, style_config_json, file_name)
        elapsed = time.time() - start_time
        report.add(file_name, status, message, elapsed=elapsed)

    report.dump(params.report_file)


def check_file(clang_format_binary, style_config_json, filename):
    with open(filename, "rb") as f:
        actual_source = f.read()

    skip_reason = rules.get_skip_reason(filename, actual_source, skip_links=False)
    if skip_reason:
        return reporter.LintStatus.SKIPPED, "Style check is omitted: {}".format(skip_reason)

    command = [clang_format_binary, '-assume-filename=' + filename, '-style=' + style_config_json]
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
