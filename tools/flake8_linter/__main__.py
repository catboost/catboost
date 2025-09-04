import configparser
import hashlib
import itertools
import io
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from typing import Generator

from devtools.ya.test.programs.test_tool.lib.migrations_config import load_yaml_config, MigrationsConfig
from library.python.testing.custom_linter_util import linter_params, reporter
from build.plugins.lib.test_const import FLAKE8_PY2_RESOURCE, FLAKE8_PY3_RESOURCE

logger = logging.getLogger(__name__)

ALLOWED_IGNORES = {"F401"}
# Supports both default and pylint formats
FLAKE_LINE_RE = r"^(.*?):(\d+):(\d+:)? \[(\w+\d+)\] (.*)"
FLAKE8_CONFIG_INDEX = 0
MIGRATIONS_CONFIG_INDEX = 1


def get_flake8_bin(params) -> str:
    if params.lint_name == "py2_flake8":
        flake8_root = params.global_resources[FLAKE8_PY2_RESOURCE]
    elif params.lint_name == "flake8":
        flake8_root = params.global_resources[FLAKE8_PY3_RESOURCE]
    else:
        raise RuntimeError("Unexpected lint name: {}".format(params.lint_name))
    return os.path.join(flake8_root, "flake8")


def get_migrations_config(params) -> MigrationsConfig:
    if params.extra_params.get("DISABLE_FLAKE8_MIGRATIONS", "no") == "yes":
        return MigrationsConfig()
    config_path = os.getenv("_YA_TEST_FLAKE8_CONFIG")
    if config_path is None and len(params.configs) > 1:
        config_path = params.configs[MIGRATIONS_CONFIG_INDEX]

    if not config_path:
        return MigrationsConfig()
    else:
        logger.debug("Loading flake8 migrations: %s", config_path)
        migrations = load_yaml_config(config_path)
        logger.debug("Building migration config")
        return MigrationsConfig(migrations)


def get_flake8_config(
    flake8_config: str, migrations_config: MigrationsConfig, source_root: str, file_path: str
) -> str | None:
    arc_rel_file_path = os.path.relpath(file_path, source_root)
    if migrations_config.is_skipped(arc_rel_file_path):
        return None
    exceptions = migrations_config.get_exceptions(arc_rel_file_path)
    if exceptions:
        logger.info("Ignore flake8 exceptions %s for file %s", str(list(exceptions)), arc_rel_file_path)

    if os.path.basename(file_path) == "__init__.py":
        exceptions |= get_noqa_exceptions(file_path)

    if exceptions:
        new_config = configparser.ConfigParser()
        new_config.read(flake8_config)  # https://bugs.python.org/issue16058 Why don't use deepcopy
        new_config["flake8"]["ignore"] += "\n" + "\n".join(x + "," for x in sorted(exceptions))

        config_stream = io.StringIO()
        new_config.write(config_stream)
        config_hash = hashlib.md5(config_stream.getvalue().encode()).hexdigest()
        config_path = config_hash + ".config"
        if not os.path.exists(config_path):
            with open(config_path, "w") as f:
                f.write(config_stream.getvalue())
        return config_path
    else:
        return flake8_config


def get_noqa_exceptions(file_path: str) -> set:
    additional_exceptions = get_file_ignores(file_path)
    validate_exceptions(additional_exceptions)
    return additional_exceptions & ALLOWED_IGNORES


def get_file_ignores(file_path: str) -> set:
    file_ignore_regex = re.compile(r"#\s*flake8\s+noqa:\s*(.*)")
    with open(file_path) as afile:
        # looking for ignores only in the first 3 lines
        for line in itertools.islice(afile, 3):
            if match := file_ignore_regex.search(line):
                ignores = match.group(1).strip()
                if ignores:
                    ignores = re.split(r"\s*,\s*", ignores)
                    return set(ignores)
    return set()


def validate_exceptions(exceptions: set) -> None:
    if exceptions - ALLOWED_IGNORES:
        logger.error(
            "Disabling %s checks. Only %s can be suppressed in the __init__.py files using # flake8 noqa",
            str(list(exceptions - ALLOWED_IGNORES)),
            str(list(ALLOWED_IGNORES)),
        )


def run_flake8_for_dir(flake8_bin: str, source_root: str, config: str, check_files: list[str]) -> dict[str, list[str]]:
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.debug("flake8 temp dir: %s", temp_dir)
        for f in check_files:
            copy_file_path = os.path.join(temp_dir, os.path.relpath(f, source_root))
            os.makedirs(os.path.dirname(copy_file_path), exist_ok=True)
            shutil.copyfile(f, copy_file_path)
        flake8_res = run_flake8(flake8_bin, temp_dir, config)
        return get_flake8_results(flake8_res, source_root, temp_dir)


def run_flake8(flake8_bin: str, dir_path: str, config: str) -> list[str]:
    cmd = [flake8_bin, dir_path, config]
    res = subprocess.run(cmd, capture_output=True, encoding="utf8", errors="replace")
    if res.stderr:
        logger.debug("flake8 stderr: %s", res.stderr)
    return res.stdout.split("\n") if res.returncode else []


def get_flake8_results(flake8_res: list[str], source_root: str, temp_dir: str) -> dict[str, list[str]]:
    flake8_errors_map = defaultdict(list)
    for line in iterate_over_results(flake8_res):
        match = re.match(FLAKE_LINE_RE, line)
        if not match:
            raise RuntimeError("Cannot parse flake8 output line: '{}'".format(line))
        file_path, row, col_with_sep, code, text = match.groups()
        file_path = file_path.replace(temp_dir, source_root)
        if col_with_sep is None:
            col_with_sep = ""
        colorized_line = f"[[unimp]]{file_path}[[rst]]:[[alt2]]{row}[[rst]]:[[alt2]]{col_with_sep}[[rst]] [[[alt1]]{code}[[rst]]] [[bad]]{text}[[rst]]"
        flake8_errors_map[file_path].append(colorized_line)
    return flake8_errors_map


def iterate_over_results(flake8_res: list[str]) -> Generator[str, None, None]:
    to_skip = {"[[bad]]", "[[rst]]"}
    for line in flake8_res:
        if line and line not in to_skip:
            yield line


def main():
    params = linter_params.get_params()
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format="%(asctime)s: %(levelname)s: %(message)s")

    flake8_bin = get_flake8_bin(params)
    flake8_config = params.configs[FLAKE8_CONFIG_INDEX]
    migrations_config = get_migrations_config(params)
    source_root = params.source_root

    logger.debug("Constructing flake8 config")
    config_map = defaultdict(list)
    report = reporter.LintReport()

    skipped_files = set()
    for file_path in params.files:
        config_path = get_flake8_config(flake8_config, migrations_config, source_root, file_path)
        if config_path:
            config_map[config_path].append(file_path)
        else:
            skipped_files.add(file_path)

    logger.debug("Configuration:\n%s", str(config_map))

    flake8_errors_map = {}
    for config_path, check_files in config_map.items():
        flake8_errors_map.update(run_flake8_for_dir(flake8_bin, source_root, config_path, check_files))

    report = reporter.LintReport()
    for file_path in params.files:
        if file_path in skipped_files:
            report.add(file_path, reporter.LintStatus.SKIPPED, "Skipped by config")
        elif file_path in flake8_errors_map:
            message = "\n".join(flake8_errors_map[file_path])
            report.add(file_path, reporter.LintStatus.FAIL, message)
        else:
            report.add(file_path, reporter.LintStatus.GOOD)
    report.dump(params.report_file)


if __name__ == "__main__":
    main()
