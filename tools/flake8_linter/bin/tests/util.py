import json
import logging
import mergedeep
import os
import shutil
import subprocess
import tempfile
from configparser import ConfigParser
from dataclasses import dataclass
from textwrap import dedent

from build.plugins.lib.test_const import FLAKE8_PY2_RESOURCE, FLAKE8_PY3_RESOURCE
from yatest.common import work_path, binary_path

# Config paths to reuse in different tests (just for convenience). This is not mandatory config paths.
FLAKE8_CONFIG_FILE = "build/config/flake8.cfg"
MIGRATIONS_CONFIG_FILE = "build/config/migrations.yaml"
DEFAULT_CONFIGS = [FLAKE8_CONFIG_FILE, MIGRATIONS_CONFIG_FILE]

# Pass test parameters to flake8 stub via env variable
STUB_CONFIG_ENV_VAR_NAME = "_FLAKE8_STUB_CONFIG"

logger = logging.getLogger(__name__)


@dataclass
class Flake8Launch:
    flake8_bin: str
    rel_file_paths: list[str]
    config_data: str


@dataclass
class RunTestResult:
    flake8_launches: list[Flake8Launch]
    linter_run_result: subprocess.CompletedProcess
    report_data: dict


class LinterRunner:
    def __init__(self, lint_name: str = "py3_flake8"):
        self._lint_name = lint_name
        self._source_root = tempfile.mkdtemp(prefix="source_root", dir=work_path())
        self._work_root = tempfile.mkdtemp(prefix="work_root", dir=work_path())
        self._params_file = os.path.join(self._work_root, "params.json")
        self._report_file = os.path.join(self._work_root, "report.json")
        self._launch_report_file = os.path.join(self._work_root, "launches.json")
        self._stub_config_file = os.path.join(self._work_root, "stub_config.json")
        self._linter_path = binary_path("tools/flake8_linter/bin/flake8_linter")
        self._global_resources = self._prepare_global_resources()

    def _prepare_global_resources(self):
        global_resource_root = tempfile.mkdtemp(prefix="global_resources", dir=work_path())
        py2_stub_path = os.path.join(global_resource_root, "py2")
        py3_stub_path = os.path.join(global_resource_root, "py3")
        stub_path = binary_path("tools/flake8_linter/bin/tests/stub")
        shutil.copytree(stub_path, py2_stub_path, copy_function=os.link)
        shutil.copytree(stub_path, py3_stub_path, copy_function=os.link)
        return {
            FLAKE8_PY2_RESOURCE: py2_stub_path,
            FLAKE8_PY3_RESOURCE: py3_stub_path,
        }

    def create_source_file(self, rel_file_path: str, data: str):
        abs_file_path = os.path.join(self._source_root, rel_file_path)
        os.makedirs(os.path.dirname(abs_file_path), exist_ok=True)
        with open(abs_file_path, "w") as f:
            f.write(data)

    def create_source_tree(self, rel_file_paths: list[str]):
        for rel_file_path in rel_file_paths:
            self.create_source_file(rel_file_path, "")

    def abs_source_file_path(self, rel_file_path: str):
        return os.path.join(self._source_root, rel_file_path)

    def run_test(
        self,
        file_rel_paths: list[str],
        config_rel_paths: list[str] = DEFAULT_CONFIGS,
        flake8_result: str = "",
        custom_params: dict = {},
        env: dict[str, str] = {},
    ) -> RunTestResult:
        self._prepare_params(config_rel_paths, file_rel_paths, custom_params)
        stub_config = {
            "output": dedent(flake8_result),
            "report_file": self._launch_report_file,
        }
        stub_env = {
            STUB_CONFIG_ENV_VAR_NAME: self._stub_config_file,
        }
        run_env = mergedeep.merge({}, env, stub_env)
        with open(self._stub_config_file, "w") as f:
            json.dump(stub_config, f)
        linter_run_result = subprocess.run(
            [self._linter_path, "--params", self._params_file],
            encoding="utf-8",
            capture_output=True,
            check=False,
            env=run_env,
        )
        logger.debug("Linter run result: %s", str(linter_run_result))

        if os.path.exists(self._report_file):
            with open(self._report_file) as f:
                report_data = json.load(f)
        else:
            report_data = None

        return RunTestResult(self._read_launches(), linter_run_result, report_data)

    def flake8_path(self, global_resource_var_name):
        return self._global_resources[global_resource_var_name]

    def _prepare_params(self, config_rel_paths: list[str], file_rel_paths: list[str], custom_params: dict):
        params = {
            "source_root": self._source_root,
            "project_path": "",
            "output_path": "",
            "lint_name": self._lint_name,
            "depends": {},
            "global_resources": self._global_resources,
            "configs": self._mk_source_abs_path(config_rel_paths),
            "report_file": self._report_file,
            "files": self._mk_source_abs_path(file_rel_paths),
        }
        mergedeep.merge(params, custom_params)
        with open(self._params_file, "w") as f:
            json.dump(params, f)

    def _mk_source_abs_path(self, paths):
        return [self.abs_source_file_path(p) for p in paths]

    def _read_launches(self):
        launches = []
        if os.path.exists(self._launch_report_file):
            with open(self._launch_report_file) as f:
                for line in f:
                    logger.debug("Launch report line: %s", line)
                    launch = json.loads(line)
                    launches.append(Flake8Launch(**launch))
        else:
            logger.debug("Launch report file not found: %s", self._launch_report_file)
        return launches


def assert_configs(got: ConfigParser, expected: ConfigParser):
    got_dict = dict(got["flake8"].items())
    expected_dict = dict(expected["flake8"].items())
    assert got_dict == expected_dict
