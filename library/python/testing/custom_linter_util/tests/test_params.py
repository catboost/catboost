import json

from library.python.testing.custom_linter_util import linter_params
from yatest.common import work_path


SOURCE_ROOT = "TEST_SOURCE_ROOT"
PROJECT_PATH = "TEST_PROJECT_PATH"
OUTPUT_PATH = "TEST_OUTPUT_PATH"
REPORT_FILE = "TEST_REPORT_FILE"
LINT_NAME = "important-lint"
DEPS = {
    "dep1": "/path/to/dep1",
    "dep2": "/path/to/dep2",
}
GLOBAL_RESOURCES = {
    "TOOL1_GLOBAL_RESOURCES": "/path/to/resource1",
    "TOOL2_GLOBAL_RESOURCES": "/path/to/resource2",
}
CONFIGS = ["path/to/config1", "path/to/config2"]
EXTRA_PARAMS = {
    "var1": "val1",
    "var2": "val2",
}
FILES = ["file1.cpp", "file2.cpp"]

EXPECTED = linter_params.LinterArgs(
    source_root=SOURCE_ROOT,
    project_path=PROJECT_PATH,
    output_path=OUTPUT_PATH,
    report_file=REPORT_FILE,
    lint_name=LINT_NAME,
    depends=DEPS,
    global_resources=GLOBAL_RESOURCES,
    configs=CONFIGS,
    extra_params=EXTRA_PARAMS,
    files=FILES,
)


def test_cmd_line_params():
    raw_args = [
        "--source-root", SOURCE_ROOT,
        "--project-path", PROJECT_PATH,
        "--output-path", OUTPUT_PATH,
        "--report-file", REPORT_FILE,
        "--lint-name", LINT_NAME,
    ]
    for rel, abs in DEPS.items():
        raw_args += ["--depends", ":".join([rel, abs])]
    for var, path in GLOBAL_RESOURCES.items():
        raw_args += ["--global-resource", ":".join([var, path])]
    for cfg in CONFIGS:
        raw_args += ["--config", cfg]
    for var, val in EXTRA_PARAMS.items():
        raw_args += ["--extra-param", "=".join([var, val])]
    raw_args += FILES

    got = linter_params.get_params(raw_args)

    assert got == EXPECTED


def test_json_params():
    params_file = work_path("params.josn")
    params = {
        "source_root": SOURCE_ROOT,
        "project_path": PROJECT_PATH,
        "output_path": OUTPUT_PATH,
        "report_file": REPORT_FILE,
        "lint_name": LINT_NAME,
        "depends": DEPS,
        "global_resources": GLOBAL_RESOURCES,
        "configs": CONFIGS,
        "extra_params": EXTRA_PARAMS,
        "files": FILES,
    }
    with open(params_file, "w") as f:
        json.dump(params, f)

    got = linter_params.get_params(["--params", params_file])

    assert got == EXPECTED
