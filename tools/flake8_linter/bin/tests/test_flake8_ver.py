import os
import pytest

from . import util
from build.plugins.lib.test_const import FLAKE8_PY2_RESOURCE, FLAKE8_PY3_RESOURCE


@pytest.mark.parametrize(
    "lint_name, global_resource_var_name",
    [
        ("py2_flake8", FLAKE8_PY2_RESOURCE),
        ("py3_flake8", FLAKE8_PY3_RESOURCE),
    ],
)
def test_flake8_version(lint_name, global_resource_var_name):
    test_file = "project/test.py"
    runner = util.LinterRunner(lint_name)
    runner.create_source_tree(util.DEFAULT_CONFIGS + [test_file])
    run_result = runner.run_test([test_file])
    expected_flake8_bin = os.path.join(runner.flake8_path(global_resource_var_name), "flake8")
    assert run_result.flake8_launches[0].flake8_bin == expected_flake8_bin


def test_raise_on_incorrect_lint_name():
    test_file = "project/test.py"
    runner = util.LinterRunner("strange_lint_name")
    runner.create_source_tree(util.DEFAULT_CONFIGS + [test_file])
    run_result = runner.run_test([test_file])
    assert run_result.linter_run_result.returncode != 0
    assert "Unexpected lint name" in run_result.linter_run_result.stderr
