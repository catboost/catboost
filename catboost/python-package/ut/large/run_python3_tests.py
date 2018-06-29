import os

import yatest.common

from test_whl import prepare_all

TEST_SCRIPT = yatest.common.source_path(os.path.join("catboost/python-package/ut/medium/test.py"))


def update_env(env):
    env["DATA_PATH"] = yatest.common.source_path(os.path.join("catboost/pytest/data"))
    env["BINARY_PATH"] = yatest.common.build_path()


def pytest_generate_tests(metafunc):

    tests = []

    for py_ver in "3.5", "3.6":
        python_bin, python_env = prepare_all(py_ver)

        update_env(python_env)

        yatest.common.execute(
            [
                python_bin,
                "-m",
                "pytest",
                "--collect-only",
                TEST_SCRIPT,
            ],
            env=python_env,
        )
        with open("test-list") as f:
            for line in f:
                line = line.strip()
                if line:
                    test_name = line.split("::", 1)[1]
                    tests.append((py_ver, test_name))

    metafunc.parametrize(("py_ver", "case"), tests)


def test(py_ver, case):
    python_bin, python_env = prepare_all(py_ver)

    update_env(python_env)

    try:
        yatest.common.execute(
            [
                python_bin,
                "-m",
                "pytest",
                TEST_SCRIPT + "::" + case,
            ],
            env=python_env,
            cwd=yatest.common.test_output_path()
        )
    except yatest.common.process.ExecutionError, e:
        stdout = e.execution_result.std_out
        stderr = e.execution_result.std_err
        assert 0, "python%s: %s\n%s\n%s" % (py_ver, case, stdout, stderr)

    canonize_path = os.path.join(yatest.common.test_output_path(), "canonize")
    if os.path.exists(canonize_path):
        canon_files = []
        with open(canonize_path) as f:
            for line in f:
                line = line.strip()
                args = line.split()
                canon_file = os.path.join(yatest.common.test_output_path(), args[0])
                diff_tool = None
                if len(args) > 1:
                    diff_tool = args[1]
                canon_files.append(yatest.common.canonical_file(canon_file, diff_tool=diff_tool, local=True))
        return canon_files

