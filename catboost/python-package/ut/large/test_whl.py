import os
import pytest
import shutil

import yatest.common


PYTHON_PACKAGE_DIR = os.path.join("catboost", "python-package")


def prepare_all(py_ver):
    dst_path = yatest.common.source_path(os.path.join(PYTHON_PACKAGE_DIR, "ut", "large", 'py_envs', py_ver))

    python_binary = os.path.abspath(os.path.join(dst_path, "python", "bin", "python" + py_ver))
    python_env = {
        "PYTHONPATH": ":".join([os.path.join(dst_path, d) for d in ["catboost", "libs", "dynlibs"]]),
        "LD_LIBRARY_PATH": os.path.join(dst_path, "python/lib/x86_64-linux-gnu"),
    }

    return python_binary, python_env


@pytest.mark.parametrize("py_ver", ["3.6"])
def test_wheel(py_ver):
    python_binary, python_env = prepare_all(py_ver)

    catboost_test_script = yatest.common.source_path(os.path.join(PYTHON_PACKAGE_DIR, "ut", "medium", "run_catboost.py"))

    catboost_source_data_path = yatest.common.source_path(os.path.join("catboost", "pytest", "data", "adult"))
    catboost_temp_data_path = os.path.join(yatest.common.test_output_path(), "data", "adult")
    shutil.copytree(catboost_source_data_path, catboost_temp_data_path)

    yatest.common.execute(
        [python_binary, catboost_test_script],
        env=python_env,
        cwd=yatest.common.test_output_path()
    )

    hnsw_test_script = yatest.common.source_path(os.path.join(PYTHON_PACKAGE_DIR, "ut", "medium", "run_catboost_hnsw.py"))

    hnsw_source_data_path = yatest.common.source_path(os.path.join("library", "python", "hnsw", "ut", "data", "floats_60000"))
    hnsw_temp_data_path = os.path.join(yatest.common.test_output_path(), "hnsw_data")
    os.mkdir(hnsw_temp_data_path)
    shutil.copy(hnsw_source_data_path, hnsw_temp_data_path)

    yatest.common.execute(
        [python_binary, hnsw_test_script],
        env=python_env,
        cwd=yatest.common.test_output_path()
    )
