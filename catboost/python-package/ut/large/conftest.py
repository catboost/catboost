import os
import tarfile
import zipfile
import shutil

from filelock import FileLock

import yatest.common


def unpack_python(dst_path, py_ver):
    arch_name = {
        "3.6": "python3.6.tar.gz",
    }[py_ver]
    tar = tarfile.open(yatest.common.binary_path("catboost/python-package/ut/large/pkg/" + arch_name))
    tar.extractall(path=dst_path)
    tar.close()


def unpack_deps(dst_path, py_ver):

    curdir = os.getcwd()

    try:

        os.mkdir(dst_path)
        os.chdir(dst_path)
        deps_dir = "deps"

        if not os.path.exists(deps_dir):
            tar = tarfile.open(yatest.common.binary_path("catboost/python-package/ut/large/pkg/deps.tgz"))
            tar.extractall(path=deps_dir)
            tar.close()

        for d in "catboost", "libs", "dynlibs":
            if os.path.exists(d):
                shutil.rmtree(d)

        deps = [
            os.path.join(deps_dir, dep) for dep in [
                "attrs-21.2.0-py2.py3-none-any.whl",
                "importlib_metadata-2.1.1-py2.py3-none-any.whl",
                "iniconfig-1.1.1-py2.py3-none-any.whl",
                "joblib-1.0.1-py3-none-any.whl",
                "packaging-20.9-py2.py3-none-any.whl",
                "pathlib2-2.3.6-py2.py3-none-any.whl",
                "plotly-4.14.3-py2.py3-none-any.whl",
                "pluggy-0.13.1-py2.py3-none-any.whl",
                "py-1.10.0-py2.py3-none-any.whl",
                "pyparsing-2.4.7-py2.py3-none-any.whl",
                "pytest-6.1.2-py3-none-any.whl",
                "python_dateutil-2.8.2-py2.py3-none-any.whl",
                "pytz-2021.1-py2.py3-none-any.whl",
                "six-1.16.0-py2.py3-none-any.whl",
                "threadpoolctl-2.2.0-py3-none-any.whl",
                "toml-0.10.2-py2.py3-none-any.whl",
                "zipp-1.2.0-py2.py3-none-any.whl"
            ]
        ]

        if py_ver == "3.6":
            deps += [
                os.path.join(deps_dir, dep) for dep in [
                    "numpy-1.16.0-cp36-cp36m-manylinux1_x86_64.whl",
                    "pandas-0.24.0-cp36-cp36m-manylinux1_x86_64.whl",
                    "scikit_learn-0.24.2-cp36-cp36m-manylinux2010_x86_64.whl",
                    "scipy-1.5.4-cp36-cp36m-manylinux1_x86_64.whl"
                ]
            ]

        for dep in deps:
            with zipfile.ZipFile(dep, "r") as f:
                f.extractall("catboost")

        files = [
            yatest.common.source_path("catboost/pytest/lib/common_helpers.py"),
            yatest.common.source_path("catboost/python-package/ut/large/catboost_pytest_lib.py"),
            yatest.common.source_path("catboost/python-package/ut/large/list_plugin.py"),
        ]
        for f in files:
            shutil.copy(f, "catboost")

        libs = os.path.join(deps_dir, "py" + py_ver + "libs.tgz")
        dynlibs = os.path.join(deps_dir, "py" + py_ver + "dynlibs.tgz")

        tar = tarfile.open(libs)
        tar.extractall(path="libs")
        tar.close()

        tar = tarfile.open(dynlibs)
        tar.extractall(path="dynlibs")
        tar.close()
    finally:
        os.chdir(curdir)


def pytest_sessionstart(session):
    test_root = yatest.common.source_path('catboost/python-package/ut/large/')

    python_envs_dir = os.path.join(test_root, 'py_envs')

    python_envs_lock = FileLock(os.path.join(test_root, 'py_envs.lock'))
    with python_envs_lock:
        if os.path.exists(python_envs_dir):
            return

        os.mkdir(python_envs_dir)
        for py_ver in ['3.6']:
            dst_path = os.path.join(python_envs_dir, py_ver)

            whl_dir = yatest.common.source_path("catboost/python-package")

            mk_wheel_env = os.environ.copy()
            mk_wheel_env_keys = list(mk_wheel_env.keys())
            for key in mk_wheel_env_keys:
                if key.startswith("YA"):
                    del mk_wheel_env[key]

            yatest.common.execute(
                [
                    yatest.common.python_path(),
                    "mk_wheel.py",
                    "-DUSE_SYSTEM_PYTHON=" + py_ver,
                    "-DCATBOOST_OPENSOURCE=yes",
                    "-DCFLAGS=-DCATBOOST_OPENSOURCE=yes",
                    "--host-platform-flag", "CATBOOST_OPENSOURCE=yes",
                    "--host-platform-flag", "CFLAGS=-DCATBOOST_OPENSOURCE=yes",
                    "--build-widget=no"
                ],
                cwd=whl_dir,
                env=mk_wheel_env,
            )

            whl_file = None
            for f in os.listdir(whl_dir):
                if f.endswith(".whl") and "cp" + py_ver.replace(".", "") in f:
                    whl_file = os.path.join(whl_dir, f)
                    break

            unpack_deps(dst_path, py_ver)

            with zipfile.ZipFile(whl_file, "r") as f:
                f.extractall(os.path.join(dst_path, "catboost"))

            unpack_python(dst_path, py_ver)
