import os
import pytest
import tarfile
import zipfile
import shutil

import yatest.common


def unpack_python(py_ver):
    arch_name = {
        "3.4": "python3_4_3_linux.tar.gz",
        "3.5": "python3.5.tar.gz",
        "3.6": "python3.6.tar.gz",
    }[py_ver]
    tar = tarfile.open(yatest.common.binary_path("catboost/python-package/ut/large/pkg/" + arch_name))
    tar.extractall(path=py_ver)
    tar.close()


def unpack_deps(py_ver):

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
            "six-1.11.0-py2.py3-none-any.whl",
            "pytz-2018.4-py2.py3-none-any.whl",
            "python_dateutil-2.7.3-py2.py3-none-any.whl",
        ]
    ]

    if py_ver == "3.4":
        deps += [
            os.path.join(deps_dir, "/numpy-1.14.3-cp34-cp34m-manylinux1_x86_64.whl"),
        ]

    if py_ver == "3.5":
        deps += [
            os.path.join(deps_dir, dep) for dep in [
                "numpy-1.14.3-cp35-cp35m-manylinux1_x86_64.whl",
                "pandas-0.23.0-cp35-cp35m-manylinux1_x86_64.whl",
            ]
        ]

    if py_ver == "3.6":
        deps += [
            os.path.join(deps_dir, dep) for dep in [
                "numpy-1.14.3-cp36-cp36m-manylinux1_x86_64.whl",
                "pandas-0.23.0-cp36-cp36m-manylinux1_x86_64.whl",
            ]
        ]

    for dep in deps:
        with zipfile.ZipFile(dep, "r") as f:
            f.extractall("catboost")

    files = [os.path.join(deps_dir, "numbers.py")]
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



@pytest.mark.parametrize("py_ver", ["3.5", "3.6"])
def test_wheel(py_ver):
    whl_dir = yatest.common.source_path("catboost/python-package")

    yatest.common.execute(
        [
            yatest.common.python_path(),
            "mk_wheel.py",
            "-DUSE_SYSTEM_PYTHON=" + py_ver,
            "-DCATBOOST_OPENSOURCE=yes",
            "-DCFLAGS=-DCATBOOST_OPENSOURCE=yes",
            "--host-platform-flag", "CATBOOST_OPENSOURCE=yes",
        ],
        cwd=whl_dir,
    )

    whl_file = None
    for f in os.listdir(whl_dir):
        if f.endswith(".whl") and "cp" + py_ver.replace(".", "") in f:
            whl_file = os.path.join(whl_dir, f)
            break

    unpack_deps(py_ver)

    with zipfile.ZipFile(whl_file, "r") as f:
        f.extractall("catboost")

    unpack_python(py_ver)
    python_binary = os.path.join(py_ver, "python", "bin", "python" + py_ver)

    test_script = yatest.common.source_path(os.path.join("catboost/python-package/ut/medium/run_catboost.py"))

    yatest.common.execute(
        [
            python_binary,
            test_script,
        ],
        env={
            "PYTHONPATH": ":".join([os.path.join(os.getcwd(), d) for d in ["catboost", "libs", "dynlibs"]]),
            "LD_LIBRARY_PATH": os.path.join(os.getcwd(), py_ver, "python/lib/x86_64-linux-gnu"),
        },
    )
