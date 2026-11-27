import os
import shlex
import shutil
from pathlib import Path

import pytest
import yatest.common as yc


@pytest.fixture
def compiler(tmp_path):
    # Execute a copied binary: neither compiler source nor data is supplied here.
    binary = tmp_path / "cython"
    shutil.copyfile(yc.binary_path("contrib/tools/cython_py2/bin/cython"), binary)
    binary.chmod(0o755)
    return binary


def compiler_environment(entry_point=None):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("Y_PYTHON_SOURCE_ROOT", None)
    if entry_point is not None:
        env["Y_PYTHON_ENTRY_POINT"] = entry_point
    return env


def generate(compiler, tmp_path, source, *args, success=True, cplus=True):
    path = tmp_path / "sample.pyx"
    path.write_bytes(source if isinstance(source, bytes) else source.encode())
    output = tmp_path / ("sample.cpp" if cplus else "sample.c")
    env = compiler_environment()
    result = yc.execute(
        [str(compiler), "-3", "-I", str(tmp_path), str(path), "-o", str(output)]
        + (["--cplus"] if cplus else [])
        + list(args),
        cwd=str(tmp_path),
        env=env,
        check_exit_code=success,
    )
    return result, output.read_text() if output.exists() else ""


def test_version(compiler):
    result = yc.execute([str(compiler), "--version"])
    assert b"3.0.12" in result.std_out + result.std_err


@pytest.mark.parametrize(
    "source, marker",
    [
        ("from libc.math cimport sqrt\ndef f(double x): return sqrt(x)\n", "sqrt"),
        ("from libcpp.vector cimport vector\ncdef vector[int] values\n", "<vector>"),
        ("from cpython.array cimport array\ncdef array value\n", "arrayobject"),
        ("from posix.unistd cimport getpid\ndef pid(): return getpid()\n", "unistd.h"),
        ("cimport openmp\ndef threads(): return openmp.omp_get_max_threads()\n", "omp.h"),
        ("def total(double[:] values): return sum(values)\n", "__Pyx_memviewslice"),
        ("ctypedef fused number:\n    int\n    double\ndef square(number x): return x*x\n", "double"),
        ("cdef public int exposed(int x): return x\ncdef api int called(int x): return x\n", "exposed"),
    ],
)
def test_codegen(compiler, tmp_path, source, marker):
    _, output = generate(compiler, tmp_path, source)
    assert marker in output
    if "cdef public" in source:
        assert (tmp_path / "sample.h").exists()
        assert (tmp_path / "sample_api.h").exists()


def test_project_shadow_and_explicit_standard_root(compiler, tmp_path):
    (tmp_path / "libc").mkdir()
    (tmp_path / "libc/math.pxd").write_text(
        "cdef extern from \"project_math.h\":\n    double project_function(double)\n"
    )
    # Both legacy supplier roots are mapped to this tool's builtin namespace.
    standard_root = "/missing/arcadia/contrib/tools/cython_py2/Cython/Includes"
    _, output = generate(
        compiler,
        tmp_path,
        "from libc.math cimport project_function\ndef f(double x): return project_function(x)\n",
        "-I",
        standard_root,
    )
    assert '"project_math.h"' in output
    _, output = generate(
        compiler, tmp_path, "from libcpp.utility cimport pair\ncdef pair[int, int] value\n", "-I", standard_root
    )
    assert "<utility>" in output


def test_encoding_and_relative_include(compiler, tmp_path):
    (tmp_path / "declarations.pxi").write_bytes(b"# coding: latin-1\ncdef int value = 7 # caf\xe9\n")
    _, output = generate(compiler, tmp_path, b'\xef\xbb\xbfinclude "declarations.pxi"\ndef f(): return value\n')
    assert "declarations.pxi" in output


def test_empty_builtin_and_missing_import(compiler, tmp_path):
    generate(compiler, tmp_path, "cimport cpython\ndef f(): return 1\n")
    result, _ = generate(
        compiler, tmp_path, "from libc.does_not_exist cimport missing\ndef f(): return missing\n", success=False
    )
    assert result.exit_code != 0
    assert b"does_not_exist" in result.std_err
    assert b"from libc.does_not_exist" in result.std_err


def test_c_codegen(compiler, tmp_path):
    _, output = generate(
        compiler, tmp_path, "from libc.math cimport sqrt\ndef f(double x): return sqrt(x)\n", cplus=False
    )
    assert "sqrt" in output
    assert (tmp_path / "sample.c").exists()


def test_numpy_declarations_for_python2(compiler, tmp_path):
    numpy_root = yc.source_path("contrib/python/numpy/py2/numpy/random/mtrand")
    _, output = generate(
        compiler, tmp_path, "cimport numpy as np\ndef ndim(np.ndarray value): return np.PyArray_NDIM(value)\n", "-2", "-I", numpy_root
    )
    assert "numpy/arrayobject.h" in output


def test_ufunc(compiler, tmp_path):
    _, output = generate(
        compiler, tmp_path, """import cython
@cython.ufunc
cdef double square(double value): return value * value
"""
    )
    assert "PyUFunc_FromFuncAndData" in output


def test_python2_language(compiler, tmp_path):
    _, output = generate(compiler, tmp_path, "def f():\n    print 'legacy'\n", "-2")
    assert "legacy" in output


@pytest.mark.parametrize("source_root", [False, True])
def test_project_root_with_supplier_like_path_keeps_its_declaration(compiler, tmp_path, source_root):
    directory = tmp_path / "project/contrib/tools/cython/Cython/Includes/libc"
    directory.mkdir(parents=True)
    (directory / "math.pxd").write_text('cdef extern from "project_math.h":\n    double project_function(double)\n')
    _, output = generate(
        compiler,
        tmp_path,
        "from libc.math cimport project_function\ndef f(double x): return project_function(x)\n",
        *(["--source-root", str(tmp_path)] if source_root else []),
        "-I",
        str(directory.parent),
    )
    assert '"project_math.h"' in output


def test_compile_time_bytes_uses_current_python3_host(compiler, tmp_path):
    _, output = generate(compiler, tmp_path, "DEF value = bytes([0x06])\ndef f(): return value\n")
    assert "\\006" in output
    assert '"[6]"' not in output


def test_annotation(compiler, tmp_path):
    generate(compiler, tmp_path, "def f(): return 1\n", "--annotate")
    assert (tmp_path / "sample.html").exists()


@pytest.mark.parametrize("explicit, check_expected", [(None, False), ("False", True)])
def test_exception_default_allows_explicit_override(compiler, tmp_path, explicit, check_expected):
    args = [] if explicit is None else ["-X", "legacy_implicit_noexcept=" + explicit]
    result, _ = generate(
        compiler,
        tmp_path,
        "cdef void work() nogil: pass\ncdef void caller() noexcept nogil: work()\n",
        *args,
    )
    assert (b"Exception check after calling 'work'" in result.std_err) == check_expected


@pytest.mark.parametrize("platform, expected", [("LINUX", "Linux"), ("DARWIN", "Darwin"), ("WINDOWS", "Windows")])
def test_compile_time_platform_uses_target(compiler, tmp_path, platform, expected):
    _, output = generate(
        compiler, tmp_path, 'DEF target = "requested:" + UNAME_SYSNAME\ndef platform(): return target\n', "--target-platform", platform
    )
    assert "requested:" + expected in output


def test_compile_time_platform_allows_explicit_override(compiler, tmp_path):
    _, output = generate(
        compiler, tmp_path, 'DEF target = "requested:" + UNAME_SYSNAME\ndef platform(): return target\n',
        "--target-platform", "WINDOWS", "-E", "UNAME_SYSNAME=Darwin",
    )
    assert "requested:Darwin" in output
    assert "requested:Windows" not in output


def test_coverage_reads_embedded_source(compiler, tmp_path):
    env = compiler_environment(entry_point=":main")
    env.pop("PYTHON_COVERAGE_CYTHON_BUILD_ROOT", None)
    result = yc.execute(
        [str(compiler), "-c", "from Cython import Utils\n"
         "with Utils.OpenFile('cython/Cython/Utility/CommonTypes.c', is_standalone=True) as source:\n"
         "    print(source.read(), end='')\n"],
        cwd=str(tmp_path),
        env=env,
    )
    assert result.std_out == Path(yc.source_path("contrib/tools/cython_py2/Cython/Utility/CommonTypes.c")).read_bytes()


def test_builtin_declaration_filenames_do_not_depend_on_checkout_depth(compiler, tmp_path):
    outputs = []
    for directory in [tmp_path / "first", tmp_path / "deeper/second"]:
        directory.mkdir(parents=True)
        _, output = generate(
            compiler,
            directory,
            "from cpython.array cimport array, clone\ndef copy(array value): return clone(value, 1, False)\n",
            "--source-root",
            str(directory),
        )
        assert "cpython/array.pxd" in output
        outputs.append(output)
    assert outputs[0] == outputs[1]


def test_depfile_tracks_project_files_and_compiler(compiler, tmp_path):
    (tmp_path / "project.pxd").write_text("cdef int project_value\n")
    (tmp_path / "declarations.pxi").write_text("cdef int included_value\n")
    generate(
        compiler,
        tmp_path,
        'from libc.math cimport sqrt\ncimport project\ninclude "declarations.pxi"\n'
        'def f(double x): return sqrt(x)\n',
        "-M",
        cplus=False,
    )
    depfile = (tmp_path / "sample.c.dep").read_text()
    assert "/__cython_resources__/" not in depfile
    prerequisites = shlex.split(depfile.replace("\\\n", "").split(":", 1)[1])
    assert set(prerequisites) == {compiler.name, "sample.pyx", "project.pxd", "declarations.pxi"}
    assert len(prerequisites) == len(set(prerequisites))


@pytest.mark.parametrize(
    "path, expected",
    [
        ("/__cython_resources__/", True),
        ("/__cython_resources__/Cython/Utility/Code.c", True),
        (r"\__cython_resources__\Cython\Includes\libc\math.pxd", True),
        ("/__cython_resources__other/file", False),
        ("/project/__cython_resources__/file", False),
        (r"C:\project\module.pxd", False),
        ("module.pxd", False),
    ],
)
def test_resource_path_recognition(compiler, path, expected):
    result = yc.execute(
        [
            str(compiler),
            "-c",
            "import sys; from Cython._ResourceFiles import is_resource_path; print(is_resource_path(sys.argv[1]))",
            path,
        ],
        env=compiler_environment(entry_point=":main"),
    )
    assert result.std_out.strip() == str(expected).encode()
