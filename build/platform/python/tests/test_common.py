import subprocess

import pytest

from build.platform.python.tests import testlib

PYTHON_VERSIONS = ["2.7", "3.4", "3.5", "3.6"]  # 3.7, 3.8 are not runnable


@pytest.mark.parametrize("pyver", PYTHON_VERSIONS)
def test_version_matched(pyver):
    testlib.check_python_version(pyver)


@pytest.mark.parametrize("pyver", PYTHON_VERSIONS)
def test_python_max_unicode_bytes(pyver):
    cmd = [testlib.get_python_bin(pyver), '-c', 'import sys; print(sys.maxunicode)']
    maxunicode = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode('utf-8')
    assert int(maxunicode) > 65535, "Found UCS2 build"


@pytest.mark.parametrize("pyver", PYTHON_VERSIONS)
def test_python_imports(pyver):
    imports = {
        "2.7": ['pkg_resources'],
        "3.4": [],
        "3.5": ['pkg_resources'],
        "3.6": [],
    }
    for imp in imports[pyver]:
        subprocess.check_call([testlib.get_python_bin(pyver), '-c', 'import ' + imp])
