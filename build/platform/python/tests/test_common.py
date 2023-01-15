import subprocess

import pytest

from build.platform.python.tests import testlib

PYTHON_VERSIONS = ["2.7", "3.4", "3.5", "3.6", "3.7", "3.8"]


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
    restrictions = {
        "2.7": ['lzma'],
        "3.4": [],
        "3.5": [],
        "3.6": [],
        "3.7": ['ssl'],  # ubuntu 12.04
        "3.8": ['ssl']  # ubuntu 12.04
    }
    imports = ['pkg_resources', 'pip', 'setuptools', 'sqlite3', 'ssl', 'bz2', 'lzma', 'zlib', 'curses', 'readline']  # see DEVTOOLS-7297
    for imp in imports:
        if imp in restrictions[pyver]:
            continue
        subprocess.check_call([testlib.get_python_bin(pyver), '-c', 'import ' + imp])
