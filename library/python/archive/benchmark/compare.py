"""
These tests allow to compare times used to extract all files from a rather huge tar
"""

import os
import pytest
import tarfile


import yatest.common

import library.python.archive as libarchive


tar_file_path = yatest.common.build_path("library/python/archive/benchmark/data/tos.tar")


def test_system_tar():
    paths = ["/bin/tar", "/usr/bin/tar"]
    for path in paths:
        if os.path.exists(path):
            tar_path = path
            break
    else:
        pytest.skip("No system tar found")
    os.makedirs("test_system_tar")
    yatest.common.execute([tar_path, "-xf", tar_file_path, "-C", "test_system_tar"])


def test_python_tar():
    os.makedirs("test_python_tar")
    with tarfile.open(tar_file_path) as tar:
        tar.extractall("test_python_tar")


def test_libarchive_tar():
    os.makedirs("test_libarchive_tar")
    libarchive.extract_tar(tar_file_path, "test_libarchive_tar")
