import os
import subprocess

import yatest.common


def get_python_bin(ver):
    if '.' in ver:
        ver = ver.replace('.', '')

    res_name = 'EXTERNAL_PYTHON{}_RESOURCE_GLOBAL'.format(ver)
    gr = yatest.common.global_resources()
    if res_name in gr:
        bindir = os.path.join(gr[res_name], 'python', 'bin')
        if 'python3' in os.listdir(bindir):
            return os.path.join(bindir, 'python3')
        return os.path.join(bindir, 'python')

    raise AssertionError("Resource '{}' is not available: {}".format(res_name, gr))


def check_python_version(version):
    ver = subprocess.check_output([get_python_bin(version), '-V'], stderr=subprocess.STDOUT).decode('utf-8')
    assert version in ver
