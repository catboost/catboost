import os
import subprocess

import yatest.common


def get_python_bin(ver):
    res_name = 'EXTERNAL_PYTHON{}_RESOURCE_GLOBAL'.format(ver.replace('.', ''))
    gr = yatest.common.global_resources()
    if res_name in gr:
        bindir = os.path.join(gr[res_name], 'python', 'bin')
        if ('python' + ver) in os.listdir(bindir):
            return os.path.join(bindir, 'python' + ver)
        return os.path.join(bindir, 'python')

    raise AssertionError("Resource '{}' is not available: {}".format(res_name, gr))


def check_python_version(version):
    ver = subprocess.check_output([get_python_bin(version), '-V'], stderr=subprocess.STDOUT).decode('utf-8')
    assert version in ver
