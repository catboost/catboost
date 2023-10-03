import os
import tarfile

import yatest.common

import yalibrary.tools


def test_jdk_from_package_equals_jdk_tool_from_yaconf_json():
    jdk_path = yatest.common.binary_path(os.path.join('build', 'platform', 'java', 'jdk', 'testing'))
    os.makedirs("extracted")
    with tarfile.open(os.path.join(jdk_path, "jdk.tar")) as tf:
        tf.extractall("extracted")
    jdk_tool_path = yalibrary.tools.toolchain_root('java', None, None)
    with open(os.path.join("extracted", "release")) as jdk_path_release:
        with open(os.path.join(jdk_tool_path, "release")) as jdk_tool_path_release:
            assert jdk_path_release.read() == jdk_tool_path_release.read()
