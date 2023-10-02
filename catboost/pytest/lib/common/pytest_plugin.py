import os
import sys

import pytest

sys.path += [
    os.path.join(os.environ['CMAKE_SOURCE_DIR'], 'library', 'python', 'pytest'),
    os.path.join(os.environ['CMAKE_SOURCE_DIR'], 'library', 'python', 'testing'),
    os.path.join(os.environ['CMAKE_SOURCE_DIR'], 'library', 'python', 'testing', 'yatest_common')
]

import yatest.common

import yatest_lib.ya

from . import tools
from . import *


pytest_config = None


class CanonicalProcessor(object):
    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_call(self, item):  # noqa
        def get_wrapper(obj):
            def wrapper(*args, **kwargs):
                obj(*args, **kwargs)
            return wrapper

        item.obj = get_wrapper(item.obj)
        yield


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    global pytest_config
    pytest_config = config
    config.ya = yatest_lib.ya.Ya(
        source_root=os.environ['CMAKE_SOURCE_DIR'],
        build_root=os.environ['CMAKE_BINARY_DIR'],
        output_dir=os.environ['TEST_OUTPUT_DIR']
    )
    config.sanitizer_extra_checks = False
    yatest.common.runtime._set_ya_config(config=config)

    config.pluginmanager.register(
        CanonicalProcessor()
    )


def pytest_runtest_setup(item):
    pytest_config.current_item_nodeid = item.nodeid
    class_name, test_name = tools.split_node_id(item.nodeid)
    test_log_path = tools.get_test_log_file_path(pytest_config.ya.output_dir, class_name, test_name)
    pytest_config.current_test_log_path = test_log_path
