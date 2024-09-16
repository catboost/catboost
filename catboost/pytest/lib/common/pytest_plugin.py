from collections import defaultdict

import os
import sys

import tempfile
import shutil

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


# base path -> test_name -> list of files to canonize
results_to_canonize = defaultdict(lambda: defaultdict())


def get_canonical_name(item):
    class_name, test_name = tools.split_node_id(item.nodeid)
    filename = "{}.{}".format(class_name.split('.')[0], test_name)
    if not filename:
        filename = "test"
    filename = filename.replace("[", "_").replace("]", "_")
    filename = tools.normalize_filename(filename)
    return filename

def get_files_to_canonize(test_result):
    if test_result is None:
        return []
    if not isinstance(test_result, list):
        test_result = [test_result]
    return [e['uri'][7:] for e in test_result]



class CanonicalProcessor(object):
    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_call(self, item):  # noqa
        def get_wrapper(obj):
            def wrapper(*args, **kwargs):
                global results_to_canonize

                canonical_name = get_canonical_name(item)

                res = obj(*args, **kwargs)
                files_to_canonize = get_files_to_canonize(res)
                results_to_canonize[os.path.dirname(item.path)][canonical_name] = files_to_canonize
            return wrapper

        item.obj = get_wrapper(item.obj)
        yield

class WorkdirProcessor(object):
    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_call(self, item):  # noqa
        def get_wrapper(obj):
            def wrapper(*args, **kwargs):
                test_output_path = yatest.common.test_output_path()
                # TODO: have to create in standard tmp dir because of max path length issues on Windows
                work_dir = tempfile.mkdtemp(prefix='work_dir_')
                prev_cwd = None
                try:
                    prev_cwd = os.getcwd()
                except Exception:
                    pass
                os.chdir(work_dir)
                try:
                    obj(*args, **kwargs)
                finally:
                    os.chdir(prev_cwd if prev_cwd else test_output_path)

                # delete only if test succeeded, otherwise leave for debugging
                shutil.rmtree(work_dir)

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
        WorkdirProcessor()
    )
    config.pluginmanager.register(
        CanonicalProcessor()
    )


def pytest_runtest_setup(item):
    pytest_config.current_item_nodeid = item.nodeid
    class_name, test_name = tools.split_node_id(item.nodeid)
    test_log_path = tools.get_test_log_file_path(pytest_config.ya.output_dir, class_name, test_name)
    pytest_config.current_test_log_path = test_log_path
