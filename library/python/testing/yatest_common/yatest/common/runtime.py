import errno
import functools
import json
import os
import threading

import six


_lock = threading.Lock()


def _get_ya_config():
    try:
        import pytest
        return pytest.config
    except (ImportError, AttributeError):
        try:
            import library.python.testing.recipe
            if library.python.testing.recipe.ya:
                return library.python.testing.recipe
        except (ImportError, AttributeError):
            pass
        raise NotImplementedError("yatest.common.* is only available from the testing runtime")


def _get_ya_plugin_instance():
    return _get_ya_config().ya


def _norm_path(path):
    if path is None:
        return None
    assert isinstance(path, six.string_types)
    if "\\" in path:
        raise AssertionError("path {} contains Windows seprators \\ - replace them with '/'".format(path))
    return os.path.normpath(path)


def _join_path(main_path, path):
    if not path:
        return main_path
    return os.path.join(main_path, _norm_path(path))


def not_test(func):
    """
    Mark any function as not a test for py.test
    :param func:
    :return:
    """
    @functools.wraps(func)
    def wrapper(*args, **kwds):
        return func(*args, **kwds)
    setattr(wrapper, '__test__', False)
    return wrapper


def source_path(path=None):
    """
    Get source path inside arcadia
    :param path: path arcadia relative, e.g. yatest.common.source_path('devtools/ya')
    :return: absolute path to the source folder
    """
    return _join_path(_get_ya_plugin_instance().source_root, path)


def build_path(path=None):
    """
    Get path inside build directory
    :param path: path relative to the build directory, e.g. yatest.common.build_path('devtools/ya/bin')
    :return: absolute path inside build directory
    """
    return _join_path(_get_ya_plugin_instance().build_root, path)


def java_path():
    """
    [DEPRECATED] Get path to java
    :return: absolute path to java
    """
    import runtime_java
    return runtime_java.get_java_path(binary_path(os.path.join('contrib', 'tools', 'jdk')))


def java_home():
    """
    Get jdk directory path
    """
    import runtime_java
    jdk_dir = runtime_java.get_build_java_dir(binary_path('jdk'))
    if not jdk_dir:
        raise Exception("Cannot find jdk - make sure 'jdk' is added to the DEPENDS section and exists for the current platform")
    return jdk_dir


def java_bin():
    """
    Get path to the java binary
    """
    return os.path.join(java_home(), "bin", "java")


def data_path(path=None):
    """
    Get path inside arcadia_tests_data directory
    :param path: path relative to the arcadia_tests_data directory, e.g. yatest.common.data_path("pers/rerank_service")
    :return: absolute path inside arcadia_tests_data
    """
    return _join_path(_get_ya_plugin_instance().data_root, path)


def output_path(path=None):
    """
    Get path inside the current test suite output dir.
    Placing files to this dir guarantees that files will be accessible after the test suite execution.
    :param path: path relative to the test suite output dir
    :return: absolute path inside the test suite output dir
    """
    return _join_path(_get_ya_plugin_instance().output_dir, path)


def ram_drive_path(path=None):
    """
    :param path: path relative to the ram drive.
    :return: absolute path inside the ram drive directory or None if no ram drive was provided by environment.
    """
    if 'YA_TEST_RAM_DRIVE_PATH' in os.environ:
        return _join_path(os.environ['YA_TEST_RAM_DRIVE_PATH'], path)


def output_ram_drive_path(path=None):
    """
    Returns path inside ram drive directory which will be saved in the testing_out_stuff directory after testing.
    Returns None if no ram drive was provided by environment.
    :param path: path relative to the output ram drive directory
    """
    if 'YA_TEST_OUTPUT_RAM_DRIVE_PATH' in os.environ:
        return _join_path(os.environ['YA_TEST_OUTPUT_RAM_DRIVE_PATH'], path)


def binary_path(path=None):
    """
    Get path to the built binary
    :param path: path to the binary relative to the build directory e.g. yatest.common.binary_path('devtools/ya/bin/ya-bin')
    :return: absolute path to the binary
    """
    path = _norm_path(path)
    return _get_ya_plugin_instance().get_binary(path)


def work_path(path=None):
    """
    Get path inside the current test suite working directory. Creating files in the work directory does not guarantee
    that files will be accessible after the test suite execution
    :param path: path relative to the test suite working dir
    :return: absolute path inside the test suite working dir
    """
    return _join_path(os.environ.get("TEST_WORK_PATH") or os.getcwd(), path)


def python_path():
    """
    Get path to the arcadia python
    :return: absolute path to python
    """
    return _get_ya_plugin_instance().python_path


def valgrind_path():
    """
    Get path to valgrind
    :return: absolute path to valgrind
    """
    return _get_ya_plugin_instance().valgrind_path


def get_param(key, default=None):
    """
    Get arbitrary parameter passed via command line
    :param key: key
    :param default: default value
    :return: parameter value or the default
    """
    return _get_ya_plugin_instance().get_param(key, default)


def get_param_dict_copy():
    """
    Return copy of dictionary with all parameters. Changes to this dictionary do *not* change parameters.

    :return: copy of dictionary with all parameters
    """
    return _get_ya_plugin_instance().get_param_dict_copy()


@not_test
def test_output_path(path=None):
    """
    Get dir in the suite output_path for the current test case
    """
    import pytest
    test_out_dir = os.path.splitext(pytest.config.current_test_log_path)[0]
    try:
        os.makedirs(test_out_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    return _join_path(test_out_dir, path)


def project_path(path=None):
    """
    Get path in build root relating to build_root/project path
    """
    return _join_path(os.path.join(build_path(), context.project_path), path)


def gdb_path():
    """
    Get path to the gdb
    """
    return _get_ya_plugin_instance().gdb_path


def c_compiler_path():
    """
    Get path to the gdb
    """
    return os.environ.get("YA_CC")


def cxx_compiler_path():
    """
    Get path to the gdb
    """
    return os.environ.get("YA_CXX")


def global_resources():
    try:
        return json.loads(os.environ.get("YA_GLOBAL_RESOURCES"))
    except (TypeError, ValueError):
        return {}


def _register_core(name, binary_path, core_path, bt_path, pbt_path):
    config = _get_ya_config()

    with _lock:
        config.test_cores_count += 1
        count_str = '' if config.test_cores_count == 1 else str(config.test_cores_count)

    log_entry = config.test_logs[config.current_item_nodeid]
    if binary_path:
        log_entry['{} binary{}'.format(name, count_str)] = binary_path
    if core_path:
        log_entry['{} core{}'.format(name, count_str)] = core_path
    if bt_path:
        log_entry['{} backtrace{}'.format(name, count_str)] = bt_path
    if pbt_path:
        log_entry['{} backtrace html{}'.format(name, count_str)] = pbt_path


@not_test
def test_source_path(path=None):
    return _join_path(os.path.join(source_path(), context.project_path), path)


class Context(object):
    """
    Runtime context
    """

    @property
    def build_type(self):
        return _get_ya_plugin_instance().get_context("build_type")

    @property
    def project_path(self):
        return _get_ya_plugin_instance().get_context("project_path")

    @property
    def test_stderr(self):
        return _get_ya_plugin_instance().get_context("test_stderr")

    @property
    def test_debug(self):
        return _get_ya_plugin_instance().get_context("test_debug")

    @property
    def test_traceback(self):
        return _get_ya_plugin_instance().get_context("test_traceback")

    @property
    def test_name(self):
        import pytest
        return pytest.config.current_test_name

    @property
    def sanitize(self):
        """
        Detect if current test run is under sanitizer

        :return: one of `None`, 'address', 'memory', 'thread', 'undefined'
        """
        return _get_ya_plugin_instance().get_context("sanitize")

    @property
    def flags(self):
        _flags = _get_ya_plugin_instance().get_context("flags")
        if _flags:
            _flags_dict = dict()
            for f in _flags:
                key, value = f.split('=', 1)
                _flags_dict[key] = value
            return _flags_dict
        else:
            return dict()

    def get_context_key(self, key):
        return _get_ya_plugin_instance().get_context(key)


context = Context()
