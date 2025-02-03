import errno
import functools
import json
import os
import sys
import threading

import six

_lock = threading.Lock()

_config = None

_relaxed_runtime_allowed = False


class NoRuntimeFormed(NotImplementedError):
    pass


def _set_ya_config(config=None, ya=None):
    global _config
    if config:
        _config = config
    elif ya:

        class Config:
            def __init__(self):
                self.ya = None

        _config = Config()
        _config.ya = ya


def _get_ya_config():
    if _config:
        return _config
    else:
        try:
            import pytest

            return pytest.config
        except (ImportError, AttributeError):
            raise NoRuntimeFormed("yatest.common.* is only available from the testing runtime")


def _get_ya_plugin_instance():
    return _get_ya_config().ya


def _norm_path(path):
    if path is None:
        return None
    assert isinstance(path, six.string_types)
    if "\\" in path:
        raise AssertionError("path {} contains Windows seprators \\ - replace them with '/'".format(path))
    return os.path.normpath(path)


def _is_binary():
    return getattr(sys, 'is_standalone_binary', False)


def _is_relaxed_runtime_allowed():
    global _relaxed_runtime_allowed
    if _relaxed_runtime_allowed:
        return True
    return not _is_binary()


def default_arg0(func):
    return default_arg(func, 0)


def default_arg1(func):
    return default_arg(func, 1)


def default_arg(func, narg):
    # Always try to call func, before checking standaloneness.
    # The context file might be provided and func might return
    # result even if it's not a standalone binary run.
    @functools.wraps(func)
    def wrapper(*args, **kw):
        try:
            return func(*args, **kw)
        except NoRuntimeFormed:
            if _is_relaxed_runtime_allowed():
                if len(args) > narg:
                    return args[narg]
                return None
            raise

    return wrapper


def default_value(value):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kw):
            try:
                return func(*args, **kw)
            except NoRuntimeFormed:
                if _is_relaxed_runtime_allowed():
                    return value
                raise

        return wrapper

    return decorator


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


@default_arg0
def source_path(path=None):
    """
    Get source path inside arcadia
    :param path: path arcadia relative, e.g. yatest.common.source_path('devtools/ya')
    :return: absolute path to the source folder
    """
    return _join_path(_get_ya_plugin_instance().source_root, path)


@default_arg0
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
    from . import runtime_java

    return runtime_java.get_java_path(binary_path(os.path.join('build', 'platform', 'java', 'jdk', 'testing')))


def java_home():
    """
    Get jdk directory path
    """
    from . import runtime_java

    jdk_dir = runtime_java.get_build_java_dir(binary_path('jdk'))
    if not jdk_dir:
        raise Exception(
            "Cannot find jdk - make sure 'jdk' is added to the DEPENDS section and exists for the current platform"
        )
    return jdk_dir


def java_bin():
    """
    Get path to the java binary
    Requires DEPENDS(jdk)
    """
    return os.path.join(java_home(), "bin", "java")


@default_arg0
def data_path(path=None):
    """
    Get path inside atd_ro_snapshot directory
    :param path: path relative to the atd_ro_snaphot directory, e.g. yatest.common.data_path("pers/rerank_service")
    :return: absolute path in arcadia
    """
    return _join_path(source_path("atd_ro_snapshot"), path)


@default_arg0
def output_path(path=None):
    """
    Get path inside the current test suite output dir.
    Placing files to this dir guarantees that files will be accessible after the test suite execution.
    :param path: path relative to the test suite output dir
    :return: absolute path inside the test suite output dir
    """
    return _join_path(_get_ya_plugin_instance().output_dir, path)


@default_arg0
def ram_drive_path(path=None):
    """
    :param path: path relative to the ram drive.
    :return: absolute path inside the ram drive directory or None if no ram drive was provided by environment.
    """
    if 'YA_TEST_RAM_DRIVE_PATH' in os.environ:
        return _join_path(os.environ['YA_TEST_RAM_DRIVE_PATH'], path)
    elif get_param("ram_drive_path"):
        return _join_path(get_param("ram_drive_path"), path)


@default_arg0
def output_ram_drive_path(path=None):
    """
    Returns path inside ram drive directory which will be saved in the testing_out_stuff directory after testing.
    Returns None if no ram drive was provided by environment.
    :param path: path relative to the output ram drive directory
    """
    if 'YA_TEST_OUTPUT_RAM_DRIVE_PATH' in os.environ:
        return _join_path(os.environ['YA_TEST_OUTPUT_RAM_DRIVE_PATH'], path)
    elif _get_ya_plugin_instance().get_context("test_output_ram_drive_path"):
        return _join_path(_get_ya_plugin_instance().get_context("test_output_ram_drive_path"), path)


@default_arg0
def binary_path(path=None):
    """
    Get path to the built binary
    :param path: path to the binary relative to the build directory e.g. yatest.common.binary_path('devtools/ya/bin/ya-bin')
    :return: absolute path to the binary
    """
    path = _norm_path(path)
    return _get_ya_plugin_instance().get_binary(path)


@default_arg0
def work_path(path=None):
    """
    Get path inside the current test suite working directory. Creating files in the work directory does not guarantee
    that files will be accessible after the test suite execution
    :param path: path relative to the test suite working dir
    :return: absolute path inside the test suite working dir
    """
    return _join_path(
        os.environ.get("TEST_WORK_PATH") or _get_ya_plugin_instance().get_context("work_path") or os.getcwd(), path
    )


@default_value("python")
def python_path():
    """
    Get path to the arcadia python.

    Warn: if you are using build with system python (-DUSE_SYSTEM_PYTHON=X) beware that some python bundles
    are built in a stripped-down form that is needed for building, not running tests.
    See comments in the file below to find out which version of python is compatible with tests.
    https://a.yandex-team.ru/arc/trunk/arcadia/build/platform/python/resources.inc
    :return: absolute path to python
    """
    return _get_ya_plugin_instance().python_path


@default_value("valgrind")
def valgrind_path():
    """
    Get path to valgrind
    :return: absolute path to valgrind
    """
    return _get_ya_plugin_instance().valgrind_path


@default_arg1
def get_param(key, default=None):
    """
    Get arbitrary parameter passed via command line
    :param key: key
    :param default: default value
    :return: parameter value or the default
    """
    return _get_ya_plugin_instance().get_param(key, default)


def set_metric_value(name, val):
    """
    Use this method only when your test environment does not support pytest fixtures,
    otherwise you should prefer using https://docs.yandex-team.ru/ya-make/manual/tests/#python
    :param name: name
    :param val: value
    """
    _get_ya_plugin_instance().set_metric_value(name, val)


@default_arg1
def get_metric_value(name, default=None):
    """
    Use this method only when your test environment does not support pytest fixtures,
    otherwise you should prefer using https://docs.yandex-team.ru/ya-make/manual/tests/#python
    :param name: name
    :param default: default
    :return: parameter value or the default
    """
    return _get_ya_plugin_instance().get_metric_value(name, default)


@default_value(lambda _: {})
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
    test_log_path = _get_ya_config().current_test_log_path
    test_out_dir, log_ext = os.path.splitext(test_log_path)
    log_ext = log_ext.strip(".")
    if log_ext.isdigit():
        test_out_dir = os.path.splitext(test_out_dir)[0]
        test_out_dir = test_out_dir + "_" + log_ext
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


@default_value("gdb")
def gdb_path():
    """
    Get path to the gdb
    """
    if _is_relaxed_runtime_allowed():
        return "gdb"
    return _get_ya_plugin_instance().gdb_path


def c_compiler_path():
    """
    Get path to the gdb
    """
    return os.environ.get("YA_CC")


def c_compiler_cmd():
    p = c_compiler_path()
    return [p, '-isystem' + os.path.dirname(os.path.dirname(p)) + '/share/include']


def get_yt_hdd_path(path=None):
    if 'HDD_PATH' in os.environ:
        return _join_path(os.environ['HDD_PATH'], path)


def cxx_compiler_path():
    """
    Get path to the gdb
    """
    return os.environ.get("YA_CXX")


def cxx_compiler_cmd():
    p = cxx_compiler_path()
    return [p, '-isystem' + os.path.dirname(os.path.dirname(p)) + '/share/include']


def global_resources():
    try:
        if "YA_GLOBAL_RESOURCES" in os.environ:
            return json.loads(os.environ.get("YA_GLOBAL_RESOURCES"))
        else:
            return _get_ya_plugin_instance().get_context("ya_global_resources")
    except (TypeError, ValueError):
        return {}


def _register_core(name, binary_path, core_path, bt_path, pbt_path):
    config = _get_ya_config()

    with _lock:
        if not hasattr(config, 'test_cores_count'):
            config.test_cores_count = 0
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
    @default_value(None)
    def build_type(self):
        return _get_ya_plugin_instance().get_context("build_type")

    @property
    @default_value(None)
    def project_path(self):
        return _get_ya_plugin_instance().get_context("project_path")

    @property
    @default_value(False)
    def test_stderr(self):
        return _get_ya_plugin_instance().get_context("test_stderr")

    @property
    @default_value(False)
    def test_debug(self):
        return _get_ya_plugin_instance().get_context("test_debug")

    @property
    @default_value(None)
    def test_traceback(self):
        return _get_ya_plugin_instance().get_context("test_traceback")

    @property
    @default_value(None)
    def test_name(self):
        return _get_ya_config().current_test_name

    @property
    @default_value("test_tool")
    def test_tool_path(self):
        return _get_ya_plugin_instance().get_context("test_tool_path")

    @property
    @default_value(None)
    def retry_index(self):
        return _get_ya_plugin_instance().get_context("retry_index")

    @property
    @default_value(False)
    def sanitize(self):
        """
        Detect if current test run is under sanitizer

        :return: one of `None`, 'address', 'memory', 'thread', 'undefined'
        """
        return _get_ya_plugin_instance().get_context("sanitize")

    @property
    @default_value(lambda _: {})
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
