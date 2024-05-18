# coding: utf-8

import base64
import errno
import sys
import os
import logging
import fnmatch
import json
import time
import traceback
import collections
import signal
import inspect
import warnings

import faulthandler
import py
import pytest
import six

import _pytest
import _pytest._io
import _pytest.mark
import _pytest.outcomes
import _pytest.skipping

from _pytest.warning_types import PytestUnhandledCoroutineWarning

from yatest_lib import test_splitter
import yatest.common as yatest_common

from library.python.pytest.plugins.metrics import test_metrics

try:
    import resource
except ImportError:
    resource = None

try:
    import library.python.pytest.yatest_tools as tools
except ImportError:
    # fallback for pytest script mode
    import yatest_tools as tools

try:
    from library.python import filelock
except ImportError:
    filelock = None


import yatest_lib.tools

import yatest_lib.external as canon

import yatest_lib.ya

from library.python.pytest import context

console_logger = logging.getLogger("console")
yatest_logger = logging.getLogger("ya.test")


_pytest.main.EXIT_NOTESTSCOLLECTED = 0
SHUTDOWN_REQUESTED = False

pytest_config = None


def configure_pdb_on_demand():
    import signal

    if hasattr(signal, "SIGUSR1"):
        def on_signal(*args):
            import ipdb
            ipdb.set_trace()

        signal.signal(signal.SIGUSR1, on_signal)


class CustomImporter(object):
    def __init__(self, roots):
        self._roots = roots

    def find_spec(self, fullname, path, target=None):
        return None

    def find_module(self, fullname, package_path=None):
        for path in self._roots:
            full_path = self._get_module_path(path, fullname)

            if os.path.exists(full_path) and os.path.isdir(full_path) and not os.path.exists(os.path.join(full_path, "__init__.py")):
                open(os.path.join(full_path, "__init__.py"), "w").close()

        return None

    def _get_module_path(self, path, fullname):
        return os.path.join(path, *fullname.split('.'))


class YaTestLoggingFileHandler(logging.FileHandler):
    pass


class _TokenFilterFormatter(logging.Formatter):
    def __init__(self, fmt):
        super(_TokenFilterFormatter, self).__init__(fmt)
        self._replacements = []
        if not self._replacements:
            if six.PY2:
                for k, v in os.environ.iteritems():
                    if k.endswith('TOKEN') and v:
                        self._replacements.append(v)
            elif six.PY3:
                for k, v in os.environ.items():
                    if k.endswith('TOKEN') and v:
                        self._replacements.append(v)
            self._replacements = sorted(self._replacements)

    def _filter(self, s):
        for r in self._replacements:
            s = s.replace(r, "[SECRET]")

        return s

    def format(self, record):
        return self._filter(super(_TokenFilterFormatter, self).format(record))


def setup_logging(log_path, level=logging.DEBUG, *other_logs):
    logs = [log_path] + list(other_logs)
    root_logger = logging.getLogger()
    for i in range(len(root_logger.handlers) - 1, -1, -1):
        if isinstance(root_logger.handlers[i], YaTestLoggingFileHandler):
            root_logger.handlers.pop(i).close()
    root_logger.setLevel(level)
    for log_file in logs:
        file_handler = YaTestLoggingFileHandler(log_file)
        log_format = '%(asctime)s - %(levelname)s - %(name)s - %(funcName)s: %(message)s'
        file_handler.setFormatter(_TokenFilterFormatter(log_format))
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)


class YaHookspec:
    @pytest.hookspec(firstresult=True)
    def pytest_ya_summarize_error(self, report):
        pass


@pytest.hookimpl(trylast=True)
def pytest_ya_summarize_error(report):
    return get_formatted_error(report)


def pytest_addhooks(pluginmanager):
    pluginmanager.add_hookspecs(YaHookspec)


def pytest_addoption(parser):
    parser.addoption("--build-root", action="store", dest="build_root", default="", help="path to the build root")
    parser.addoption("--dep-root", action="append", dest="dep_roots", default=[], help="path to the dep build roots")
    parser.addoption("--source-root", action="store", dest="source_root", default="", help="path to the source root")
    parser.addoption("--data-root", action="store", dest="data_root", default="", help="path to the arcadia_tests_data root")
    parser.addoption("--output-dir", action="store", dest="output_dir", default="", help="path to the test output dir")
    parser.addoption("--python-path", action="store", dest="python_path", default="", help="path the canonical python binary")
    parser.addoption("--valgrind-path", action="store", dest="valgrind_path", default="", help="path the canonical valgring binary")
    parser.addoption("--test-filter", action="append", dest="test_filter", default=None, help="test filter")
    parser.addoption("--test-file-filter", action="store", dest="test_file_filter", default=None, help="test file filter")
    parser.addoption("--test-param", action="append", dest="test_params", default=None, help="test parameters")
    parser.addoption("--test-log-level", action="store", dest="test_log_level", choices=["critical", "error", "warning", "info", "debug"], default="debug", help="test log level")
    parser.addoption("--mode", action="store", choices=[yatest_lib.ya.RunMode.List, yatest_lib.ya.RunMode.Run], dest="mode", default=yatest_lib.ya.RunMode.Run, help="testing mode")
    parser.addoption("--test-list-file", action="store", dest="test_list_file")
    parser.addoption("--modulo", default=1, type=int)
    parser.addoption("--modulo-index", default=0, type=int)
    parser.addoption("--partition-mode", default='SEQUENTIAL', help="Split tests according to partitoin mode")
    parser.addoption("--split-by-tests", action='store_true', help="Split test execution by tests instead of suites", default=False)
    parser.addoption("--project-path", action="store", default="", help="path to CMakeList where test is declared")
    parser.addoption("--build-type", action="store", default="", help="build type")
    parser.addoption("--flags", action="append", dest="flags", default=[], help="build flags (-D)")
    parser.addoption("--sanitize", action="store", default="", help="sanitize mode")
    parser.addoption("--test-stderr", action="store_true", default=False, help="test stderr")
    parser.addoption("--test-debug", action="store_true", default=False, help="test debug mode")
    parser.addoption("--root-dir", action="store", default=None)
    parser.addoption("--ya-trace", action="store", dest="ya_trace_path", default=None, help="path to ya trace report")
    parser.addoption("--ya-version", action="store", dest="ya_version", default=0, type=int, help="allows to be compatible with ya and the new changes in ya-dev")
    parser.addoption(
        "--test-suffix", action="store", dest="test_suffix", default=None, help="add suffix to every test name"
    )
    parser.addoption("--gdb-path", action="store", dest="gdb_path", default="", help="path the canonical gdb binary")
    parser.addoption("--collect-cores", action="store_true", dest="collect_cores", default=False, help="allows core dump file recovering during test")
    parser.addoption("--sanitizer-extra-checks", action="store_true", dest="sanitizer_extra_checks", default=False, help="enables extra checks for tests built with sanitizers")
    parser.addoption("--report-deselected", action="store_true", dest="report_deselected", default=False, help="report deselected tests to the trace file")
    parser.addoption("--pdb-on-sigusr1", action="store_true", default=False, help="setup pdb.set_trace on SIGUSR1")
    parser.addoption("--test-tool-bin", help="Path to test_tool")
    parser.addoption("--test-list-path", dest="test_list_path", action="store", help="path to test list", default="")


def from_ya_test():
    return "YA_TEST_RUNNER" in os.environ


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    global pytest_config
    pytest_config = config

    config.option.continue_on_collection_errors = True

    config.from_ya_test = from_ya_test()
    config.test_logs = collections.defaultdict(dict)
    test_metrics.metrics = {}
    config.suite_metrics = {}
    config.configure_timestamp = time.time()
    context = {
        "project_path": config.option.project_path,
        "test_stderr": config.option.test_stderr,
        "test_debug": config.option.test_debug,
        "build_type": config.option.build_type,
        "test_traceback": config.option.tbstyle,
        "flags": config.option.flags,
        "sanitize": config.option.sanitize,
    }

    if config.option.collectonly:
        config.option.mode = yatest_lib.ya.RunMode.List

    config.ya = yatest_lib.ya.Ya(
        config.option.mode,
        config.option.source_root,
        config.option.build_root,
        config.option.dep_roots,
        config.option.output_dir,
        config.option.test_params,
        context,
        config.option.python_path,
        config.option.valgrind_path,
        config.option.gdb_path,
        config.option.data_root,
    )
    config.option.test_log_level = {
        "critical": logging.CRITICAL,
        "error": logging.ERROR,
        "warning": logging.WARN,
        "info": logging.INFO,
        "debug": logging.DEBUG,
    }[config.option.test_log_level]

    if not config.option.collectonly:
        setup_logging(os.path.join(config.ya.output_dir, "run.log"), config.option.test_log_level)
    config.current_item_nodeid = None
    config.current_test_name = None
    config.test_cores_count = 0
    config.collect_cores = config.option.collect_cores
    config.sanitizer_extra_checks = config.option.sanitizer_extra_checks
    try:
        config.test_tool_bin = config.option.test_tool_bin
    except AttributeError:
        logging.info("test_tool_bin not specified")

    if config.sanitizer_extra_checks:
        for envvar in ['LSAN_OPTIONS', 'ASAN_OPTIONS']:
            if envvar in os.environ:
                os.environ.pop(envvar)
            if envvar + '_ORIGINAL' in os.environ:
                os.environ[envvar] = os.environ[envvar + '_ORIGINAL']

    extra_sys_path = []
    # Arcadia paths from the test DEPENDS section of ya.make
    extra_sys_path.append(os.path.join(config.option.source_root, config.option.project_path))
    # Build root is required for correct import of protobufs, because imports are related to the root
    # (like import devtools.dummy_arcadia.protos.lib.my_proto_pb2)
    extra_sys_path.append(config.option.build_root)

    for path in config.option.dep_roots:
        if os.path.isabs(path):
            extra_sys_path.append(path)
        else:
            extra_sys_path.append(os.path.join(config.option.source_root, path))

    sys_path_set = set(sys.path)
    for path in extra_sys_path:
        if path not in sys_path_set:
            sys.path.append(path)
            sys_path_set.add(path)

    os.environ["PYTHONPATH"] = os.pathsep.join(sys.path)

    if not config.option.collectonly:
        if config.option.ya_trace_path:
            config.ya_trace_reporter = TraceReportGenerator(config.option.ya_trace_path)
        else:
            config.ya_trace_reporter = DryTraceReportGenerator(config.option.ya_trace_path)
    config.ya_version = config.option.ya_version

    sys.meta_path.append(CustomImporter([config.option.build_root] + [os.path.join(config.option.build_root, dep) for dep in config.option.dep_roots]))
    if config.option.pdb_on_sigusr1:
        configure_pdb_on_demand()

    yatest_common.runtime._set_ya_config(config=config)
    # Dump python backtrace in case of any errors
    faulthandler.enable()
    if hasattr(signal, "SIGQUIT"):
        # SIGQUIT is used by test_tool to teardown tests which overruns timeout
        faulthandler.register(signal.SIGQUIT, chain=True)

    if hasattr(signal, "SIGUSR2"):
        signal.signal(signal.SIGUSR2, _graceful_shutdown)

    # register custom markers
    config.addinivalue_line(
        "markers", "xfaildiff: Allows to mark test which is expected to have a diff with canonical data"
    )


session_should_exit = False


def _graceful_shutdown_on_log(should_exit):
    if should_exit:
        pytest.exit("Graceful shutdown requested")


def pytest_runtest_logreport(report):
    _graceful_shutdown_on_log(session_should_exit)


def pytest_runtest_logstart(nodeid, location):
    _graceful_shutdown_on_log(session_should_exit)


def pytest_runtest_logfinish(nodeid, location):
    _graceful_shutdown_on_log(session_should_exit)


def _graceful_shutdown(*args):
    global session_should_exit
    session_should_exit = True
    try:
        import library.python.coverage
        library.python.coverage.stop_coverage_tracing()
    except ImportError:
        pass
    traceback.print_stack(file=sys.stderr)
    capman = pytest_config.pluginmanager.getplugin("capturemanager")
    capman.suspend(in_=True)
    _graceful_shutdown_on_log(not capman.is_globally_capturing())


def _get_rusage():
    return resource and resource.getrusage(resource.RUSAGE_SELF)


def _collect_test_rusage(item):
    if resource and hasattr(item, "rusage"):
        finish_rusage = _get_rusage()
        ya_inst = pytest_config.ya

        def add_metric(attr_name, metric_name=None, modifier=None):
            if not metric_name:
                metric_name = attr_name

            if not modifier:
                def modifier(x):
                    return x

            if hasattr(item.rusage, attr_name):
                ya_inst.set_metric_value(metric_name, modifier(getattr(finish_rusage, attr_name) - getattr(item.rusage, attr_name)))

        for args in [
            ("ru_maxrss", "ru_rss", lambda x: x*1024),  # to be the same as in util/system/rusage.cpp
            ("ru_utime",),
            ("ru_stime",),
            ("ru_ixrss", None, lambda x: x*1024),
            ("ru_idrss", None, lambda x: x*1024),
            ("ru_isrss", None, lambda x: x*1024),
            ("ru_majflt", "ru_major_pagefaults"),
            ("ru_minflt", "ru_minor_pagefaults"),
            ("ru_nswap",),
            ("ru_inblock",),
            ("ru_oublock",),
            ("ru_msgsnd",),
            ("ru_msgrcv",),
            ("ru_nsignals",),
            ("ru_nvcsw",),
            ("ru_nivcsw",),
        ]:
            add_metric(*args)


def _get_item_tags(item):
    tags = []
    for key, value in item.keywords.items():
        if key == 'pytestmark' and isinstance(value, list):
            for mark in value:
                tags.append(mark.name)
        elif isinstance(value, _pytest.mark.MarkDecorator):
            tags.append(key)
    return tags


def pytest_runtest_setup(item):
    item.rusage = _get_rusage()
    pytest_config.test_cores_count = 0
    pytest_config.current_item_nodeid = item.nodeid
    class_name, test_name = tools.split_node_id(item.nodeid)
    test_log_path = tools.get_test_log_file_path(pytest_config.ya.output_dir, class_name, test_name)
    setup_logging(
        os.path.join(pytest_config.ya.output_dir, "run.log"),
        pytest_config.option.test_log_level,
        test_log_path
    )
    pytest_config.test_logs[item.nodeid]['log'] = test_log_path
    pytest_config.test_logs[item.nodeid]['logsdir'] = pytest_config.ya.output_dir
    pytest_config.current_test_log_path = test_log_path
    pytest_config.current_test_name = "{}::{}".format(class_name, test_name)
    separator = "#" * 100
    yatest_logger.info(separator)
    yatest_logger.info(test_name)
    yatest_logger.info(separator)
    yatest_logger.info("Test setup")

    test_item = CrashedTestItem(item.nodeid, item.location[0], pytest_config.option.test_suffix)
    pytest_config.ya_trace_reporter.on_start_test_class(test_item)
    pytest_config.ya_trace_reporter.on_start_test_case(test_item)


def pytest_runtest_teardown(item, nextitem):
    yatest_logger.info("Test teardown")


def pytest_runtest_call(item):
    class_name, test_name = tools.split_node_id(item.nodeid)
    yatest_logger.info("Test call (class_name: %s, test_name: %s)", class_name, test_name)


def pytest_deselected(items):
    config = pytest_config
    if config.option.report_deselected:
        for item in items:
            deselected_item = DeselectedTestItem(item.nodeid, item.location[0], config.option.test_suffix)
            config.ya_trace_reporter.on_start_test_class(deselected_item)
            config.ya_trace_reporter.on_finish_test_case(deselected_item)
            config.ya_trace_reporter.on_finish_test_class(deselected_item)


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(items, config):

    def filter_items(filters):
        filtered_items = []
        deselected_items = []
        for item in items:
            canonical_node_id = str(CustomTestItem(item.nodeid, item.location[0], pytest_config.option.test_suffix))
            matched = False
            for flt in filters:
                if "::" not in flt and "*" not in flt:
                    flt += "*"  # add support for filtering by module name
                if canonical_node_id.endswith(flt) or fnmatch.fnmatch(tools.escape_for_fnmatch(canonical_node_id), tools.escape_for_fnmatch(flt)):
                    matched = True
            if matched:
                filtered_items.append(item)
            else:
                deselected_items.append(item)

        config.hook.pytest_deselected(items=deselected_items)
        items[:] = filtered_items

    def filter_by_full_name(filters):
        filter_set = {flt for flt in filters}
        filtered_items = []
        deselected_items = []
        for item in items:
            if item.nodeid in filter_set:
                filtered_items.append(item)
            else:
                deselected_items.append(item)

        config.hook.pytest_deselected(items=deselected_items)
        items[:] = filtered_items

    # XXX - check to be removed when tests for peerdirs don't run
    for item in items:
        if not item.nodeid:
            item._nodeid = os.path.basename(item.location[0])
    if os.path.exists(config.option.test_list_path):
        with open(config.option.test_list_path, 'r') as afile:
            chunks = json.load(afile)
            filters = chunks[config.option.modulo_index]
            filter_by_full_name(filters)
    else:
        if config.option.test_filter:
            filter_items(config.option.test_filter)
        partition_mode = config.option.partition_mode
        modulo = config.option.modulo
        if modulo > 1:
            items[:] = sorted(items, key=lambda item: item.nodeid)
            modulo_index = config.option.modulo_index
            split_by_tests = config.option.split_by_tests
            items_by_classes = {}
            res = []
            for item in items:
                if item.nodeid.count("::") == 2 and not split_by_tests:
                    class_name = item.nodeid.rsplit("::", 1)[0]
                    if class_name not in items_by_classes:
                        items_by_classes[class_name] = []
                        res.append(items_by_classes[class_name])
                    items_by_classes[class_name].append(item)
                else:
                    res.append([item])
            chunk_items = test_splitter.get_splitted_tests(res, modulo, modulo_index, partition_mode, is_sorted=True)
            items[:] = []
            for item in chunk_items:
                items.extend(item)
            yatest_logger.info("Modulo %s tests are: %s", modulo_index, chunk_items)

    if config.option.mode == yatest_lib.ya.RunMode.Run:
        for item in items:
            test_item = NotLaunchedTestItem(item.nodeid, item.location[0], config.option.test_suffix)
            config.ya_trace_reporter.on_start_test_class(test_item)
            config.ya_trace_reporter.on_finish_test_case(test_item)
            config.ya_trace_reporter.on_finish_test_class(test_item)
    elif config.option.mode == yatest_lib.ya.RunMode.List:
        tests = []
        for item in items:
            item = CustomTestItem(item.nodeid, item.location[0], pytest_config.option.test_suffix, item.keywords)
            record = {
                "class": item.class_name,
                "test": item.test_name,
                "tags": _get_item_tags(item),
            }
            tests.append(record)
        if config.option.test_list_file:
            with open(config.option.test_list_file, 'w') as afile:
                json.dump(tests, afile)
        # TODO prettyboy remove after test_tool release - currently it's required for backward compatibility
        sys.stderr.write(json.dumps(tests))


def pytest_collectreport(report):
    if not report.passed:
        if hasattr(pytest_config, 'ya_trace_reporter'):
            test_item = TestItem(report, None, None, pytest_config.option.test_suffix)
            pytest_config.ya_trace_reporter.on_error(test_item)
        else:
            sys.stderr.write(yatest_lib.tools.to_utf8(report.longrepr))


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    testfunction = pyfuncitem.obj
    iscoroutinefunction = getattr(inspect, "iscoroutinefunction", None)
    if iscoroutinefunction is not None and iscoroutinefunction(testfunction):
        msg = "Coroutine functions are not natively supported and have been skipped.\n"
        msg += "You need to install a suitable plugin for your async framework, for example:\n"
        msg += "  - pytest-asyncio\n"
        msg += "  - pytest-trio\n"
        msg += "  - pytest-tornasync"
        warnings.warn(PytestUnhandledCoroutineWarning(msg.format(pyfuncitem.nodeid)))
        _pytest.outcomes.skip(msg="coroutine function and no async plugin installed (see warnings)")
    funcargs = pyfuncitem.funcargs
    testargs = {arg: funcargs[arg] for arg in pyfuncitem._fixtureinfo.argnames}
    pyfuncitem.retval = testfunction(**testargs)
    return True


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    def logreport(report, result, call, markers):
        test_item = TestItem(report, item.location[0], result, pytest_config.option.test_suffix, markers=markers)
        if not pytest_config.suite_metrics and context.Ctx.get("YA_PYTEST_START_TIMESTAMP"):
            pytest_config.suite_metrics["pytest_startup_duration"] = call.start - context.Ctx["YA_PYTEST_START_TIMESTAMP"]
            pytest_config.ya_trace_reporter.dump_suite_metrics()

        pytest_config.ya_trace_reporter.on_log_report(test_item)

        if report.outcome == "failed":
            yatest_logger.error(report.longrepr)

        if report.when == "call":
            _collect_test_rusage(item)
            pytest_config.ya_trace_reporter.on_finish_test_case(test_item)
        elif report.when == "setup":
            pytest_config.ya_trace_reporter.on_start_test_class(test_item)
            if report.outcome != "passed":
                pytest_config.ya_trace_reporter.on_start_test_case(test_item)
                pytest_config.ya_trace_reporter.on_finish_test_case(test_item)
            else:
                pytest_config.ya_trace_reporter.on_start_test_case(test_item)
        elif report.when == "teardown":
            if report.outcome == "failed":
                pytest_config.ya_trace_reporter.on_start_test_case(test_item)
                pytest_config.ya_trace_reporter.on_finish_test_case(test_item)
            else:
                pytest_config.ya_trace_reporter.on_finish_test_case(test_item, duration_only=True)
            pytest_config.ya_trace_reporter.on_finish_test_class(test_item)

    outcome = yield
    rep = outcome.get_result()
    result = None
    if hasattr(item, 'retval') and item.retval is not None:
        result = item.retval
        if not pytest_config.from_ya_test:
            ti = TestItem(rep, item.location[0], result, pytest_config.option.test_suffix)
            tr = pytest_config.pluginmanager.getplugin('terminalreporter')
            tr.write_line("{} - Validating canonical data is not supported when running standalone binary".format(ti), yellow=True, bold=True)
    logreport(rep, result, call, item.own_markers)


def pytest_make_parametrize_id(config, val, argname):
    # Avoid <, > symbols in canondata file names
    if inspect.isfunction(val) and val.__name__ == "<lambda>":
        return str(argname)
    return None


def get_formatted_error(report):
    if isinstance(report.longrepr, tuple):
        text = ""
        for entry in report.longrepr:
            text += colorize(entry)
    else:
        text = colorize(report.longrepr)
    text = yatest_lib.tools.to_utf8(text)
    return text


def colorize(longrepr):
    # use default pytest colorization
    if pytest_config.option.tbstyle != "short":
        io = py.io.TextIO()
        if six.PY2:
            writer = py.io.TerminalWriter(file=io)
        else:
            writer = _pytest._io.TerminalWriter(file=io)
        # enable colorization
        writer.hasmarkup = True

        if hasattr(longrepr, 'reprtraceback') and hasattr(longrepr.reprtraceback, 'toterminal'):
            longrepr.reprtraceback.toterminal(writer)
            return io.getvalue().strip()
        return yatest_lib.tools.to_utf8(longrepr)

    # Use arcadia style colorization
    text = yatest_lib.tools.to_utf8(longrepr)
    return tools.colorize_pytest_error(text)


class TestItem(object):

    def __init__(self, report, location, result, test_suffix, markers=None):
        self._result = result
        self.nodeid = report.nodeid
        self._class_name, self._test_name = tools.split_node_id(self.nodeid, test_suffix)
        self._error = ""
        self._status = None
        self._location = location
        self._duration = hasattr(report, 'duration') and report.duration or 0
        self._keywords = getattr(report, "keywords", {})
        self._xfaildiff = any(m.name == 'xfaildiff' for m in (markers or []))
        self._process_report(report)

    def _process_report(self, report):
        if report.longrepr:
            self.set_error(report)
            if hasattr(report, 'when') and report.when != "call":
                self.set_error(report.when + " failed:\n" + self._error)

        report_teststatus = _pytest.skipping.pytest_report_teststatus(report)
        if report_teststatus is not None:
            report_teststatus = report_teststatus[0]

        if report_teststatus == 'xfailed':
            self._status = 'xfail'
            self.set_error(report.wasxfail or 'test was marked as xfail', 'imp')
        elif report_teststatus == 'xpassed':
            self._status = 'xpass'
            self.set_error("Test unexpectedly passed")
        elif report.skipped:
            self._status = 'skipped'
            self.set_error(yatest_lib.tools.to_utf8(report.longrepr[-1]))
        elif report.passed:
            if self._xfaildiff:
                self._status = 'xfaildiff'
            else:
                self._status = 'good'
        else:
            self._status = 'fail'

    @property
    def status(self):
        return self._status

    def set_status(self, status):
        self._status = status

    @property
    def test_name(self):
        return tools.normalize_name(self._test_name)

    @property
    def class_name(self):
        return tools.normalize_name(self._class_name)

    @property
    def error(self):
        return self._error

    @property
    def location(self):
        return self._location

    def set_error(self, entry, marker='bad'):
        assert entry != ""
        if isinstance(entry, _pytest.reports.BaseReport):
            self._error = pytest_config.pluginmanager.hook.pytest_ya_summarize_error(
                report=entry
            )
        else:
            self._error = "[[{}]]{}".format(yatest_lib.tools.to_str(marker), yatest_lib.tools.to_str(entry))

    @property
    def duration(self):
        return self._duration

    @property
    def result(self):
        if 'not_canonize' in self._keywords:
            return None
        return self._result

    @property
    def keywords(self):
        return self._keywords

    def __str__(self):
        return "{}::{}".format(self.class_name, self.test_name)


class CustomTestItem(TestItem):

    def __init__(self, nodeid, location, test_suffix, keywords=None):
        self._result = None
        self.nodeid = nodeid
        self._location = location
        self._class_name, self._test_name = tools.split_node_id(nodeid, test_suffix)
        self._duration = 0
        self._error = ""
        self._keywords = keywords if keywords is not None else {}


class NotLaunchedTestItem(CustomTestItem):

    def __init__(self, nodeid, location, test_suffix):
        super(NotLaunchedTestItem, self).__init__(nodeid, location, test_suffix)
        self._status = "not_launched"


class CrashedTestItem(CustomTestItem):

    def __init__(self, nodeid, location, test_suffix):
        super(CrashedTestItem, self).__init__(nodeid, location, test_suffix)
        self._status = "crashed"


class DeselectedTestItem(CustomTestItem):

    def __init__(self, nodeid, location, test_suffix):
        super(DeselectedTestItem, self).__init__(nodeid, location, test_suffix)
        self._status = "deselected"


class TraceReportGenerator(object):

    def __init__(self, out_file_path):
        self._filename = out_file_path
        self._file = open(out_file_path, 'w')
        self._wreckage_filename = out_file_path + '.wreckage'
        self._test_messages = {}
        self._test_duration = {}
        # Some machinery to avoid data corruption due sloppy fork()
        self._current_test = (None, None)
        self._pid = os.getpid()
        self._check_intricate_respawn()

    def _check_intricate_respawn(self):
        pid_file = self._filename + '.pid'
        try:
            # python2 doesn't support open(f, 'x')
            afile = os.fdopen(os.open(pid_file, os.O_WRONLY | os.O_EXCL | os.O_CREAT), 'w')
            afile.write(str(self._pid))
            afile.close()
            return
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        # Looks like the test binary was respawned
        if from_ya_test():
            try:
                with open(pid_file) as afile:
                    prev_pid = afile.read()
            except Exception as e:
                prev_pid = '(failed to obtain previous pid: {})'.format(e)

            parts = [
                "Aborting test run: test machinery found that the test binary {} has already been run before.".format(sys.executable),
                "Looks like test has incorrect respawn/relaunch logic within test binary.",
                "Test should not try to restart itself - this is a poorly designed test case that leads to errors and could corrupt internal test machinery files.",
                "Debug info: previous pid:{} current:{}".format(prev_pid, self._pid),
            ]
            msg = '\n'.join(parts)
            yatest_logger.error(msg)

            if filelock:
                lock = filelock.FileLock(self._wreckage_filename + '.lock')
                lock.acquire()

            with open(self._wreckage_filename, 'a') as afile:
                self._file = afile

                self._dump_trace('chunk_event', {"errors": [('fail', '[[bad]]' + msg)]})

            raise Exception(msg)
        else:
            # Test binary is launched without `ya make -t`'s testing machinery - don't rely on clean environment
            pass

    def on_start_test_class(self, test_item):
        pytest_config.ya.set_test_item_node_id(test_item.nodeid)
        class_name = test_item.class_name.decode('utf-8') if sys.version_info[0] < 3 else test_item.class_name
        self._current_test = (class_name, None)

    def on_finish_test_class(self, test_item):
        pytest_config.ya.set_test_item_node_id(test_item.nodeid)

    def on_start_test_case(self, test_item):
        class_name = yatest_lib.tools.to_utf8(test_item.class_name)
        subtest_name = yatest_lib.tools.to_utf8(test_item.test_name)
        message = {
            'class': class_name,
            'subtest': subtest_name,
        }
        # Enable when CI is ready, see YA-465
        if False and test_item.location:  # noqa PLR1727
            message['path'] = test_item.location
        if test_item.nodeid in pytest_config.test_logs:
            message['logs'] = pytest_config.test_logs[test_item.nodeid]
        pytest_config.ya.set_test_item_node_id(test_item.nodeid)
        self._current_test = (class_name, subtest_name)
        self.trace('subtest-started', message)

    def on_finish_test_case(self, test_item, duration_only=False):
        if test_item.result is not None:
            try:
                result = canon.serialize(test_item.result)
            except Exception as e:
                yatest_logger.exception("Error while serializing test results")
                test_item.set_error("Invalid test result: {}".format(e))
                test_item.set_status("fail")
                result = None
        else:
            result = None

        if duration_only and test_item.nodeid in self._test_messages:  # add teardown time
            message = self._test_messages[test_item.nodeid]
        else:
            comment = self._test_messages[test_item.nodeid]['comment'] if test_item.nodeid in self._test_messages else ''
            comment += self._get_comment(test_item)
            message = {
                'class': yatest_lib.tools.to_utf8(test_item.class_name),
                'subtest': yatest_lib.tools.to_utf8(test_item.test_name),
                'status': test_item.status,
                'comment': comment,
                'result': result,
                'metrics': test_metrics.get(test_item.nodeid),
                'is_diff_test': 'diff_test' in test_item.keywords,
                'tags': _get_item_tags(test_item),
            }
            # Enable when CI is ready, see YA-465
            if False and test_item.location:  # noqa PLR1727
                message['path'] = test_item.location
            if test_item.nodeid in pytest_config.test_logs:
                message['logs'] = pytest_config.test_logs[test_item.nodeid]

        message['time'] = self._test_duration.get(test_item.nodeid, test_item.duration)

        self.trace('subtest-finished', message)
        self._test_messages[test_item.nodeid] = message

    def dump_suite_metrics(self):
        message = {"metrics": pytest_config.suite_metrics}
        self.trace("chunk_event", message)

    def on_error(self, test_item):
        self.trace('chunk_event', {"errors": [(test_item.status, self._get_comment(test_item))]})

    def on_log_report(self, test_item):
        if test_item.nodeid in self._test_duration:
            self._test_duration[test_item.nodeid] += test_item._duration
        else:
            self._test_duration[test_item.nodeid] = test_item._duration

    @staticmethod
    def _get_comment(test_item, limit=8*1024):
        msg = yatest_lib.tools.to_utf8(test_item.error)
        if not msg:
            return ""

        if len(msg) > limit:
            msg = msg[:limit - 3] + "..."

        return msg + "[[rst]]"

    def _dump_trace(self, name, value):
        event = {
            'timestamp': time.time(),
            'value': value,
            'name': name
        }

        data = yatest_lib.tools.to_str(json.dumps(event, ensure_ascii=False))
        self._file.write(data + '\n')
        self._file.flush()

    def _check_sloppy_fork(self, name, value):
        if self._pid == os.getpid():
            return

        yatest_logger.error("Skip tracing to avoid data corruption, name = %s, value = %s", name, value)

        try:
            # Lock wreckage tracefile to avoid race if multiple tests use fork sloppily
            if filelock:
                lock = filelock.FileLock(self._wreckage_filename + '.lock')
                lock.acquire()

            with open(self._wreckage_filename, 'a') as afile:
                self._file = afile

                parts = [
                    "It looks like you have leaked process - it could corrupt internal test machinery files.",
                    "Usually it happens when you casually use fork() without os._exit(),",
                    "which results in two pytest processes running at the same time.",
                    "Pid of the original pytest's process is {}, however current process has {} pid.".format(self._pid, os.getpid()),
                ]
                if self._current_test[1]:
                    parts.append("Most likely the problem is in '{}' test.".format(self._current_test))
                else:
                    parts.append("Most likely new process was created before any test was launched (during the import stage?).")

                if value.get('comment'):
                    comment = value.get('comment', '').strip()
                    # multiline comment
                    newline_required = '\n' if '\n' in comment else ''
                    parts.append("Debug info: name = '{}' comment:{}{}".format(name, newline_required, comment))
                else:
                    val_str = json.dumps(value, ensure_ascii=False).encode('utf-8')
                    parts.append("Debug info: name = '{}' value = '{}'".format(name, base64.b64encode(val_str)))

                msg = "[[bad]]{}".format('\n'.join(parts))
                class_name, subtest_name = self._current_test
                if subtest_name:
                    data = {
                        'class': class_name,
                        'subtest': subtest_name,
                        'status': 'fail',
                        'comment': msg,
                    }
                    # overwrite original status
                    self._dump_trace('subtest-finished', data)
                else:
                    self._dump_trace('chunk_event', {"errors": [('fail', msg)]})
        except Exception as e:
            yatest_logger.exception(e)
        finally:
            os._exit(38)

    def trace(self, name, value):
        self._check_sloppy_fork(name, value)
        self._dump_trace(name, value)


class DryTraceReportGenerator(TraceReportGenerator):
    """
    Generator does not write any information.
    """

    def __init__(self, *args, **kwargs):
        self._test_messages = {}
        self._test_duration = {}

    def trace(self, name, value):
        pass
