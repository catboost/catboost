import os
import sys
import logging
import json
import six

from .tools import to_str
from .external import ExternalDataInfo

TESTING_OUT_DIR_NAME = "testing_out_stuff"  # XXX import from test.const

yatest_logger = logging.getLogger("ya.test")


class RunMode(object):
    Run = "run"
    List = "list"


class TestMisconfigurationException(Exception):
    pass


class Ya(object):
    """
    Adds integration with ya, helps in finding dependencies
    """

    def __init__(
        self,
        mode=None,
        source_root=None,
        build_root=None,
        dep_roots=None,
        output_dir=None,
        test_params=None,
        context=None,
        python_path=None,
        valgrind_path=None,
        gdb_path=None,
        data_root=None,
        env_file=None,
    ):
        context_file_path = os.environ.get("YA_TEST_CONTEXT_FILE", None)
        if context_file_path:
            with open(context_file_path, 'r') as afile:
                test_context = json.load(afile)
            context_runtime = test_context["runtime"]
            context_internal = test_context.get("internal", {})
            context_build = test_context.get("build", {})
            context_resources = test_context.get("resources", {})
        else:
            context_runtime = {}
            context_internal = {}
            context_build = {}
            context_resources = {}
        self._mode = mode
        self._build_root = to_str(context_runtime.get("build_root", "")) or build_root
        self._source_root = to_str(context_runtime.get("source_root", "")) or source_root or self._detect_source_root()
        self._output_dir = to_str(context_runtime.get("output_path", "")) or output_dir or self._detect_output_root()
        if not self._output_dir:
            raise Exception("Run ya make -t before running test binary")
        if not self._source_root:
            logging.warning("Source root was not set neither determined, use --source-root to set it explicitly")
        if not self._build_root:
            if self._source_root:
                self._build_root = self._source_root
            else:
                logging.warning("Build root was not set neither determined, use --build-root to set it explicitly")

        if data_root:
            self._data_root = data_root
        elif self._source_root:
            self._data_root = os.path.abspath(os.path.join(self._source_root, "..", "arcadia_tests_data"))

        self._dep_roots = dep_roots

        self._python_path = to_str(context_runtime.get("python_bin", "")) or python_path
        self._valgrind_path = valgrind_path
        self._gdb_path = to_str(context_runtime.get("gdb_bin", "")) or gdb_path
        self._test_params = {}
        self._context = {}
        self._test_item_node_id = None

        ram_drive_path = to_str(context_runtime.get("ram_drive_path", ""))
        if ram_drive_path:
            self._test_params["ram_drive_path"] = ram_drive_path
        if test_params:
            self._test_params.update(dict(x.split('=', 1) for x in test_params))
        self._test_params.update(context_runtime.get("test_params", {}))

        self._context["project_path"] = context_runtime.get("project_path")
        self._context["modulo"] = context_runtime.get("split_count", 1)
        self._context["modulo_index"] = context_runtime.get("split_index", 0)
        self._context["work_path"] = to_str(context_runtime.get("work_path"))
        self._context["test_tool_path"] = context_runtime.get("test_tool_path")
        self._context["test_output_ram_drive_path"] = to_str(context_runtime.get("test_output_ram_drive_path"))
        self._context["retry_index"] = context_runtime.get("retry_index")

        self._context["ya_global_resources"] = context_resources.get("global")

        self._context["sanitize"] = context_build.get("sanitizer")
        self._context["ya_trace_path"] = context_internal.get("trace_file")

        self._env_file = context_internal.get("env_file") or env_file

        if context:
            for k, v in context.items():
                if k not in self._context or v is not None:
                    self._context[k] = v

        if self._env_file and os.path.exists(self._env_file):
            yatest_logger.debug("Reading variables from env_file at %s", self._env_file)
            var_list = []
            with open(self._env_file) as file:
                for ljson in file.readlines():
                    variable = json.loads(ljson)
                    for key, value in six.iteritems(variable):
                        os.environ[key] = str(value)
                        var_list.append(key)
            yatest_logger.debug("Variables loaded: %s", var_list)

    @property
    def source_root(self):
        return self._source_root

    @property
    def data_root(self):
        return self._data_root

    @property
    def build_root(self):
        return self._build_root

    @property
    def dep_roots(self):
        return self._dep_roots

    @property
    def output_dir(self):
        return self._output_dir

    @property
    def python_path(self):
        return self._python_path or sys.executable

    @property
    def valgrind_path(self):
        if not self._valgrind_path:
            raise ValueError("path to valgrind was not pass correctly, use --valgrind-path to fix it")
        return self._valgrind_path

    @property
    def gdb_path(self):
        return self._gdb_path

    @property
    def env_file(self):
        return self._env_file

    def get_binary(self, *path):
        assert self._build_root, "Build root was not set neither determined, use --build-root to set it explicitly"
        path = list(path)
        if os.name == "nt":
            if not path[-1].endswith(".exe"):
                path[-1] += ".exe"

        target_dirs = [self.build_root]
        # Search for binaries within PATH dirs to be able to get path to the binaries specified by basename for exectests
        if 'PATH' in os.environ:
            target_dirs += os.environ['PATH'].split(':')

        for target_dir in target_dirs:
            binary_path = os.path.join(target_dir, *path)
            if os.path.exists(binary_path):
                yatest_logger.debug("Binary was found by %s", binary_path)
                return binary_path

        error_message = "Cannot find binary '{binary}': make sure it was added in the DEPENDS section".format(binary=path)
        yatest_logger.debug(error_message)
        if self._mode == RunMode.Run:
            raise TestMisconfigurationException(error_message)

    def _build_root_rel(self, path):
        real_build_root = os.path.realpath(self.build_root)
        real_path = os.path.abspath(path)
        if path.startswith(real_build_root):
            return os.path.relpath(real_path, real_build_root)
        return path

    def file(self, path, diff_tool=None, local=False, diff_file_name=None, diff_tool_timeout=None):
        if diff_tool:
            if isinstance(diff_tool, tuple):
                diff_tool = list(diff_tool)
            # Normalize path to diff_tool - abs path in run_test node won't be accessible in canonize node
            if isinstance(diff_tool, list):
                diff_tool[0] = self._build_root_rel(diff_tool[0])
            else:
                diff_tool = self._build_root_rel(diff_tool)

        return ExternalDataInfo.serialize_file(path, diff_tool=diff_tool, local=local, diff_file_name=diff_file_name, diff_tool_timeout=diff_tool_timeout)

    def get_param(self, key, default=None):
        return self._test_params.get(key, default)

    def get_param_dict_copy(self):
        return dict(self._test_params)

    def get_context(self, key):
        return self._context.get(key)

    def _detect_source_root(self):
        root = None
        try:
            import library.python.find_root
            # try to determine source root from cwd
            cwd = os.getcwd()
            root = library.python.find_root.detect_root(cwd)

            if not root:
                # try to determine root pretending we are in the test work dir made from --keep-temps run
                env_subdir = os.path.join("environment", "arcadia")
                root = library.python.find_root.detect_root(cwd, detector=lambda p: os.path.exists(os.path.join(p, env_subdir)))
        except ImportError:
            logging.warning("Unable to import library.python.find_root")

        return root

    def _detect_output_root(self):

        # if run from kept test working dir
        if os.path.exists(TESTING_OUT_DIR_NAME):
            return TESTING_OUT_DIR_NAME

        # if run from source dir
        if sys.version_info.major == 3:
            test_results_dir = "py3test"
        else:
            test_results_dir = "pytest"

        test_results_output_path = os.path.join("test-results", test_results_dir, TESTING_OUT_DIR_NAME)
        if os.path.exists(test_results_output_path):
            return test_results_output_path

        if os.path.exists(os.path.dirname(test_results_output_path)):
            os.mkdir(test_results_output_path)
            return test_results_output_path

        return None

    def set_test_item_node_id(self, node_id):
        self._test_item_node_id = node_id

    def get_test_item_node_id(self):
        assert self._test_item_node_id
        return self._test_item_node_id

    def set_metric_value(self, name, val):
        from library.python.pytest.plugins.metrics import test_metrics
        node_id = self.get_test_item_node_id()
        if node_id not in test_metrics.metrics:
            test_metrics[node_id] = {}

        test_metrics[node_id][name] = val

    def get_metric_value(self, name, default=None):
        from library.python.pytest.plugins.metrics import test_metrics
        res = test_metrics.get(self.get_test_item_node_id(), {}).get(name)
        if res is None:
            return default
        return res
