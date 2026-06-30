import os
import pytest
import six

from library.python.pytest.plugins.metrics import test_metrics

MAX_ALLOWED_LINKS_COUNT = 10


@pytest.fixture
def metrics(request):

    class Metrics(object):
        @classmethod
        def set(cls, name, value):
            assert len(name) <= 256, "Length of the metric name must less than 128"
            assert type(value) in [int, float], "Metric value must be of type int or float"
            test_name = request.node.nodeid
            if test_name not in test_metrics.metrics:
                test_metrics[test_name] = {}
            test_metrics[test_name][name] = value

        @classmethod
        def set_benchmark(cls, benchmark_values):
            # report of google has key 'benchmarks' which is a list of benchmark results
            # yandex benchmark has key 'benchmark', which is a list of benchmark results
            # use this to differentiate which kind of result it is
            if 'benchmarks' in benchmark_values:
                cls.set_gbenchmark(benchmark_values)
            else:
                cls.set_ybenchmark(benchmark_values)

        @classmethod
        def set_ybenchmark(cls, benchmark_values):
            for benchmark in benchmark_values["benchmark"]:
                name = benchmark["name"]
                for key, value in six.iteritems(benchmark):
                    if key != "name":
                        cls.set("{}_{}".format(name, key), value)

        @classmethod
        def set_gbenchmark(cls, benchmark_values):
            time_unit_multipliers = {"ns": 1, "us": 1000, "ms": 1000000}
            time_keys = {"real_time", "cpu_time"}
            ignore_keys = {"name", "run_name", "time_unit", "run_type", "repetition_index"}
            for benchmark in benchmark_values["benchmarks"]:
                name = benchmark["name"].replace('/', '_')  # ci does not work properly with '/' in metric name
                time_unit_mult = time_unit_multipliers[benchmark.get("time_unit", "ns")]
                for k, v in six.iteritems(benchmark):
                    if k in time_keys:
                        cls.set("{}_{}".format(name, k), v * time_unit_mult)
                    elif k not in ignore_keys and isinstance(v, (float, int)):
                        cls.set("{}_{}".format(name, k), v)
    return Metrics


@pytest.fixture
def links(request):

    class Links(object):
        @classmethod
        def set(cls, name, path):

            if len(request.config.test_logs[request.node.nodeid]) >= MAX_ALLOWED_LINKS_COUNT:
                raise Exception("Cannot add more than {} links to test".format(MAX_ALLOWED_LINKS_COUNT))

            reserved_names = ["log", "logsdir", "stdout", "stderr"]
            if name in reserved_names:
                raise Exception("Attachment name should not belong to the reserved list: {}".format(", ".join(reserved_names)))
            output_dir = request.config.ya.output_dir

            if not os.path.exists(path):
                raise Exception("Path to be attached does not exist: {}".format(path))

            if os.path.isabs(path) and ".." in os.path.relpath(path, output_dir):
                raise Exception("Test attachment must be inside yatest.common.output_path()")

            request.config.test_logs[request.node.nodeid][name] = path

        @classmethod
        def get(cls, name):
            if name not in request.config.test_logs[request.node.nodeid]:
                raise KeyError("Attachment with name '{}' does not exist".format(name))
            return request.config.test_logs[request.node.nodeid][name]

    return Links
