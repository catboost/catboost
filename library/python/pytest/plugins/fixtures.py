import os
import pytest


MAX_ALLOWED_LINKS_COUNT = 10


@pytest.fixture
def metrics(request):

    class Metrics(object):
        @classmethod
        def set(cls, name, value):
            assert len(name) <= 128, "Length of the metric name must less than 128"
            assert type(value) in [int, float], "Metric value must be of type int or float"
            test_name = request.node.nodeid
            if test_name not in request.config.test_metrics:
                request.config.test_metrics[test_name] = {}
            request.config.test_metrics[test_name][name] = value

        @classmethod
        def set_benchmark(cls, benchmark_values):
            for benchmark in benchmark_values["benchmark"]:
                name = benchmark["name"]
                for key, value in benchmark.iteritems():
                    if key != "name":
                        cls.set("{}_{}".format(name, key), value)

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
