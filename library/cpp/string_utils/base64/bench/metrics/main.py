import yatest.common as yc


def test_export_metrics(metrics):
    metrics.set_benchmark(yc.execute_benchmark('library/cpp/string_utils/base64/bench/bench'))
