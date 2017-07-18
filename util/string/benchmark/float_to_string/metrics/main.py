import yatest.common as yc


def test_export_metrics(metrics):
    metrics.set_benchmark(yc.execute_benchmark(
        'util/string/benchmark/float_to_string/float_to_string',
        threads=8))
