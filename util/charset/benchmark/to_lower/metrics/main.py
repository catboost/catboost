import yatest.common as yc


def test_export_metrics(metrics):
    metrics.set_benchmark(yc.execute_benchmark('util/charset/benchmark/to_lower/to_lower'))
