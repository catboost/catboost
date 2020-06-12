import yatest.common as yc


def test_export_metrics(metrics):
    metrics.set_benchmark(yc.execute_benchmark(
        'library/cpp/testing/benchmark/examples/examples',
        threads=8))
