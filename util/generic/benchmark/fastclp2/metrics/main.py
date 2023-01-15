import yatest.common as yc


def test_export_metrics(metrics):
    metrics.set_benchmark(yc.execute_benchmark(
        'util/generic/benchmark/fastclp2/fastclp2',
        threads=8))
