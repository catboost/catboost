import yatest.common as yc


def test_export_metrics(metrics):
    metrics.set_benchmark(yc.execute_benchmark(
        'util/generic/benchmark/rotate_bits/rotate_bits',
        threads=8))
