import yatest.common as yc


def test_export_metrics(metrics):
    metrics.set_benchmark(yc.execute_benchmark(
        'util/memory/benchmark/pool/pool',
        threads=8))
