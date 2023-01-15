import yatest.common as yc


def test_export_metrics(metrics):
    metrics.set_benchmark(yc.execute_benchmark(
        'util/random/benchmark/prng/prng',
        threads=8))
