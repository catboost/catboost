import yatest.common as yc


def test_export_metrics(metrics):
    metrics.set_benchmark(yc.execute_benchmark(
        'util/system/benchmark/cpu_id/cpu_id',
        threads=8))
