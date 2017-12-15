import yatest.common as yc


def test_export_metrics(metrics):
    metrics.set_benchmark(yc.execute_benchmark(
        'util/system/benchmark/create_destroy_thread/create_destroy_thread',
        threads=8))
