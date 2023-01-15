import yatest.common as yc


def test_export_metrics(metrics):
    metrics.set_benchmark(yc.execute_benchmark(
        'util/string/benchmark/subst_global/subst_global',
        threads=8))
