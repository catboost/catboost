import yatest.common as yc


def test_export_metrics(metrics):
    metrics.set_benchmark(yc.execute_benchmark(
        'util/generic/benchmark/vector_count_ctor/vector_count_ctor',
        threads=8))
