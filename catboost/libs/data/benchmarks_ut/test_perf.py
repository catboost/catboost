import yatest


def test(metrics):
    metrics.set_benchmark(yatest.common.execute_benchmark("catboost/libs/data/benchmarks/benchmarks"))
