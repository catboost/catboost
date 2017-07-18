import yatest.common as yc


def test_export_metrics(metrics):
    metrics.set_benchmark(yc.execute_benchmark('util/charset/benchmark/utf8_to_wide/utf8_to_wide'))
