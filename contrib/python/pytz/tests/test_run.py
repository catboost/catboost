import yatest.common as yc


def test_run():
    runner = yc.build_path('contrib/python/pytz/tests/runner/doctest.pytz')
    yc.execute(runner)
