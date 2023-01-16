PY3TEST()



TEST_SRCS(
    test_perf.py
)

DEPENDS(
    catboost/libs/data/benchmarks
)

END()
