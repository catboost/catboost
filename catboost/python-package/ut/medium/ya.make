PYTEST()



FORK_TESTS()
FORK_SUBTESTS()

PEERDIR(
    contrib/python/pandas
    contrib/python/numpy
    library/python/pytest
    catboost/python-package/lib
    catboost/pytest/lib
)

TEST_SRCS(
    test.py
    test_whl.py
)

NO_CHECK_IMPORTS(widget.ipythonwidget)

SIZE(MEDIUM)

DATA(
    arcadia/catboost/pytest/data
    arcadia/catboost/python-package
)

DEPENDS(
    catboost/tools/model_comparator
    catboost/python-package/catboost/no_cuda
    catboost/python-package/ut/medium/python_binary
)

ALLOCATOR(LF)

END()
