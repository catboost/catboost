PYTEST()



FORK_TESTS()
FORK_SUBTESTS()

PEERDIR(
    contrib/python/pandas
    contrib/python/numpy-1.11.1
    library/python/pytest
    catboost/python-package/lib
    catboost/pytest/lib
)

TEST_SRCS(
    test.py
)

NO_CHECK_IMPORTS(widget.ipythonwidget)

SIZE(MEDIUM)

DATA(
    arcadia/catboost/pytest/data
)

DEPENDS(
    catboost/tools/model_comparator
)

END()

