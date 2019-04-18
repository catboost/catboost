

PYTEST()

SIZE(MEDIUM)
REQUIREMENTS(network:full)

FORK_SUBTESTS()

NO_CHECK_IMPORTS(widget.ipythonwidget)

PEERDIR(
    contrib/python/pandas
    contrib/python/numpy
    contrib/python/six
    library/python/pytest
    catboost/python-package/lib
    catboost/pytest/lib
)

TEST_SRCS(
    conftest.py
    test.py
    test_whl.py
)

DATA(
    arcadia/catboost/pytest/data
    arcadia/catboost/python-package
)

DEPENDS(
    catboost/tools/model_comparator
    catboost/python-package/catboost/no_cuda
    catboost/python-package/ut/medium/python_binary
)

END()
