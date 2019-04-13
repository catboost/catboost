PYTEST()



# NO FORK

PEERDIR(
    contrib/python/pandas
    contrib/python/numpy
    library/python/pytest
    catboost/python-package/lib
    catboost/pytest/lib
)

SRCDIR(
    catboost/python-package/ut/medium
)

TEST_SRCS(
    gpu/conftest.py
    test.py
)

NO_CHECK_IMPORTS(widget.ipythonwidget)

SIZE(MEDIUM)

DATA(
    arcadia/catboost/pytest/data
    arcadia/catboost/python-package
)

DEPENDS(
    catboost/tools/limited_precision_json_diff
    catboost/tools/model_comparator
    catboost/python-package/catboost
    catboost/python-package/ut/medium/python_binary
)

ALLOCATOR(LF)

END()
