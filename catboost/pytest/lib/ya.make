

PY_LIBRARY()

PY_SRCS(
    NAMESPACE catboost_pytest_lib
    __init__.py
    common_helpers.py
)

PEERDIR(
    contrib/python/numpy
)

END()
