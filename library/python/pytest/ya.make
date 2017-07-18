LIBRARY()



PY_SRCS(
    TOP_LEVEL
    __main__.py
    yatest_tools.py
)

PEERDIR(
    library/python/pytest/plugins
    library/python/testing/yatest_common
    library/python/testing/yatest_lib
    contrib/python/py-1.4.30
    contrib/python/pytest
    contrib/python/PyYAML-3.11
    contrib/python/dateutil
    contrib/python/requests
)

IF (NOT OS_WINDOWS)
    PEERDIR(
        contrib/python/ipython
    )
ENDIF()

END()

RECURSE(
    plugins
    empty
)
