LIBRARY()



PY_SRCS(
    main.py
    yatest_tools.py
)

PEERDIR(
    library/python/pytest/plugins
    library/python/testing/yatest_common
    library/python/testing/yatest_lib
    contrib/python/py
    contrib/python/pytest
    contrib/python/PyYAML
    contrib/python/dateutil
    contrib/python/requests
)

IF (NOT OS_WINDOWS)
    PEERDIR(
        contrib/python/ipython
    )
ENDIF()

END()
