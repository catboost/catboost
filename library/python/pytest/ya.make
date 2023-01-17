PY23_LIBRARY()



PY_SRCS(
    __init__.py
    config.py
    context.py
    main.py
    rewrite.py
    yatest_tools.py
)

PEERDIR(
    contrib/python/dateutil
    contrib/python/ipdb
    contrib/python/py
    contrib/python/pytest
    library/python/pytest/plugins
    library/python/testing/yatest_common
    library/python/testing/yatest_lib
)

IF (NOT OPENSOURCE)
    PEERDIR(contrib/tools/gprof2dot)
ENDIF()

RESOURCE_FILES(
    PREFIX library/python/pytest/
    pytest.yatest.ini
)

END()

RECURSE_FOR_TESTS(
    ut
)
