PY23_LIBRARY()



PY_SRCS(
    __init__.py
    main.py
    rewrite.py
    yatest_tools.py
)

PEERDIR(
    library/python/pytest/plugins
    library/python/testing/yatest_common
    library/python/testing/yatest_lib
    contrib/python/py
    contrib/python/pytest
    contrib/python/dateutil
)

NO_LINT()

RESOURCE_FILES(
    PREFIX library/python/pytest/
    pytest.yatest.ini
)

END()
