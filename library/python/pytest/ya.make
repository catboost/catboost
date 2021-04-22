PY23_LIBRARY()



PY_SRCS(
    __init__.py
    main.py
    rewrite.py
    yatest_tools.py
    context.py
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

RESOURCE_FILES(
    PREFIX library/python/pytest/
    pytest.yatest.ini
)

END()
