

PY23_LIBRARY()

PY_SRCS(
    NAMESPACE
    yatest_lib
    external.py
    test_splitter.py
    tools.py
    ya.py
)

PEERDIR(
    contrib/python/six
)

IF(PYTHON2)
    PEERDIR(
        contrib/deprecated/python/enum34
    )
ENDIF()

END()

RECURSE_FOR_TESTS(tests)
