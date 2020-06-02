

PY23_LIBRARY()

PY_SRCS(
    NAMESPACE
    yatest_lib
    external.py
    test_splitter.py
    tools.py
)

PEERDIR(
    contrib/python/six
)

IF(PYTHON2)
    PEERDIR(
        contrib/python/enum34
    )
ENDIF()

NO_LINT()

END()

RECURSE_FOR_TESTS(tests)
