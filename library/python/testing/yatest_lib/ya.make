

PY23_LIBRARY()

PY_SRCS(
    NAMESPACE
    yatest_lib
    external.py
    tools.py
)

PEERDIR(
    contrib/python/six
)

NO_LINT()

END()
