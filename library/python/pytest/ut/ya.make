PY23_TEST()

TEST_SRCS(
    test_tools.py
)

PEERDIR(
    contrib/python/pytest-mock
    library/python/pytest
)

STYLE_PYTHON()

END()
