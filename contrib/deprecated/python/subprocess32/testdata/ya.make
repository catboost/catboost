PY2TEST()



SIZE(MEDIUM)

NO_LINT()

SRCDIR(
    contrib/deprecated/python/subprocess32
)

TEST_SRCS(
    test_subprocess32.py
)

TEST_CWD(
    contrib/deprecated/python/subprocess32
)

END()
