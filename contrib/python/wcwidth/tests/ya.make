PY23_TEST()



PEERDIR(
    contrib/python/wcwidth
)

TEST_SRCS(
    test_core.py
    test_ucslevel.py
)

NO_LINT()

END()
