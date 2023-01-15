PY2TEST()



PEERDIR(
    contrib/python/traitlets
)

SRCDIR(contrib/python/traitlets/py2/traitlets/tests)

TEST_SRCS(
    test_traitlets.py
    test_traitlets_enum.py
)

NO_LINT()

END()
