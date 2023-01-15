PY2TEST()



PEERDIR(
    contrib/python/traitlets
)

SRCDIR(contrib/python/traitlets/py2)

TEST_SRCS(
    traitlets/tests/test_traitlets.py
    traitlets/tests/test_traitlets_enum.py
)

NO_LINT()

END()
