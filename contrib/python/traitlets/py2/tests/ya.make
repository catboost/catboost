PY2TEST()



PEERDIR(
    contrib/python/traitlets
)

ENV(
    YA_PYTEST_DISABLE_DOCTEST=yes
)

SRCDIR(contrib/python/traitlets/py2/traitlets)

TEST_SRCS(
    tests/__init__.py
    tests/_warnings.py
    tests/test_traitlets.py
    tests/test_traitlets_enum.py
    tests/utils.py
)

NO_LINT()

END()
