PY2TEST()

REQUIREMENTS(ram:30)



VERSION(1.2.3)

ORIGINAL_SOURCE(mirror://pypi/s/scipy/scipy-1.2.3.tar.gz)

SIZE(MEDIUM)

FORK_TESTS()

PEERDIR(
    contrib/python/scipy/py2
    contrib/python/scipy/py2/scipy/conftest
)

NO_LINT()

NO_CHECK_IMPORTS()

TEST_SRCS(
    __init__.py
    test_construct.py
    test_csc.py
    test_csr.py
    test_extract.py
    test_matrix_io.py
    test_sparsetools.py
    test_sputils.py
)

END()
