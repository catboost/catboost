PY2TEST()



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
    test_basic.py
    test_blas.py
    test_cython_blas.py
    test_cython_lapack.py
    test_decomp.py
    test_decomp_cholesky.py
    test_decomp_ldl.py
    test_decomp_polar.py
    test_decomp_update.py
    test_fblas.py
    test_interpolative.py
    test_procrustes.py
    test_sketches.py
    test_solvers.py
    test_special_matrices.py
)

END()
