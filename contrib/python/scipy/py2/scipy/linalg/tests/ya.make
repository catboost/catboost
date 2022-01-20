PROGRAM()



NO_LINT()
NO_CHECK_IMPORTS()

PEERDIR(
    contrib/python/nose/runner

    contrib/python/scipy
)

TEST_SRCS(
    test_build.py
    test_decomp_cholesky.py
    test_special_matrices.py
    test_interpolative.py
    test_basic.py
    test_matfuncs.py
    test_blas.py
    test_fblas.py
    test_decomp.py
    test_decomp_update.py
    test_solvers.py
    test_cython_blas.py
    test_procrustes.py
    test_lapack.py
    test_cython_lapack.py
    test_decomp_polar.py
    test_solve_toeplitz.py
)

END()
