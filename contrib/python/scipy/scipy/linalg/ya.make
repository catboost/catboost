PY23_LIBRARY()



NO_COMPILER_WARNINGS()

ADDINCLSELF()

ADDINCL(
    contrib/python/scipy
)

PEERDIR(
    contrib/python/numpy
    contrib/python/numpy/numpy/f2py/src

    contrib/python/scipy/scipy/_build_utils/src
    contrib/python/scipy/scipy/_lib
    contrib/python/scipy/scipy/linalg/src

    contrib/libs/clapack
    contrib/libs/cblas
)

NO_LINT()

PY_SRCS(
    NAMESPACE scipy.linalg

    __init__.py
    basic.py
    blas.py
    calc_lwork.py
    decomp_cholesky.py
    decomp_lu.py
    _decomp_polar.py
    decomp.py
    decomp_qr.py
    _decomp_qz.py
    decomp_schur.py
    decomp_svd.py
    _expm_frechet.py
    flinalg.py
    _interpolative_backend.py
    interpolative.py
    lapack.py
    linalg_version.py
    _matfuncs_inv_ssq.py
    matfuncs.py
    _matfuncs_sqrtm.py
    misc.py
    _procrustes.py
    _solvers.py
    special_matrices.py

    CYTHON_C
    _solve_toeplitz.pyx
    cython_lapack.pyx
    cython_blas.pyx
    _decomp_update.pyx
)

SRCS(
    _fblasmodule.c
    _flapackmodule.c
    _calc_lworkmodule.c
    _flinalgmodule.c
    _interpolativemodule.c

    _blas_subroutine_wrappers.f
    _lapack_subroutine_wrappers.f

    _fblas-f2pywrappers.f
    _flapack-f2pywrappers.f
)

PY_REGISTER(scipy.linalg._fblas)
PY_REGISTER(scipy.linalg._flapack)
PY_REGISTER(scipy.linalg._calc_lwork)
PY_REGISTER(scipy.linalg._flinalg)
PY_REGISTER(scipy.linalg._interpolative)

END()
