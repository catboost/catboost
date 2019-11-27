PY23_LIBRARY()



NO_COMPILER_WARNINGS()

IF (OS_WINDOWS)
    CFLAGS(-D_USE_MATH_DEFINES)
ENDIF()

ADDINCL(
    contrib/python/scipy
    contrib/python/scipy/scipy/special
    contrib/python/scipy/scipy/special/c_misc
)

PEERDIR(
    contrib/python/numpy
    contrib/python/numpy/numpy/f2py/src

    contrib/python/scipy/scipy/_lib
    contrib/python/scipy/scipy/linalg
    contrib/python/scipy/scipy/special/_precompute
    contrib/python/scipy/scipy/special/amos
    contrib/python/scipy/scipy/special/cephes
    contrib/python/scipy/scipy/special/c_misc
    contrib/python/scipy/scipy/special/cdflib
    contrib/python/scipy/scipy/special/mach
    contrib/python/scipy/scipy/special/specfun
)

NO_LINT()

PY_SRCS(
    NAMESPACE scipy.special

    __init__.py
    lambertw.py
    orthogonal.py
    _ellip_harm.py
    basic.py
    spfun_stats.py
#    _testutils.py
#    _mptestutils.py
    add_newdocs.py
    _spherical_bessel.py

    CYTHON_C
    _comb.pyx
    _ufuncs.pyx
    _ellip_harm_2.pyx

    CYTHON_CPP
    _ufuncs_cxx.pyx
)

SRCS(
    amos_wrappers.c
    cdf_wrappers.c
    sf_error.c
    specfun_wrappers.c
    _faddeeva.cxx
    Faddeeva.cc
)

SRCS(
    _logit.c
    specfunmodule.c
)

PY_REGISTER(scipy.special.specfun)

END()
