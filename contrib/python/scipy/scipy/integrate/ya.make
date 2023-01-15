PY23_LIBRARY()



NO_COMPILER_WARNINGS()

PEERDIR(
    contrib/python/numpy
    contrib/python/numpy/numpy/f2py/src

    contrib/python/scipy/scipy/special
    contrib/python/scipy/scipy/sparse
    contrib/python/scipy/scipy/optimize

    contrib/python/scipy/scipy/integrate/odepack
    contrib/python/scipy/scipy/integrate/quadpack
    contrib/python/scipy/scipy/integrate/dop
    contrib/python/scipy/scipy/integrate/mach
)

NO_LINT()

PY_SRCS(
    NAMESPACE scipy.integrate

    __init__.py
    quadpack.py
    quadrature.py
    _ode.py
    odepack.py
    _bvp.py
)

SRCS(
    _odepackmodule.c
    _quadpackmodule.c

    lsodamodule.c
    vodemodule.c
    _dopmodule.c
)

PY_REGISTER(scipy.integrate._odepack)
PY_REGISTER(scipy.integrate._quadpack)
PY_REGISTER(scipy.integrate.vode)
PY_REGISTER(scipy.integrate._dop)
PY_REGISTER(scipy.integrate.lsoda)

END()
