PY23_LIBRARY()



NO_COMPILER_WARNINGS()

NO_LINT()

PY_SRCS(
    NAMESPACE scipy.optimize

    __init__.py
    _basinhopping.py
    cobyla.py
    _differentialevolution.py
    _hungarian.py
    lbfgsb.py
    linesearch.py
    _linprog.py
    _minimize.py
    minpack.py
    nnls.py
    nonlin.py
    _numdiff.py
    optimize.py
    _root.py
    slsqp.py
    _spectral.py
    tnc.py
    _trustregion_dogleg.py
    _trustregion_ncg.py
    _trustregion.py
    _tstutils.py
    zeros.py

    CYTHON_C
    _group_columns.pyx
)

SRCS(
    _minpackmodule.c
    zeros.c
)

PY_REGISTER(scipy.optimize._minpack)
PY_REGISTER(scipy.optimize._zeros)

PEERDIR(
    contrib/python/scipy/scipy/optimize/tnc
    contrib/python/scipy/scipy/optimize/nnls
    contrib/python/scipy/scipy/optimize/cobyla
    contrib/python/scipy/scipy/optimize/slsqp
    contrib/python/scipy/scipy/optimize/lbfgsb
    contrib/python/scipy/scipy/optimize/_lsq
    contrib/python/scipy/scipy/optimize/minpack
    contrib/python/scipy/scipy/optimize/minpack2
    contrib/python/scipy/scipy/optimize/Zeros
)

END()
