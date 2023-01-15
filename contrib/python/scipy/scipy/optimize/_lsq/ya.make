PY23_LIBRARY()



ADDINCL(
    contrib/python/scipy
)

NO_LINT()

PY_SRCS(
    NAMESPACE scipy.optimize._lsq

    bvls.py
    common.py
    dogbox.py
    __init__.py
    least_squares.py
    lsq_linear.py
    trf_linear.py
    trf.py

    givens_elimination.pyx
)

END()
