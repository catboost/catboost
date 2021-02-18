

PY3_PROGRAM()

PEERDIR(
    contrib/python/numpy
)

PY_SRCS(
    NAMESPACE numpy.f2py
    __init__.py
    __main__.py
    __version__.py
    auxfuncs.py
    capi_maps.py
    cb_rules.py
    cfuncs.py
    common_rules.py
    crackfortran.py
    diagnose.py
    f2py2e.py
    f2py_testing.py
    f90mod_rules.py
    func2subr.py
    rules.py
    use_rules.py
)

NO_LINT()

END()
