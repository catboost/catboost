

PY_PROGRAM()

PEERDIR(
    contrib/python/numpy
)


PY_SRCS(
    NAMESPACE numpy.f2py

    __main__.py

    __init__.py
    func2subr.py
    rules.py
    auxfuncs.py
    __version__.py
    f90mod_rules.py
    use_rules.py
    cfuncs.py
    cb_rules.py
    info.py
    diagnose.py
    capi_maps.py
    crackfortran.py
    f2py_testing.py
    common_rules.py
    f2py2e.py
)

NO_LINT()

END()
