

PY3_PROGRAM()

PEERDIR(
    contrib/python/numpy
)

PY_SRCS(
    NAMESPACE numpy.f2py
    __init__.py
    __init__.pyi
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
    f90mod_rules.py
    func2subr.py
    rules.py
    symbolic.py
    use_rules.py
)

NO_LINT()

END()
