

LIBRARY()

PY_SRCS(
    fortran.py
    flatc.py
    pyx.py
    swig.py
    td.py
    xs.py
    yasm.py

    _common.py
    _custom_command.py
    _import_wrapper.py
    _metric_resolvers.py
    _requirements.py
    _test_const.py
)

END()

RECURSE(
    tests
)
