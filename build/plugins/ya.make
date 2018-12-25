

PY_LIBRARY()

PY_SRCS(
    code_generator.py
    gobuild.py
    fortran.py
    flatc.py
    pyx.py
    swig.py
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
