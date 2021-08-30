

PY2_LIBRARY()

PY_SRCS(
    code_generator.py
    ssqls.py
    swig.py

    _common.py
    _custom_command.py
    _import_wrapper.py
    _requirements.py
    _test_const.py
)

PEERDIR(build/plugins/lib)

END()

RECURSE(
    tests
)
