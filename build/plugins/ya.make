

PY2_LIBRARY()

PY_SRCS(
    code_generator.py
    ssqls.py

    _common.py
    _requirements.py
    _test_const.py
)

PEERDIR(build/plugins/lib)

END()

RECURSE(
    tests
)
