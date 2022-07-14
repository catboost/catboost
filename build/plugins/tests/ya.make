PY2TEST()



PEERDIR(
    build/plugins
)

PY_SRCS(
    fake_ymake.py
)

TEST_SRCS(
    test_code_generator.py
    test_common.py
    test_requirements.py
    test_ssqls.py
)

NO_CHECK_IMPORTS(
    build.plugins.code_generator
    build.plugins.ssqls
)

END()
