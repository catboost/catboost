PYTEST()



PEERDIR(
    build/plugins
)

TEST_SRCS(
    test_code_generator.py
    test_common.py
    test_requirements.py
    test_yasm.py
    test_swig.py
    test_pyx.py
    test_xs.py
    test_fortran.py
    test_flatc.py
)

END()
