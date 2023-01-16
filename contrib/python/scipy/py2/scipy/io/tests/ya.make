PROGRAM()



NO_LINT()
NO_CHECK_IMPORTS()

PEERDIR(
    contrib/python/nose/runner

    contrib/python/scipy
)

TEST_SRCS(
    test_fortran.py
    test_idl.py
    test_mmio.py
    test_netcdf.py
    test_wavfile.py
)

END()
