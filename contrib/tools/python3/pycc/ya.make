PY3_PROGRAM()



PEERDIR(
    library/python/runtime_py3
    library/python/runtime_py3/main
)

NO_CHECK_IMPORTS()

NO_PYTHON_INCLUDES()

ENABLE(PYBUILD_NO_PYC)

PY_SRCS(
    MAIN main.py
)

END()
