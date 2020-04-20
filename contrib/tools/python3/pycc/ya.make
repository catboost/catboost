PY3_PROGRAM()



ENABLE(PYBUILD_NO_PYC)

DISABLE(PYTHON_SQLITE3)

PEERDIR(
    library/python/runtime_py3
    library/python/runtime_py3/main
)

NO_CHECK_IMPORTS()

NO_PYTHON_INCLUDES()

NO_PYTHON_COVERAGE()

PY_SRCS(
    MAIN main.py
)

END()
