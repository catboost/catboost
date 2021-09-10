

PY3_PROGRAM_BIN(pycc)

ENABLE(PYBUILD_NO_PYC)

DISABLE(PYTHON_SQLITE3)

PEERDIR(
    library/python/runtime_py3
    library/python/runtime_py3/main
)

NO_CHECK_IMPORTS()

NO_PYTHON_INCLUDES()

NO_PYTHON_COVERAGE()

SRCDIR(
    contrib/tools/python3/pycc
)

PY_SRCS(
    MAIN main.py
)

END()
