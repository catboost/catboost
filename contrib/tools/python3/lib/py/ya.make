ENABLE(PYBUILD_NO_PYC)

PY3_LIBRARY()



LICENSE(Python-2.0)

NO_PYTHON_INCLUDES()

SRCDIR(contrib/tools/python3/src/Lib)

INCLUDE(../srcs.cmake)

PY_SRCS(
    TOP_LEVEL
    ${PYTHON3_LIB_SRCS}
)

END()
