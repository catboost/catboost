ENABLE(PYBUILD_NO_PY)

PY3_LIBRARY()



LICENSE(Python-2.0)

NO_PYTHON_INCLUDES()

PEERDIR(
    contrib/tools/python3/lib/py
)

INCLUDE(../../lib/srcs.cmake)

PY_SRCS(
    TOP_LEVEL
    ${PYTHON3_LIB_SRCS}
)

END()
