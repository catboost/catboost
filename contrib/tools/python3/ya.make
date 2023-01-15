PROGRAM()



VERSION(3.7.7)

LICENSE(Python-2.0)

USE_PYTHON3()

PEERDIR(
    contrib/tools/python3/src/Modules/_sqlite
)

CFLAGS(
    -DPy_BUILD_CORE
)

SRCS(
    src/Programs/python.c
)

END()
