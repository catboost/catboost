PROGRAM()



VERSION(3.8.5)

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
