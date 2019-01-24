PROGRAM()



LICENSE(Python-2.0)

PEERDIR(
    contrib/tools/python3/lib
    library/python/runtime_py3
)

ADDINCL(
    contrib/tools/python3/src/Include
)

SRCS(
    src/Programs/python.c
)

END()

RECURSE(
    lib
    src
    src/Lib
    src/Lib/lib2to3
)
