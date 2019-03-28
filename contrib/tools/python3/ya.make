PROGRAM()



LICENSE(Python-2.0)

USE_PYTHON3()

SRCS(
    src/Programs/python.c
)

END()

RECURSE(
    lib
    lib/py
    src
    src/Lib
    src/Lib/lib2to3
)
