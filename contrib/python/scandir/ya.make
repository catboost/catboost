PY2_LIBRARY()

LICENSE(BSD-3-Clause)

VERSION(1.10.0)

NO_WERROR()
NO_WSHADOW()
NO_COMPILER_WARNINGS()



PY_REGISTER(_scandir)

SRCS(
    _scandir.c
)

PY_SRCS(
    TOP_LEVEL
    scandir.py
)

RESOURCE_FILES(
    PREFIX contrib/python/scandir/
    .dist-info/METADATA
    .dist-info/top_level.txt
)

NO_LINT()

END()

RECURSE_FOR_TESTS(
    tests
)
