

PY2_LIBRARY()

LICENSE(BSD-3-Clause)

VERSION(1.1.1)

PY_SRCS(
    TOP_LEVEL
    markupsafe/_compat.py
    markupsafe/_constants.py
    markupsafe/__init__.py
    markupsafe/_native.py
)

#SRCS(
#    markupsafe/_speedups.c
#)

#PY_REGISTER(
#    markupsafe._speedups
#)

NO_LINT()

NO_COMPILER_WARNINGS()

RESOURCE_FILES(
    PREFIX contrib/python/MarkupSafe/py2/
    .dist-info/METADATA
    .dist-info/top_level.txt
)

END()

RECURSE_FOR_TESTS(
    tests
)
