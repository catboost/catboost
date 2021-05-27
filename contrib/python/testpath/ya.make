PY23_LIBRARY()

LICENSE(BSD-3-Clause)



VERSION(0.5.0)

NO_LINT()

PY_SRCS(
    TOP_LEVEL
    testpath/__init__.py
    testpath/asserts.py
    testpath/commands.py
    testpath/env.py
    testpath/tempdir.py
)

RESOURCE_FILES(
    PREFIX contrib/python/testpath/
    .dist-info/METADATA
    .dist-info/top_level.txt
)

END()
