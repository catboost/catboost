PY23_LIBRARY()

LICENSE(BSD-3-Clause)



VERSION(0.4.2)

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
    # testpath/cli-32.exe
    # testpath/cli-64.exe
)

END()
