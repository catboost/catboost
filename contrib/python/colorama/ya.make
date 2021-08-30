PY23_LIBRARY()

LICENSE(BSD-3-Clause)



VERSION(0.4.4)

NO_LINT()

PY_SRCS(
    TOP_LEVEL
    colorama/__init__.py
    colorama/ansi.py
    colorama/ansitowin32.py
    colorama/initialise.py
    colorama/win32.py
    colorama/winterm.py
)

RESOURCE_FILES(
    PREFIX contrib/python/colorama/
    .dist-info/METADATA
    .dist-info/top_level.txt
)

END()
