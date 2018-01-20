LIBRARY(colorama)

VERSION(0.3.6)

LICENSE(
    BSD3
)



PY_SRCS(
    TOP_LEVEL
    colorama/__init__.py
    colorama/ansi.py
    colorama/ansitowin32.py
    colorama/initialise.py
    colorama/win32.py
    colorama/winterm.py
)

NO_LINT()

END()
