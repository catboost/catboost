PY2_LIBRARY() # Backport from Python 3.

LICENSE(MIT)

VERSION(1.0.0)



PY_SRCS(
    TOP_LEVEL
    backports/shutil_get_terminal_size/__init__.py
    backports/shutil_get_terminal_size/get_terminal_size.py
)

NO_LINT()

END()
