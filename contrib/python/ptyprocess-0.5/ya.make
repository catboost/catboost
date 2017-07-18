LIBRARY(ptyprocess)



PY_SRCS(
    TOP_LEVEL
    ptyprocess/__init__.py
    ptyprocess/_fork_pty.py
    ptyprocess/ptyprocess.py
    ptyprocess/util.py
)

NO_LINT()

END()
