PY3_PROGRAM()



PEERDIR(
    contrib/python/ipython
)

PY_SRCS(
    MAIN
    __main__.py=main
    crash.py
    mod/__init__.py
)

END()
