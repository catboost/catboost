PY23_LIBRARY()



NO_CHECK_IMPORTS(
    scipy.misc.__init__ # XXX: fixme
)

PY_SRCS(
    NAMESPACE scipy.misc

    __init__.py
    common.py
    doccer.py
#    pilutil.py
)

END()
