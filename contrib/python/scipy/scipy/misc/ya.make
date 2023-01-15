PY23_LIBRARY()

LICENSE(BSD-3-Clause)



NO_CHECK_IMPORTS(
    scipy.misc.pilutil
)

NO_LINT()

PY_SRCS(
    NAMESPACE scipy.misc

    __init__.py
    common.py
    doccer.py
    pilutil.py
)

END()
