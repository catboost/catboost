PY23_LIBRARY()

LICENSE(BSD-3-Clause)



NO_LINT()

PY_SRCS(
    NAMESPACE scipy.special._precompute

    __init__.py
    expn_asy.py
    gammainc_asy.py
    utils.py
)

END()
