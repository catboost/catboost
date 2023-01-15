PY23_LIBRARY()



NO_LINT()

PY_SRCS(
    NAMESPACE scipy.special._precompute

    __init__.py
    expn_asy.py
    gammainc_asy.py
    utils.py
)

END()
