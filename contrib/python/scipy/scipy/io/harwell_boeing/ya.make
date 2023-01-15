PY23_LIBRARY()

LICENSE(BSD-3-Clause)



NO_LINT()

PY_SRCS(
    NAMESPACE scipy.io.harwell_boeing

    __init__.py
    _fortran_format_parser.py
    hb.py
)

END()
