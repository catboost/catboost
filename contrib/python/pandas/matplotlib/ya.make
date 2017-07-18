PY_LIBRARY()

# Version: 0.19.1



PEERDIR(
    contrib/python/matplotlib-2.0.0b3
    contrib/python/pandas
)

SRCDIR(contrib/python/pandas)

PY_SRCS(
    TOP_LEVEL
    pandas/tseries/converter.py
    pandas/tseries/plotting.py
)

END()
