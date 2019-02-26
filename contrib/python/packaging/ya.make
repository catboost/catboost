PY23_LIBRARY()

LICENSE(
    APACHE
    BSD
)
 


VERSION(18.0)

# License: BSD or Apache License, Version 2.0

PEERDIR(
    contrib/python/pyparsing
    contrib/python/six
)

PY_SRCS(
    TOP_LEVEL
    packaging/__about__.py
    packaging/__init__.py
    packaging/_compat.py
    packaging/_structures.py
    packaging/markers.py
    packaging/requirements.py
    packaging/specifiers.py
    packaging/utils.py
    packaging/version.py
)

END()
