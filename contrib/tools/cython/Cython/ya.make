PY23_LIBRARY()

WITHOUT_LICENSE_TEXTS()

OWNER(g:yatool)

NO_LINT()

# Minimal set of the files required to support coverage (DEVTOOLS-4095)
PY_SRCS(
    __init__.py
    Coverage.py
    Shadow.py
    Utils.py
)

PEERDIR(
    contrib/python/six
    library/python/resource
)

END()
