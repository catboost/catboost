PY23_LIBRARY()

# Proxy library
LICENSE(Not-Applicable)



IF (PYTHON2)
    PEERDIR(contrib/python/MarkupSafe/py2)
ELSE()
    PEERDIR(contrib/python/MarkupSafe/py3)
ENDIF()

NO_LINT()

END()

RECURSE(
    py2
    py3
)
