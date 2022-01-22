PY23_LIBRARY()

LICENSE(Service-Py23-Proxy)



IF (PYTHON2)
    PEERDIR(contrib/python/tornado/tornado-4)
ELSE()
    PEERDIR(contrib/python/tornado/tornado-6)
ENDIF()

NO_LINT()

END()

RECURSE(
    tornado-6
)
