PY23_LIBRARY()



TEST_SRCS(test_ctypes.py)

PEERDIR(
    library/python/symbols/python
)

END()

RECURSE_FOR_TESTS(
    py2
    py3
)
