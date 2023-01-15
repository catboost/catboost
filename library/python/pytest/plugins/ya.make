

PY23_LIBRARY()

PY_SRCS(
    ya.py
    collection.py
    conftests.py
    fixtures.py
)

PEERDIR(
    library/python/find_root
    library/python/testing/filter
)

END()
