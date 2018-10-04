

PY23_LIBRARY()

PY_SRCS(
    ya.py
    collection.py
    conftests.py
    fixtures.py
    newinterpret.py
    reinterpret.py
)

PEERDIR(
    library/python/find_root
)

END()

NEED_CHECK()
