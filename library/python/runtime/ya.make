LIBRARY()



NO_WSHADOW()

PEERDIR(
    library/resource
)

PY_SRCS(
    entry_points.py
    import_test.py

    TOP_LEVEL
    __res.pyx
    __run_import_test__.py  # delete after the next test_tool release
    sitecustomize.pyx
)

END()
