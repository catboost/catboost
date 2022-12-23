PY2TEST()



PEERDIR(
    contrib/deprecated/python/scandir
)

TEST_SRCS(
    test_scandir.py
    test_walk.py
)

DATA(
    arcadia/contrib/deprecated/python/scandir/tests
)

NO_LINT()

END()
