PY2TEST()



PEERDIR(
    contrib/python/scandir
)

TEST_SRCS(
    test_scandir.py
    test_walk.py
)

DATA(
    arcadia/contrib/python/scandir/tests
)

NO_LINT()

END()
