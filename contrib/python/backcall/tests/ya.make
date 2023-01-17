PY3TEST()



PEERDIR(
    contrib/python/backcall
)

TEST_SRCS(
    test_callback_prototypes.py
)

NO_LINT()

END()
