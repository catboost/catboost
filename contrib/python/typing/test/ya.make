PY2TEST()



PEERDIR(
    contrib/python/typing
)

TEST_SRCS(
    mod_generics_cache.py
    test_typing.py
)

NO_LINT()

END()
