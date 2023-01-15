PY23_LIBRARY()



PEERDIR(
    contrib/python/more-itertools
)

TEST_SRCS(
    test_more.py
    test_recipes.py
)

NO_LINT()

END()

RECURSE_FOR_TESTS(
    py2
    py3
)
