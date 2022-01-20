PY2TEST()



PEERDIR(
    contrib/python/more-itertools
)

SRCDIR(contrib/python/more-itertools/py2/more_itertools/tests)

TEST_SRCS(
    test_more.py
    test_recipes.py
)

NO_LINT()

END()
