PYTEST()



DEPENDS(library/python/runtime/test/traceback)

TEST_SRCS(test_traceback.py)

END()

RECURSE_FOR_TESTS(traceback)
