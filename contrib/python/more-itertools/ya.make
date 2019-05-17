PY23_LIBRARY()



VERSION(5.0.0)

LICENSE(MIT)

PEERDIR(
    contrib/python/six
)

NO_LINT()

PY_SRCS(
    TOP_LEVEL
    more_itertools/__init__.py
    more_itertools/more.py
    more_itertools/recipes.py
)

RESOURCE_FILES(
    PREFIX contrib/python/more-itertools/
    .dist-info/METADATA
    .dist-info/top_level.txt
)

END()

RECURSE_FOR_TESTS(
    tests
)
