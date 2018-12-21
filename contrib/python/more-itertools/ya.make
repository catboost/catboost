PY23_LIBRARY()



VERSION(4.3.0)

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

END()
