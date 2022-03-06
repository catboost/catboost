PY2TEST()



VERSION(1.2.3)

ORIGINAL_SOURCE(mirror://pypi/s/scipy/scipy-1.2.3.tar.gz)

SIZE(MEDIUM)

FORK_TESTS()

PEERDIR(
    contrib/python/scipy/py2
    contrib/python/scipy/py2/scipy/conftest
)

NO_LINT()

NO_CHECK_IMPORTS()

TEST_SRCS(
    __init__.py
    test_connected_components.py
    test_conversions.py
    test_graph_laplacian.py
    test_reordering.py
    test_shortest_path.py
    test_spanning_tree.py
    test_traversal.py
)

END()
