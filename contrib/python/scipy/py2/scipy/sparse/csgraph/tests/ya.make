PROGRAM()



NO_LINT()
NO_CHECK_IMPORTS()

PEERDIR(
    contrib/python/nose/runner

    contrib/python/scipy
)

TEST_SRCS(
    test_connected_components.py
    test_conversions.py
    test_graph_components.py
    test_graph_laplacian.py
    test_reordering.py
    test_shortest_path.py
    test_spanning_tree.py
    test_traversal.py
)

END()
