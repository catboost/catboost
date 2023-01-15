PROGRAM()



NO_LINT()
NO_CHECK_IMPORTS()

DATA(
    arcadia/contrib/python/scipy/scipy/spatial/tests
)

PEERDIR(
    contrib/python/nose/runner

    contrib/python/scipy
)

TEST_SRCS(
    test_distance.py
    test_kdtree.py
    test__plotutils.py
    test__procrustes.py
    test_qhull.py
    test_spherical_voronoi.py
)

END()
