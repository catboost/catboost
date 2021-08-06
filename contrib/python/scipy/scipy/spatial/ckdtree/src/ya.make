PY23_LIBRARY()

LICENSE(BSD-3-Clause)



PEERDIR(
    contrib/python/numpy
)

NO_COMPILER_WARNINGS()

SRCS(
    build.cxx
    count_neighbors.cxx
    cpp_exc.cxx
    globals.cxx
    query_ball_point.cxx
    query_ball_tree.cxx
    query.cxx
    query_pairs.cxx
    sparse_distances.cxx
)

END()
