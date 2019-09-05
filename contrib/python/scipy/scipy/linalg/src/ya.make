LIBRARY()



PEERDIR(
    contrib/python/scipy/scipy/linalg/src/id_dist/src
)

SRCS(
    calc_lwork.f
    det.f
    lu.f
)

END()

RECURSE(
    lapack_deprecations
)
