LIBRARY()



PEERDIR(
    contrib/python/scipy-0.18.1/scipy
)

END()

RECURSE(
    scipy/stats/tests
)
