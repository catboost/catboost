PY23_LIBRARY()

LICENSE(BSD3)



VERSION(0.18.1)

PEERDIR(
    contrib/python/scipy/scipy
)

END()

RECURSE(
    scipy/linalg/src/lapack_deprecations
)

RECURSE_FOR_TESTS(
    scipy/_build_utils/tests
    scipy/cluster/tests
    scipy/constants/tests
    scipy/fftpack/tests
    scipy/integrate/tests
    scipy/interpolate/tests
    scipy/io/arff/tests
    scipy/io/harwell_boeing/tests
    scipy/io/matlab/tests
    scipy/io/tests
    scipy/_lib/tests
    scipy/linalg/tests
    scipy/misc/tests
    scipy/ndimage/tests
    scipy/odr/tests
    scipy/optimize/tests
    scipy/signal/tests
    scipy/sparse/csgraph/tests
    scipy/sparse/linalg/dsolve/tests
    scipy/sparse/linalg/eigen/arpack/tests
    scipy/sparse/linalg/eigen/lobpcg/tests
    scipy/sparse/linalg/isolve/tests
    scipy/sparse/linalg/tests
    scipy/sparse/tests
    scipy/spatial/tests
    scipy/special/tests
    scipy/stats/tests
)
