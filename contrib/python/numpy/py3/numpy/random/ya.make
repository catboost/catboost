PY3_LIBRARY()

LICENSE(BSD-3-Clause)



ADDINCL(
    FOR cython contrib/python/numpy/py3
    contrib/python/numpy/include/numpy/core/include
    contrib/python/numpy/include/numpy/core/include/numpy
    contrib/python/numpy/include/numpy/core/src/common
    contrib/python/numpy/include/numpy/core/src/npymath
    contrib/python/numpy/include/numpy/random
)

CFLAGS(
    -DHAVE_CBLAS
    -DHAVE_NPY_CONFIG_H=1
    -DNO_ATLAS_INFO=1
    -D_FILE_OFFSET_BITS=64
    -D_LARGEFILE64_SOURCE=1
    -D_LARGEFILE_SOURCE=1
    -DNP_RANDOM_LEGACY=1
    -DNPY_NO_DEPRECATED_API=0
)

SRCS(
    src/distributions/logfactorial.c
    src/distributions/distributions.c
    src/distributions/random_mvhg_count.c
    src/distributions/random_mvhg_marginals.c
    src/distributions/random_hypergeometric.c
    src/legacy/legacy-distributions.c
    src/mt19937/mt19937.c
    src/mt19937/mt19937-jump.c
    src/philox/philox.c
    src/pcg64/pcg64.c
    src/sfc64/sfc64.c
)

PY_SRCS(
    NAMESPACE numpy.random
    __init__.py
    _pickle.py

    CYTHON_C
    _bounded_integers.pyx
    _common.pyx
    _generator.pyx
    _mt19937.pyx
    _pcg64.pyx
    _philox.pyx
    _sfc64.pyx
    bit_generator.pyx
    mtrand.pyx
)

NO_LINT()

NO_COMPILER_WARNINGS()

END()
