PY23_LIBRARY()

LICENSE(
    BSD3
)



NO_COMPILER_WARNINGS()

CFLAGS(
    -DHAVE_CBLAS
    -DHAVE_NPY_CONFIG_H=1
    -DNO_ATLAS_INFO=1
    -D_FILE_OFFSET_BITS=64
    -D_LARGEFILE64_SOURCE=1
    -D_LARGEFILE_SOURCE=1
    -DNPY_INTERNAL_BUILD=1
)

ADDINCL(
    contrib/python/numpy/numpy/core/include
    contrib/python/numpy/numpy/core/include/numpy
    contrib/python/numpy/numpy/core/src/common
    contrib/python/numpy/numpy/core/src/multiarray
    contrib/python/numpy/numpy/core/src/npymath
)

SRCS(
    _operand_flag_tests.c
    _rational_tests.c
    _struct_ufunc_tests.c
    _umath_tests.c
    cpuid.c
    extobj.c
    loops.c
    matmul.c
    override.c
    reduction.c
    scalarmath.c
    ufunc_object.c
    ufunc_type_resolution.c
    umathmodule.c
)

PY_REGISTER(numpy.core._operand_flag_tests)
PY_REGISTER(numpy.core._rational_tests)
PY_REGISTER(numpy.core._struct_ufunc_tests)
PY_REGISTER(numpy.core._umath_tests)

END()
