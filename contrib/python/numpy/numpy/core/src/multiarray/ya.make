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
    contrib/python/numpy/numpy/core/src/umath
)

SRCS(
    _multiarray_tests.c
    alloc.c
    array_assign_array.c
    array_assign_scalar.c
    arrayfunction_override.c
    arrayobject.c
    arraytypes.c
    buffer.c
    calculation.c
    common.c
    compiled_base.c
    conversion_utils.c
    convert.c
    convert_datatype.c
    ctors.c
    datetime.c
    datetime_busday.c
    datetime_busdaycal.c
    datetime_strings.c
    descriptor.c
    dragon4.c
    dtype_transfer.c
    einsum.c
    flagsobject.c
    getset.c
    hashdescr.c
    item_selection.c
    iterators.c
    lowlevel_strided_loops.c
    mapping.c
    methods.c
    multiarraymodule.c
    nditer_api.c
    nditer_constr.c
    nditer_pywrap.c
    nditer_templ.c
    number.c
    refcount.c
    scalarapi.c
    scalartypes.c
    sequence.c
    shape.c
    strfuncs.c
    temp_elide.c
    typeinfo.c
    usertypes.c
    vdot.c
)

PY_REGISTER(numpy.core._multiarray_umath)
PY_REGISTER(numpy.core._multiarray_tests)

END()
