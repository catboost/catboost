LIBRARY()



USE_PYTHON()

NO_COMPILER_WARNINGS()

CFLAGS(
    -DHAVE_CBLAS
    -DHAVE_NPY_CONFIG_H=1
    -DNO_ATLAS_INFO=1
    -D_FILE_OFFSET_BITS=64
    -D_LARGEFILE64_SOURCE=1
    -D_LARGEFILE_SOURCE=1
)

ADDINCL(
    contrib/python/numpy-1.11.1/numpy/core/include
    contrib/python/numpy-1.11.1/numpy/core/include/numpy
    contrib/python/numpy-1.11.1/numpy/core/src/private
    contrib/python/numpy-1.11.1/numpy/core/src/multiarray
)

IF (NOT MSVC)
    SRCS(
        python_xerbla.c
    )
ENDIF()

SRCS(
    alloc.c
    array_assign.c
    array_assign_array.c
    array_assign_scalar.c
    arrayobject.c
    arraytypes.c
    buffer.c
    calculation.c
    cblasfuncs.c
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
    multiarray_tests.c
    multiarraymodule.c
    nditer_api.c
    nditer_constr.c
    nditer_pywrap.c
    nditer_templ.c
    number.c
    numpymemoryview.c
    numpyos.c
    refcount.c
    scalarapi.c
    scalartypes.c
    sequence.c
    shape.c
    ucsnarrow.c
    usertypes.c
    vdot.c
)

PY_REGISTER(numpy.core.multiarray)

END()
