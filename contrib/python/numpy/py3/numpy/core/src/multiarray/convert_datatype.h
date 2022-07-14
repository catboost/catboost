#ifndef NUMPY_CORE_SRC_MULTIARRAY_CONVERT_DATATYPE_H_
#define NUMPY_CORE_SRC_MULTIARRAY_CONVERT_DATATYPE_H_

#include "array_method.h"

extern NPY_NO_EXPORT npy_intp REQUIRED_STR_LEN[];

NPY_NO_EXPORT PyObject *
PyArray_GetCastingImpl(PyArray_DTypeMeta *from, PyArray_DTypeMeta *to);

NPY_NO_EXPORT PyObject *
_get_castingimpl(PyObject *NPY_UNUSED(module), PyObject *args);

NPY_NO_EXPORT PyArray_VectorUnaryFunc *
PyArray_GetCastFunc(PyArray_Descr *descr, int type_num);

NPY_NO_EXPORT int
PyArray_ObjectType(PyObject *op, int minimum_type);

NPY_NO_EXPORT PyArrayObject **
PyArray_ConvertToCommonType(PyObject *op, int *retn);

NPY_NO_EXPORT PyArray_Descr *
PyArray_LegacyResultType(
        npy_intp narrs, PyArrayObject **arr,
        npy_intp ndtypes, PyArray_Descr **dtypes);

NPY_NO_EXPORT int
PyArray_ValidType(int type);

NPY_NO_EXPORT int
dtype_kind_to_ordering(char kind);

/* Used by PyArray_CanCastArrayTo and in the legacy ufunc type resolution */
NPY_NO_EXPORT npy_bool
can_cast_scalar_to(PyArray_Descr *scal_type, char *scal_data,
                    PyArray_Descr *to, NPY_CASTING casting);

NPY_NO_EXPORT PyArray_Descr *
ensure_dtype_nbo(PyArray_Descr *type);

NPY_NO_EXPORT int
should_use_min_scalar(npy_intp narrs, PyArrayObject **arr,
                      npy_intp ndtypes, PyArray_Descr **dtypes);

NPY_NO_EXPORT const char *
npy_casting_to_string(NPY_CASTING casting);

NPY_NO_EXPORT void
npy_set_invalid_cast_error(
        PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
        NPY_CASTING casting, npy_bool scalar);

NPY_NO_EXPORT PyArray_Descr *
PyArray_CastDescrToDType(PyArray_Descr *descr, PyArray_DTypeMeta *given_DType);

NPY_NO_EXPORT PyArray_Descr *
PyArray_FindConcatenationDescriptor(
        npy_intp n, PyArrayObject **arrays, PyObject *requested_dtype);

NPY_NO_EXPORT int
PyArray_AddCastingImplementation(PyBoundArrayMethodObject *meth);

NPY_NO_EXPORT int
PyArray_AddCastingImplementation_FromSpec(PyArrayMethod_Spec *spec, int private);

NPY_NO_EXPORT NPY_CASTING
PyArray_MinCastSafety(NPY_CASTING casting1, NPY_CASTING casting2);

NPY_NO_EXPORT NPY_CASTING
PyArray_GetCastSafety(
        PyArray_Descr *from, PyArray_Descr *to, PyArray_DTypeMeta *to_dtype);

NPY_NO_EXPORT int
PyArray_CheckCastSafety(NPY_CASTING casting,
        PyArray_Descr *from, PyArray_Descr *to, PyArray_DTypeMeta *to_dtype);

NPY_NO_EXPORT NPY_CASTING
legacy_same_dtype_resolve_descriptors(
        PyArrayMethodObject *self,
        PyArray_DTypeMeta *dtypes[2],
        PyArray_Descr *given_descrs[2],
        PyArray_Descr *loop_descrs[2]);

NPY_NO_EXPORT int
legacy_cast_get_strided_loop(
        PyArrayMethod_Context *context,
        int aligned, int move_references, npy_intp *strides,
        PyArrayMethod_StridedLoop **out_loop, NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags);

NPY_NO_EXPORT NPY_CASTING
simple_cast_resolve_descriptors(
        PyArrayMethodObject *self,
        PyArray_DTypeMeta *dtypes[2],
        PyArray_Descr *input_descrs[2],
        PyArray_Descr *loop_descrs[2]);

NPY_NO_EXPORT int
PyArray_InitializeCasts(void);

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_CONVERT_DATATYPE_H_ */
