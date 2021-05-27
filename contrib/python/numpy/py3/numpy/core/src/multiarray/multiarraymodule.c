/*
  Python Multiarray Module -- A useful collection of functions for creating and
  using ndarrays

  Original file
  Copyright (c) 1995, 1996, 1997 Jim Hugunin, hugunin@mit.edu

  Modified for numpy in 2005

  Travis E. Oliphant
  oliphant@ee.byu.edu
  Brigham Young University
*/

/* $Id: multiarraymodule.c,v 1.36 2005/09/14 00:14:00 teoliphant Exp $ */

#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "structmember.h"

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _UMATHMODULE
#define _MULTIARRAYMODULE
#include <numpy/npy_common.h>
#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"

#include "numpy/npy_math.h"

#include "npy_config.h"
#include "npy_pycompat.h"
#include "npy_import.h"

NPY_NO_EXPORT int NPY_NUMUSERTYPES = 0;

/* Internal APIs */
#include "alloc.h"
#include "arrayfunction_override.h"
#include "arraytypes.h"
#include "arrayobject.h"
#include "hashdescr.h"
#include "descriptor.h"
#include "dragon4.h"
#include "calculation.h"
#include "number.h"
#include "scalartypes.h"
#include "convert_datatype.h"
#include "conversion_utils.h"
#include "nditer_pywrap.h"
#include "methods.h"
#include "_datetime.h"
#include "datetime_strings.h"
#include "datetime_busday.h"
#include "datetime_busdaycal.h"
#include "item_selection.h"
#include "shape.h"
#include "ctors.h"
#include "array_assign.h"
#include "common.h"
#include "multiarraymodule.h"
#include "cblasfuncs.h"
#include "vdot.h"
#include "templ_common.h" /* for npy_mul_with_overflow_intp */
#include "compiled_base.h"
#include "mem_overlap.h"
#include "alloc.h"
#include "typeinfo.h"

#include "get_attr_string.h"

/*
 *****************************************************************************
 **                    INCLUDE GENERATED CODE                               **
 *****************************************************************************
 */
#include "funcs.inc"
#include "umathmodule.h"

NPY_NO_EXPORT int initscalarmath(PyObject *);
NPY_NO_EXPORT int set_matmul_flags(PyObject *d); /* in ufunc_object.c */

/*
 * global variable to determine if legacy printing is enabled, accessible from
 * C. For simplicity the mode is encoded as an integer where '0' means no
 * legacy mode, and '113' means 1.13 legacy mode. We can upgrade this if we
 * have more complex requirements in the future.
 */
int npy_legacy_print_mode = 0;

static PyObject *
set_legacy_print_mode(PyObject *NPY_UNUSED(self), PyObject *args)
{
    if (!PyArg_ParseTuple(args, "i", &npy_legacy_print_mode)) {
        return NULL;
    }
    Py_RETURN_NONE;
}


/* Only here for API compatibility */
NPY_NO_EXPORT PyTypeObject PyBigArray_Type;


/*NUMPY_API
 * Get Priority from object
 */
NPY_NO_EXPORT double
PyArray_GetPriority(PyObject *obj, double default_)
{
    PyObject *ret;
    double priority = NPY_PRIORITY;

    if (PyArray_CheckExact(obj)) {
        return priority;
    }
    else if (PyArray_CheckAnyScalarExact(obj)) {
        return NPY_SCALAR_PRIORITY;
    }

    ret = PyArray_LookupSpecial_OnInstance(obj, "__array_priority__");
    if (ret == NULL) {
        if (PyErr_Occurred()) {
            PyErr_Clear(); /* TODO[gh-14801]: propagate crashes during attribute access? */
        }
        return default_;
    }

    priority = PyFloat_AsDouble(ret);
    Py_DECREF(ret);
    return priority;
}

/*NUMPY_API
 * Multiply a List of ints
 */
NPY_NO_EXPORT int
PyArray_MultiplyIntList(int const *l1, int n)
{
    int s = 1;

    while (n--) {
        s *= (*l1++);
    }
    return s;
}

/*NUMPY_API
 * Multiply a List
 */
NPY_NO_EXPORT npy_intp
PyArray_MultiplyList(npy_intp const *l1, int n)
{
    npy_intp s = 1;

    while (n--) {
        s *= (*l1++);
    }
    return s;
}

/*NUMPY_API
 * Multiply a List of Non-negative numbers with over-flow detection.
 */
NPY_NO_EXPORT npy_intp
PyArray_OverflowMultiplyList(npy_intp const *l1, int n)
{
    npy_intp prod = 1;
    int i;

    for (i = 0; i < n; i++) {
        npy_intp dim = l1[i];

        if (dim == 0) {
            return 0;
        }
        if (npy_mul_with_overflow_intp(&prod, prod, dim)) {
            return -1;
        }
    }
    return prod;
}

/*NUMPY_API
 * Produce a pointer into array
 */
NPY_NO_EXPORT void *
PyArray_GetPtr(PyArrayObject *obj, npy_intp const* ind)
{
    int n = PyArray_NDIM(obj);
    npy_intp *strides = PyArray_STRIDES(obj);
    char *dptr = PyArray_DATA(obj);

    while (n--) {
        dptr += (*strides++) * (*ind++);
    }
    return (void *)dptr;
}

/*NUMPY_API
 * Compare Lists
 */
NPY_NO_EXPORT int
PyArray_CompareLists(npy_intp const *l1, npy_intp const *l2, int n)
{
    int i;

    for (i = 0; i < n; i++) {
        if (l1[i] != l2[i]) {
            return 0;
        }
    }
    return 1;
}

/*
 * simulates a C-style 1-3 dimensional array which can be accessed using
 * ptr[i]  or ptr[i][j] or ptr[i][j][k] -- requires pointer allocation
 * for 2-d and 3-d.
 *
 * For 2-d and up, ptr is NOT equivalent to a statically defined
 * 2-d or 3-d array.  In particular, it cannot be passed into a
 * function that requires a true pointer to a fixed-size array.
 */

/*NUMPY_API
 * Simulate a C-array
 * steals a reference to typedescr -- can be NULL
 */
NPY_NO_EXPORT int
PyArray_AsCArray(PyObject **op, void *ptr, npy_intp *dims, int nd,
                 PyArray_Descr* typedescr)
{
    PyArrayObject *ap;
    npy_intp n, m, i, j;
    char **ptr2;
    char ***ptr3;

    if ((nd < 1) || (nd > 3)) {
        PyErr_SetString(PyExc_ValueError,
                        "C arrays of only 1-3 dimensions available");
        Py_XDECREF(typedescr);
        return -1;
    }
    if ((ap = (PyArrayObject*)PyArray_FromAny(*op, typedescr, nd, nd,
                                      NPY_ARRAY_CARRAY, NULL)) == NULL) {
        return -1;
    }
    switch(nd) {
    case 1:
        *((char **)ptr) = PyArray_DATA(ap);
        break;
    case 2:
        n = PyArray_DIMS(ap)[0];
        ptr2 = (char **)PyArray_malloc(n * sizeof(char *));
        if (!ptr2) {
            PyErr_NoMemory();
            return -1;
        }
        for (i = 0; i < n; i++) {
            ptr2[i] = PyArray_BYTES(ap) + i*PyArray_STRIDES(ap)[0];
        }
        *((char ***)ptr) = ptr2;
        break;
    case 3:
        n = PyArray_DIMS(ap)[0];
        m = PyArray_DIMS(ap)[1];
        ptr3 = (char ***)PyArray_malloc(n*(m+1) * sizeof(char *));
        if (!ptr3) {
            PyErr_NoMemory();
            return -1;
        }
        for (i = 0; i < n; i++) {
            ptr3[i] = (char **) &ptr3[n + m * i];
            for (j = 0; j < m; j++) {
                ptr3[i][j] = PyArray_BYTES(ap) + i*PyArray_STRIDES(ap)[0] + j*PyArray_STRIDES(ap)[1];
            }
        }
        *((char ****)ptr) = ptr3;
    }
    if (nd) {
        memcpy(dims, PyArray_DIMS(ap), nd*sizeof(npy_intp));
    }
    *op = (PyObject *)ap;
    return 0;
}

/* Deprecated --- Use PyArray_AsCArray instead */

/*NUMPY_API
 * Convert to a 1D C-array
 */
NPY_NO_EXPORT int
PyArray_As1D(PyObject **NPY_UNUSED(op), char **NPY_UNUSED(ptr),
             int *NPY_UNUSED(d1), int NPY_UNUSED(typecode))
{
    /* 2008-07-14, 1.5 */
    PyErr_SetString(PyExc_NotImplementedError,
                "PyArray_As1D: use PyArray_AsCArray.");
    return -1;
}

/*NUMPY_API
 * Convert to a 2D C-array
 */
NPY_NO_EXPORT int
PyArray_As2D(PyObject **NPY_UNUSED(op), char ***NPY_UNUSED(ptr),
             int *NPY_UNUSED(d1), int *NPY_UNUSED(d2), int NPY_UNUSED(typecode))
{
    /* 2008-07-14, 1.5 */
    PyErr_SetString(PyExc_NotImplementedError,
                "PyArray_As2D: use PyArray_AsCArray.");
    return -1;
}

/* End Deprecated */

/*NUMPY_API
 * Free pointers created if As2D is called
 */
NPY_NO_EXPORT int
PyArray_Free(PyObject *op, void *ptr)
{
    PyArrayObject *ap = (PyArrayObject *)op;

    if ((PyArray_NDIM(ap) < 1) || (PyArray_NDIM(ap) > 3)) {
        return -1;
    }
    if (PyArray_NDIM(ap) >= 2) {
        PyArray_free(ptr);
    }
    Py_DECREF(ap);
    return 0;
}

/*
 * Get the ndarray subclass with the highest priority
 */
NPY_NO_EXPORT PyTypeObject *
PyArray_GetSubType(int narrays, PyArrayObject **arrays) {
    PyTypeObject *subtype = &PyArray_Type;
    double priority = NPY_PRIORITY;
    int i;

    /* Get the priority subtype for the array */
    for (i = 0; i < narrays; ++i) {
        if (Py_TYPE(arrays[i]) != subtype) {
            double pr = PyArray_GetPriority((PyObject *)(arrays[i]), 0.0);
            if (pr > priority) {
                priority = pr;
                subtype = Py_TYPE(arrays[i]);
            }
        }
    }

    return subtype;
}


/*
 * Concatenates a list of ndarrays.
 */
NPY_NO_EXPORT PyArrayObject *
PyArray_ConcatenateArrays(int narrays, PyArrayObject **arrays, int axis,
                          PyArrayObject* ret)
{
    int iarrays, idim, ndim;
    npy_intp shape[NPY_MAXDIMS];
    PyArrayObject_fields *sliding_view = NULL;

    if (narrays <= 0) {
        PyErr_SetString(PyExc_ValueError,
                        "need at least one array to concatenate");
        return NULL;
    }

    /* All the arrays must have the same 'ndim' */
    ndim = PyArray_NDIM(arrays[0]);

    if (ndim == 0) {
        PyErr_SetString(PyExc_ValueError,
                        "zero-dimensional arrays cannot be concatenated");
        return NULL;
    }

    /* Handle standard Python negative indexing */
    if (check_and_adjust_axis(&axis, ndim) < 0) {
        return NULL;
    }

    /*
     * Figure out the final concatenated shape starting from the first
     * array's shape.
     */
    memcpy(shape, PyArray_SHAPE(arrays[0]), ndim * sizeof(shape[0]));
    for (iarrays = 1; iarrays < narrays; ++iarrays) {
        npy_intp *arr_shape;

        if (PyArray_NDIM(arrays[iarrays]) != ndim) {
            PyErr_Format(PyExc_ValueError,
                         "all the input arrays must have same number of "
                         "dimensions, but the array at index %d has %d "
                         "dimension(s) and the array at index %d has %d "
                         "dimension(s)",
                         0, ndim, iarrays, PyArray_NDIM(arrays[iarrays]));
            return NULL;
        }
        arr_shape = PyArray_SHAPE(arrays[iarrays]);

        for (idim = 0; idim < ndim; ++idim) {
            /* Build up the size of the concatenation axis */
            if (idim == axis) {
                shape[idim] += arr_shape[idim];
            }
            /* Validate that the rest of the dimensions match */
            else if (shape[idim] != arr_shape[idim]) {
                PyErr_Format(PyExc_ValueError,
                             "all the input array dimensions for the "
                             "concatenation axis must match exactly, but "
                             "along dimension %d, the array at index %d has "
                             "size %d and the array at index %d has size %d",
                             idim, 0, shape[idim], iarrays, arr_shape[idim]);
                return NULL;
            }
        }
    }

    if (ret != NULL) {
        if (PyArray_NDIM(ret) != ndim) {
            PyErr_SetString(PyExc_ValueError,
                            "Output array has wrong dimensionality");
            return NULL;
        }
        if (!PyArray_CompareLists(shape, PyArray_SHAPE(ret), ndim)) {
            PyErr_SetString(PyExc_ValueError,
                            "Output array is the wrong shape");
            return NULL;
        }
        Py_INCREF(ret);
    }
    else {
        npy_intp s, strides[NPY_MAXDIMS];
        int strideperm[NPY_MAXDIMS];

        /* Get the priority subtype for the array */
        PyTypeObject *subtype = PyArray_GetSubType(narrays, arrays);

        /* Get the resulting dtype from combining all the arrays */
        PyArray_Descr *dtype = PyArray_ResultType(narrays, arrays, 0, NULL);
        if (dtype == NULL) {
            return NULL;
        }

        /*
         * Figure out the permutation to apply to the strides to match
         * the memory layout of the input arrays, using ambiguity
         * resolution rules matching that of the NpyIter.
         */
        PyArray_CreateMultiSortedStridePerm(narrays, arrays, ndim, strideperm);
        s = dtype->elsize;
        for (idim = ndim-1; idim >= 0; --idim) {
            int iperm = strideperm[idim];
            strides[iperm] = s;
            s *= shape[iperm];
        }

        /* Allocate the array for the result. This steals the 'dtype' reference. */
        ret = (PyArrayObject *)PyArray_NewFromDescr(subtype,
                                                        dtype,
                                                        ndim,
                                                        shape,
                                                        strides,
                                                        NULL,
                                                        0,
                                                        NULL);
        if (ret == NULL) {
            return NULL;
        }
    }

    /*
     * Create a view which slides through ret for assigning the
     * successive input arrays.
     */
    sliding_view = (PyArrayObject_fields *)PyArray_View(ret,
                                                        NULL, &PyArray_Type);
    if (sliding_view == NULL) {
        Py_DECREF(ret);
        return NULL;
    }
    for (iarrays = 0; iarrays < narrays; ++iarrays) {
        /* Set the dimension to match the input array's */
        sliding_view->dimensions[axis] = PyArray_SHAPE(arrays[iarrays])[axis];

        /* Copy the data for this array */
        if (PyArray_AssignArray((PyArrayObject *)sliding_view, arrays[iarrays],
                            NULL, NPY_SAME_KIND_CASTING) < 0) {
            Py_DECREF(sliding_view);
            Py_DECREF(ret);
            return NULL;
        }

        /* Slide to the start of the next window */
        sliding_view->data += sliding_view->dimensions[axis] *
                                 sliding_view->strides[axis];
    }

    Py_DECREF(sliding_view);
    return ret;
}

/*
 * Concatenates a list of ndarrays, flattening each in the specified order.
 */
NPY_NO_EXPORT PyArrayObject *
PyArray_ConcatenateFlattenedArrays(int narrays, PyArrayObject **arrays,
                                    NPY_ORDER order, PyArrayObject *ret)
{
    int iarrays;
    npy_intp shape = 0;
    PyArrayObject_fields *sliding_view = NULL;

    if (narrays <= 0) {
        PyErr_SetString(PyExc_ValueError,
                        "need at least one array to concatenate");
        return NULL;
    }

    /*
     * Figure out the final concatenated shape starting from the first
     * array's shape.
     */
    for (iarrays = 0; iarrays < narrays; ++iarrays) {
        shape += PyArray_SIZE(arrays[iarrays]);
        /* Check for overflow */
        if (shape < 0) {
            PyErr_SetString(PyExc_ValueError,
                            "total number of elements "
                            "too large to concatenate");
            return NULL;
        }
    }

    if (ret != NULL) {
        if (PyArray_NDIM(ret) != 1) {
            PyErr_SetString(PyExc_ValueError,
                            "Output array must be 1D");
            return NULL;
        }
        if (shape != PyArray_SIZE(ret)) {
            PyErr_SetString(PyExc_ValueError,
                            "Output array is the wrong size");
            return NULL;
        }
        Py_INCREF(ret);
    }
    else {
        npy_intp stride;

        /* Get the priority subtype for the array */
        PyTypeObject *subtype = PyArray_GetSubType(narrays, arrays);

        /* Get the resulting dtype from combining all the arrays */
        PyArray_Descr *dtype = PyArray_ResultType(narrays, arrays, 0, NULL);
        if (dtype == NULL) {
            return NULL;
        }

        stride = dtype->elsize;

        /* Allocate the array for the result. This steals the 'dtype' reference. */
        ret = (PyArrayObject *)PyArray_NewFromDescr(subtype,
                                                        dtype,
                                                        1,
                                                        &shape,
                                                        &stride,
                                                        NULL,
                                                        0,
                                                        NULL);
        if (ret == NULL) {
            return NULL;
        }
    }

    /*
     * Create a view which slides through ret for assigning the
     * successive input arrays.
     */
    sliding_view = (PyArrayObject_fields *)PyArray_View(ret,
                                                        NULL, &PyArray_Type);
    if (sliding_view == NULL) {
        Py_DECREF(ret);
        return NULL;
    }

    for (iarrays = 0; iarrays < narrays; ++iarrays) {
        /* Adjust the window dimensions for this array */
        sliding_view->dimensions[0] = PyArray_SIZE(arrays[iarrays]);

        /* Copy the data for this array */
        if (PyArray_CopyAsFlat((PyArrayObject *)sliding_view, arrays[iarrays],
                            order) < 0) {
            Py_DECREF(sliding_view);
            Py_DECREF(ret);
            return NULL;
        }

        /* Slide to the start of the next window */
        sliding_view->data +=
            sliding_view->strides[0] * PyArray_SIZE(arrays[iarrays]);
    }

    Py_DECREF(sliding_view);
    return ret;
}

NPY_NO_EXPORT PyObject *
PyArray_ConcatenateInto(PyObject *op, int axis, PyArrayObject *ret)
{
    int iarrays, narrays;
    PyArrayObject **arrays;

    if (!PySequence_Check(op)) {
        PyErr_SetString(PyExc_TypeError,
                        "The first input argument needs to be a sequence");
        return NULL;
    }

    /* Convert the input list into arrays */
    narrays = PySequence_Size(op);
    if (narrays < 0) {
        return NULL;
    }
    arrays = PyArray_malloc(narrays * sizeof(arrays[0]));
    if (arrays == NULL) {
        PyErr_NoMemory();
        return NULL;
    }
    for (iarrays = 0; iarrays < narrays; ++iarrays) {
        PyObject *item = PySequence_GetItem(op, iarrays);
        if (item == NULL) {
            narrays = iarrays;
            goto fail;
        }
        arrays[iarrays] = (PyArrayObject *)PyArray_FROM_O(item);
        Py_DECREF(item);
        if (arrays[iarrays] == NULL) {
            narrays = iarrays;
            goto fail;
        }
    }

    if (axis >= NPY_MAXDIMS) {
        ret = PyArray_ConcatenateFlattenedArrays(narrays, arrays, NPY_CORDER, ret);
    }
    else {
        ret = PyArray_ConcatenateArrays(narrays, arrays, axis, ret);
    }

    for (iarrays = 0; iarrays < narrays; ++iarrays) {
        Py_DECREF(arrays[iarrays]);
    }
    PyArray_free(arrays);

    return (PyObject *)ret;

fail:
    /* 'narrays' was set to how far we got in the conversion */
    for (iarrays = 0; iarrays < narrays; ++iarrays) {
        Py_DECREF(arrays[iarrays]);
    }
    PyArray_free(arrays);

    return NULL;
}

/*NUMPY_API
 * Concatenate
 *
 * Concatenate an arbitrary Python sequence into an array.
 * op is a python object supporting the sequence interface.
 * Its elements will be concatenated together to form a single
 * multidimensional array. If axis is NPY_MAXDIMS or bigger, then
 * each sequence object will be flattened before concatenation
*/
NPY_NO_EXPORT PyObject *
PyArray_Concatenate(PyObject *op, int axis)
{
    return PyArray_ConcatenateInto(op, axis, NULL);
}

static int
_signbit_set(PyArrayObject *arr)
{
    static char bitmask = (char) 0x80;
    char *ptr;  /* points to the npy_byte to test */
    char byteorder;
    int elsize;

    elsize = PyArray_DESCR(arr)->elsize;
    byteorder = PyArray_DESCR(arr)->byteorder;
    ptr = PyArray_DATA(arr);
    if (elsize > 1 &&
        (byteorder == NPY_LITTLE ||
         (byteorder == NPY_NATIVE &&
          PyArray_ISNBO(NPY_LITTLE)))) {
        ptr += elsize - 1;
    }
    return ((*ptr & bitmask) != 0);
}


/*NUMPY_API
 * ScalarKind
 *
 * Returns the scalar kind of a type number, with an
 * optional tweak based on the scalar value itself.
 * If no scalar is provided, it returns INTPOS_SCALAR
 * for both signed and unsigned integers, otherwise
 * it checks the sign of any signed integer to choose
 * INTNEG_SCALAR when appropriate.
 */
NPY_NO_EXPORT NPY_SCALARKIND
PyArray_ScalarKind(int typenum, PyArrayObject **arr)
{
    NPY_SCALARKIND ret = NPY_NOSCALAR;

    if ((unsigned int)typenum < NPY_NTYPES) {
        ret = _npy_scalar_kinds_table[typenum];
        /* Signed integer types are INTNEG in the table */
        if (ret == NPY_INTNEG_SCALAR) {
            if (!arr || !_signbit_set(*arr)) {
                ret = NPY_INTPOS_SCALAR;
            }
        }
    } else if (PyTypeNum_ISUSERDEF(typenum)) {
        PyArray_Descr* descr = PyArray_DescrFromType(typenum);

        if (descr->f->scalarkind) {
            ret = descr->f->scalarkind((arr ? *arr : NULL));
        }
        Py_DECREF(descr);
    }

    return ret;
}

/*NUMPY_API
 *
 * Determines whether the data type 'thistype', with
 * scalar kind 'scalar', can be coerced into 'neededtype'.
 */
NPY_NO_EXPORT int
PyArray_CanCoerceScalar(int thistype, int neededtype,
                        NPY_SCALARKIND scalar)
{
    PyArray_Descr* from;
    int *castlist;

    /* If 'thistype' is not a scalar, it must be safely castable */
    if (scalar == NPY_NOSCALAR) {
        return PyArray_CanCastSafely(thistype, neededtype);
    }
    if ((unsigned int)neededtype < NPY_NTYPES) {
        NPY_SCALARKIND neededscalar;

        if (scalar == NPY_OBJECT_SCALAR) {
            return PyArray_CanCastSafely(thistype, neededtype);
        }

        /*
         * The lookup table gives us exactly what we need for
         * this comparison, which PyArray_ScalarKind would not.
         *
         * The rule is that positive scalars can be coerced
         * to a signed ints, but negative scalars cannot be coerced
         * to unsigned ints.
         *   _npy_scalar_kinds_table[int]==NEGINT > POSINT,
         *      so 1 is returned, but
         *   _npy_scalar_kinds_table[uint]==POSINT < NEGINT,
         *      so 0 is returned, as required.
         *
         */
        neededscalar = _npy_scalar_kinds_table[neededtype];
        if (neededscalar >= scalar) {
            return 1;
        }
        if (!PyTypeNum_ISUSERDEF(thistype)) {
            return 0;
        }
    }

    from = PyArray_DescrFromType(thistype);
    if (from->f->cancastscalarkindto
        && (castlist = from->f->cancastscalarkindto[scalar])) {
        while (*castlist != NPY_NOTYPE) {
            if (*castlist++ == neededtype) {
                Py_DECREF(from);
                return 1;
            }
        }
    }
    Py_DECREF(from);

    return 0;
}

/* Could perhaps be redone to not make contiguous arrays */

/*NUMPY_API
 * Numeric.innerproduct(a,v)
 */
NPY_NO_EXPORT PyObject *
PyArray_InnerProduct(PyObject *op1, PyObject *op2)
{
    PyArrayObject *ap1 = NULL;
    PyArrayObject *ap2 = NULL;
    int typenum;
    PyArray_Descr *typec = NULL;
    PyObject* ap2t = NULL;
    npy_intp dims[NPY_MAXDIMS];
    PyArray_Dims newaxes = {dims, 0};
    int i;
    PyObject* ret = NULL;

    typenum = PyArray_ObjectType(op1, 0);
    typenum = PyArray_ObjectType(op2, typenum);
    typec = PyArray_DescrFromType(typenum);
    if (typec == NULL) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_TypeError,
                            "Cannot find a common data type.");
        }
        goto fail;
    }

    Py_INCREF(typec);
    ap1 = (PyArrayObject *)PyArray_FromAny(op1, typec, 0, 0,
                                           NPY_ARRAY_ALIGNED, NULL);
    if (ap1 == NULL) {
        Py_DECREF(typec);
        goto fail;
    }
    ap2 = (PyArrayObject *)PyArray_FromAny(op2, typec, 0, 0,
                                           NPY_ARRAY_ALIGNED, NULL);
    if (ap2 == NULL) {
        goto fail;
    }

    newaxes.len = PyArray_NDIM(ap2);
    if ((PyArray_NDIM(ap1) >= 1) && (newaxes.len >= 2)) {
        for (i = 0; i < newaxes.len - 2; i++) {
            dims[i] = (npy_intp)i;
        }
        dims[newaxes.len - 2] = newaxes.len - 1;
        dims[newaxes.len - 1] = newaxes.len - 2;

        ap2t = PyArray_Transpose(ap2, &newaxes);
        if (ap2t == NULL) {
            goto fail;
        }
    }
    else {
        ap2t = (PyObject *)ap2;
        Py_INCREF(ap2);
    }

    ret = PyArray_MatrixProduct2((PyObject *)ap1, ap2t, NULL);
    if (ret == NULL) {
        goto fail;
    }


    Py_DECREF(ap1);
    Py_DECREF(ap2);
    Py_DECREF(ap2t);
    return ret;

fail:
    Py_XDECREF(ap1);
    Py_XDECREF(ap2);
    Py_XDECREF(ap2t);
    Py_XDECREF(ret);
    return NULL;
}

/*NUMPY_API
 * Numeric.matrixproduct(a,v)
 * just like inner product but does the swapaxes stuff on the fly
 */
NPY_NO_EXPORT PyObject *
PyArray_MatrixProduct(PyObject *op1, PyObject *op2)
{
    return PyArray_MatrixProduct2(op1, op2, NULL);
}

/*NUMPY_API
 * Numeric.matrixproduct2(a,v,out)
 * just like inner product but does the swapaxes stuff on the fly
 */
NPY_NO_EXPORT PyObject *
PyArray_MatrixProduct2(PyObject *op1, PyObject *op2, PyArrayObject* out)
{
    PyArrayObject *ap1, *ap2, *out_buf = NULL, *result = NULL;
    PyArrayIterObject *it1, *it2;
    npy_intp i, j, l;
    int typenum, nd, axis, matchDim;
    npy_intp is1, is2, os;
    char *op;
    npy_intp dimensions[NPY_MAXDIMS];
    PyArray_DotFunc *dot;
    PyArray_Descr *typec = NULL;
    NPY_BEGIN_THREADS_DEF;

    typenum = PyArray_ObjectType(op1, 0);
    typenum = PyArray_ObjectType(op2, typenum);
    typec = PyArray_DescrFromType(typenum);
    if (typec == NULL) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_TypeError,
                            "Cannot find a common data type.");
        }
        return NULL;
    }

    Py_INCREF(typec);
    ap1 = (PyArrayObject *)PyArray_FromAny(op1, typec, 0, 0,
                                        NPY_ARRAY_ALIGNED, NULL);
    if (ap1 == NULL) {
        Py_DECREF(typec);
        return NULL;
    }
    ap2 = (PyArrayObject *)PyArray_FromAny(op2, typec, 0, 0,
                                        NPY_ARRAY_ALIGNED, NULL);
    if (ap2 == NULL) {
        Py_DECREF(ap1);
        return NULL;
    }

#if defined(HAVE_CBLAS)
    if (PyArray_NDIM(ap1) <= 2 && PyArray_NDIM(ap2) <= 2 &&
            (NPY_DOUBLE == typenum || NPY_CDOUBLE == typenum ||
             NPY_FLOAT == typenum || NPY_CFLOAT == typenum)) {
        return cblas_matrixproduct(typenum, ap1, ap2, out);
    }
#endif

    if (PyArray_NDIM(ap1) == 0 || PyArray_NDIM(ap2) == 0) {
        result = (PyArray_NDIM(ap1) == 0 ? ap1 : ap2);
        result = (PyArrayObject *)Py_TYPE(result)->tp_as_number->nb_multiply(
                                        (PyObject *)ap1, (PyObject *)ap2);
        Py_DECREF(ap1);
        Py_DECREF(ap2);
        return (PyObject *)result;
    }
    l = PyArray_DIMS(ap1)[PyArray_NDIM(ap1) - 1];
    if (PyArray_NDIM(ap2) > 1) {
        matchDim = PyArray_NDIM(ap2) - 2;
    }
    else {
        matchDim = 0;
    }
    if (PyArray_DIMS(ap2)[matchDim] != l) {
        dot_alignment_error(ap1, PyArray_NDIM(ap1) - 1, ap2, matchDim);
        goto fail;
    }
    nd = PyArray_NDIM(ap1) + PyArray_NDIM(ap2) - 2;
    if (nd > NPY_MAXDIMS) {
        PyErr_SetString(PyExc_ValueError, "dot: too many dimensions in result");
        goto fail;
    }
    j = 0;
    for (i = 0; i < PyArray_NDIM(ap1) - 1; i++) {
        dimensions[j++] = PyArray_DIMS(ap1)[i];
    }
    for (i = 0; i < PyArray_NDIM(ap2) - 2; i++) {
        dimensions[j++] = PyArray_DIMS(ap2)[i];
    }
    if (PyArray_NDIM(ap2) > 1) {
        dimensions[j++] = PyArray_DIMS(ap2)[PyArray_NDIM(ap2)-1];
    }

    is1 = PyArray_STRIDES(ap1)[PyArray_NDIM(ap1)-1];
    is2 = PyArray_STRIDES(ap2)[matchDim];
    /* Choose which subtype to return */
    out_buf = new_array_for_sum(ap1, ap2, out, nd, dimensions, typenum, &result);
    if (out_buf == NULL) {
        goto fail;
    }
    /* Ensure that multiarray.dot(<Nx0>,<0xM>) -> zeros((N,M)) */
    if (PyArray_SIZE(ap1) == 0 && PyArray_SIZE(ap2) == 0) {
        memset(PyArray_DATA(out_buf), 0, PyArray_NBYTES(out_buf));
    }

    dot = PyArray_DESCR(out_buf)->f->dotfunc;
    if (dot == NULL) {
        PyErr_SetString(PyExc_ValueError,
                        "dot not available for this type");
        goto fail;
    }

    op = PyArray_DATA(out_buf);
    os = PyArray_DESCR(out_buf)->elsize;
    axis = PyArray_NDIM(ap1)-1;
    it1 = (PyArrayIterObject *)
        PyArray_IterAllButAxis((PyObject *)ap1, &axis);
    if (it1 == NULL) {
        goto fail;
    }
    it2 = (PyArrayIterObject *)
        PyArray_IterAllButAxis((PyObject *)ap2, &matchDim);
    if (it2 == NULL) {
        Py_DECREF(it1);
        goto fail;
    }
    NPY_BEGIN_THREADS_DESCR(PyArray_DESCR(ap2));
    while (it1->index < it1->size) {
        while (it2->index < it2->size) {
            dot(it1->dataptr, is1, it2->dataptr, is2, op, l, NULL);
            op += os;
            PyArray_ITER_NEXT(it2);
        }
        PyArray_ITER_NEXT(it1);
        PyArray_ITER_RESET(it2);
    }
    NPY_END_THREADS_DESCR(PyArray_DESCR(ap2));
    Py_DECREF(it1);
    Py_DECREF(it2);
    if (PyErr_Occurred()) {
        /* only for OBJECT arrays */
        goto fail;
    }
    Py_DECREF(ap1);
    Py_DECREF(ap2);

    /* Trigger possible copy-back into `result` */
    PyArray_ResolveWritebackIfCopy(out_buf);
    Py_DECREF(out_buf);

    return (PyObject *)result;

fail:
    Py_XDECREF(ap1);
    Py_XDECREF(ap2);
    Py_XDECREF(out_buf);
    Py_XDECREF(result);
    return NULL;
}


/*NUMPY_API
 * Copy and Transpose
 *
 * Could deprecate this function, as there isn't a speed benefit over
 * calling Transpose and then Copy.
 */
NPY_NO_EXPORT PyObject *
PyArray_CopyAndTranspose(PyObject *op)
{
    PyArrayObject *arr, *tmp, *ret;
    int i;
    npy_intp new_axes_values[NPY_MAXDIMS];
    PyArray_Dims new_axes;

    /* Make sure we have an array */
    arr = (PyArrayObject *)PyArray_FROM_O(op);
    if (arr == NULL) {
        return NULL;
    }

    if (PyArray_NDIM(arr) > 1) {
        /* Set up the transpose operation */
        new_axes.len = PyArray_NDIM(arr);
        for (i = 0; i < new_axes.len; ++i) {
            new_axes_values[i] = new_axes.len - i - 1;
        }
        new_axes.ptr = new_axes_values;

        /* Do the transpose (always returns a view) */
        tmp = (PyArrayObject *)PyArray_Transpose(arr, &new_axes);
        if (tmp == NULL) {
            Py_DECREF(arr);
            return NULL;
        }
    }
    else {
        tmp = arr;
        arr = NULL;
    }

    /* TODO: Change this to NPY_KEEPORDER for NumPy 2.0 */
    ret = (PyArrayObject *)PyArray_NewCopy(tmp, NPY_CORDER);

    Py_XDECREF(arr);
    Py_DECREF(tmp);
    return (PyObject *)ret;
}

/*
 * Implementation which is common between PyArray_Correlate
 * and PyArray_Correlate2.
 *
 * inverted is set to 1 if computed correlate(ap2, ap1), 0 otherwise
 */
static PyArrayObject*
_pyarray_correlate(PyArrayObject *ap1, PyArrayObject *ap2, int typenum,
                   int mode, int *inverted)
{
    PyArrayObject *ret;
    npy_intp length;
    npy_intp i, n1, n2, n, n_left, n_right;
    npy_intp is1, is2, os;
    char *ip1, *ip2, *op;
    PyArray_DotFunc *dot;

    NPY_BEGIN_THREADS_DEF;

    n1 = PyArray_DIMS(ap1)[0];
    n2 = PyArray_DIMS(ap2)[0];
    if (n1 == 0) {
        PyErr_SetString(PyExc_ValueError, "first array argument cannot be empty");
        return NULL;
    }
    if (n2 == 0) {
        PyErr_SetString(PyExc_ValueError, "second array argument cannot be empty");
        return NULL;
    }
    if (n1 < n2) {
        ret = ap1;
        ap1 = ap2;
        ap2 = ret;
        ret = NULL;
        i = n1;
        n1 = n2;
        n2 = i;
        *inverted = 1;
    } else {
        *inverted = 0;
    }

    length = n1;
    n = n2;
    switch(mode) {
    case 0:
        length = length - n + 1;
        n_left = n_right = 0;
        break;
    case 1:
        n_left = (npy_intp)(n/2);
        n_right = n - n_left - 1;
        break;
    case 2:
        n_right = n - 1;
        n_left = n - 1;
        length = length + n - 1;
        break;
    default:
        PyErr_SetString(PyExc_ValueError, "mode must be 0, 1, or 2");
        return NULL;
    }

    /*
     * Need to choose an output array that can hold a sum
     * -- use priority to determine which subtype.
     */
    ret = new_array_for_sum(ap1, ap2, NULL, 1, &length, typenum, NULL);
    if (ret == NULL) {
        return NULL;
    }
    dot = PyArray_DESCR(ret)->f->dotfunc;
    if (dot == NULL) {
        PyErr_SetString(PyExc_ValueError,
                        "function not available for this data type");
        goto clean_ret;
    }

    NPY_BEGIN_THREADS_DESCR(PyArray_DESCR(ret));
    is1 = PyArray_STRIDES(ap1)[0];
    is2 = PyArray_STRIDES(ap2)[0];
    op = PyArray_DATA(ret);
    os = PyArray_DESCR(ret)->elsize;
    ip1 = PyArray_DATA(ap1);
    ip2 = PyArray_BYTES(ap2) + n_left*is2;
    n = n - n_left;
    for (i = 0; i < n_left; i++) {
        dot(ip1, is1, ip2, is2, op, n, ret);
        n++;
        ip2 -= is2;
        op += os;
    }
    if (small_correlate(ip1, is1, n1 - n2 + 1, PyArray_TYPE(ap1),
                        ip2, is2, n, PyArray_TYPE(ap2),
                        op, os)) {
        ip1 += is1 * (n1 - n2 + 1);
        op += os * (n1 - n2 + 1);
    }
    else {
        for (i = 0; i < (n1 - n2 + 1); i++) {
            dot(ip1, is1, ip2, is2, op, n, ret);
            ip1 += is1;
            op += os;
        }
    }
    for (i = 0; i < n_right; i++) {
        n--;
        dot(ip1, is1, ip2, is2, op, n, ret);
        ip1 += is1;
        op += os;
    }

    NPY_END_THREADS_DESCR(PyArray_DESCR(ret));
    if (PyErr_Occurred()) {
        goto clean_ret;
    }

    return ret;

clean_ret:
    Py_DECREF(ret);
    return NULL;
}

/*
 * Revert a one dimensional array in-place
 *
 * Return 0 on success, other value on failure
 */
static int
_pyarray_revert(PyArrayObject *ret)
{
    npy_intp length = PyArray_DIM(ret, 0);
    npy_intp os = PyArray_DESCR(ret)->elsize;
    char *op = PyArray_DATA(ret);
    char *sw1 = op;
    char *sw2;

    if (PyArray_ISNUMBER(ret) && !PyArray_ISCOMPLEX(ret)) {
        /* Optimization for unstructured dtypes */
        PyArray_CopySwapNFunc *copyswapn = PyArray_DESCR(ret)->f->copyswapn;
        sw2 = op + length * os - 1;
        /* First reverse the whole array byte by byte... */
        while(sw1 < sw2) {
            const char tmp = *sw1;
            *sw1++ = *sw2;
            *sw2-- = tmp;
        }
        /* ...then swap in place every item */
        copyswapn(op, os, NULL, 0, length, 1, NULL);
    }
    else {
        char *tmp = PyArray_malloc(PyArray_DESCR(ret)->elsize);
        if (tmp == NULL) {
            PyErr_NoMemory();
            return -1;
        }
        sw2 = op + (length - 1) * os;
        while (sw1 < sw2) {
            memcpy(tmp, sw1, os);
            memcpy(sw1, sw2, os);
            memcpy(sw2, tmp, os);
            sw1 += os;
            sw2 -= os;
        }
        PyArray_free(tmp);
    }

    return 0;
}

/*NUMPY_API
 * correlate(a1,a2,mode)
 *
 * This function computes the usual correlation (correlate(a1, a2) !=
 * correlate(a2, a1), and conjugate the second argument for complex inputs
 */
NPY_NO_EXPORT PyObject *
PyArray_Correlate2(PyObject *op1, PyObject *op2, int mode)
{
    PyArrayObject *ap1, *ap2, *ret = NULL;
    int typenum;
    PyArray_Descr *typec;
    int inverted;
    int st;

    typenum = PyArray_ObjectType(op1, 0);
    typenum = PyArray_ObjectType(op2, typenum);

    typec = PyArray_DescrFromType(typenum);
    Py_INCREF(typec);
    ap1 = (PyArrayObject *)PyArray_FromAny(op1, typec, 1, 1,
                                        NPY_ARRAY_DEFAULT, NULL);
    if (ap1 == NULL) {
        Py_DECREF(typec);
        return NULL;
    }
    ap2 = (PyArrayObject *)PyArray_FromAny(op2, typec, 1, 1,
                                        NPY_ARRAY_DEFAULT, NULL);
    if (ap2 == NULL) {
        goto clean_ap1;
    }

    if (PyArray_ISCOMPLEX(ap2)) {
        PyArrayObject *cap2;
        cap2 = (PyArrayObject *)PyArray_Conjugate(ap2, NULL);
        if (cap2 == NULL) {
            goto clean_ap2;
        }
        Py_DECREF(ap2);
        ap2 = cap2;
    }

    ret = _pyarray_correlate(ap1, ap2, typenum, mode, &inverted);
    if (ret == NULL) {
        goto clean_ap2;
    }

    /*
     * If we inverted input orders, we need to reverse the output array (i.e.
     * ret = ret[::-1])
     */
    if (inverted) {
        st = _pyarray_revert(ret);
        if (st) {
            goto clean_ret;
        }
    }

    Py_DECREF(ap1);
    Py_DECREF(ap2);
    return (PyObject *)ret;

clean_ret:
    Py_DECREF(ret);
clean_ap2:
    Py_DECREF(ap2);
clean_ap1:
    Py_DECREF(ap1);
    return NULL;
}

/*NUMPY_API
 * Numeric.correlate(a1,a2,mode)
 */
NPY_NO_EXPORT PyObject *
PyArray_Correlate(PyObject *op1, PyObject *op2, int mode)
{
    PyArrayObject *ap1, *ap2, *ret = NULL;
    int typenum;
    int unused;
    PyArray_Descr *typec;

    typenum = PyArray_ObjectType(op1, 0);
    typenum = PyArray_ObjectType(op2, typenum);

    typec = PyArray_DescrFromType(typenum);
    Py_INCREF(typec);
    ap1 = (PyArrayObject *)PyArray_FromAny(op1, typec, 1, 1,
                                            NPY_ARRAY_DEFAULT, NULL);
    if (ap1 == NULL) {
        Py_DECREF(typec);
        return NULL;
    }
    ap2 = (PyArrayObject *)PyArray_FromAny(op2, typec, 1, 1,
                                           NPY_ARRAY_DEFAULT, NULL);
    if (ap2 == NULL) {
        goto fail;
    }

    ret = _pyarray_correlate(ap1, ap2, typenum, mode, &unused);
    if (ret == NULL) {
        goto fail;
    }
    Py_DECREF(ap1);
    Py_DECREF(ap2);
    return (PyObject *)ret;

fail:
    Py_XDECREF(ap1);
    Py_XDECREF(ap2);
    Py_XDECREF(ret);
    return NULL;
}


static PyObject *
array_putmask(PyObject *NPY_UNUSED(module), PyObject *args, PyObject *kwds)
{
    PyObject *mask, *values;
    PyObject *array;

    static char *kwlist[] = {"arr", "mask", "values", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!OO:putmask", kwlist,
                &PyArray_Type, &array, &mask, &values)) {
        return NULL;
    }
    return PyArray_PutMask((PyArrayObject *)array, values, mask);
}

/*
 * Compare the field dictionaries for two types.
 *
 * Return 1 if the field types and field names of the two descrs are equal and
 * in the same order, 0 if not.
 */
static int
_equivalent_fields(PyArray_Descr *type1, PyArray_Descr *type2) {

    int val;

    if (type1->fields == type2->fields && type1->names == type2->names) {
        return 1;
    }
    if (type1->fields == NULL || type2->fields == NULL) {
        return 0;
    }

    val = PyObject_RichCompareBool(type1->fields, type2->fields, Py_EQ);
    if (val != 1 || PyErr_Occurred()) {
        PyErr_Clear();
        return 0;
    }

    val = PyObject_RichCompareBool(type1->names, type2->names, Py_EQ);
    if (val != 1 || PyErr_Occurred()) {
        PyErr_Clear();
        return 0;
    }

    return 1;
}

/*
 * Compare the subarray data for two types.
 * Return 1 if they are the same, 0 if not.
 */
static int
_equivalent_subarrays(PyArray_ArrayDescr *sub1, PyArray_ArrayDescr *sub2)
{
    int val;

    if (sub1 == sub2) {
        return 1;

    }
    if (sub1 == NULL || sub2 == NULL) {
        return 0;
    }

    val = PyObject_RichCompareBool(sub1->shape, sub2->shape, Py_EQ);
    if (val != 1 || PyErr_Occurred()) {
        PyErr_Clear();
        return 0;
    }

    return PyArray_EquivTypes(sub1->base, sub2->base);
}


/*NUMPY_API
 *
 * This function returns true if the two typecodes are
 * equivalent (same basic kind and same itemsize).
 */
NPY_NO_EXPORT unsigned char
PyArray_EquivTypes(PyArray_Descr *type1, PyArray_Descr *type2)
{
    int type_num1, type_num2, size1, size2;

    if (type1 == type2) {
        return NPY_TRUE;
    }

    type_num1 = type1->type_num;
    type_num2 = type2->type_num;
    size1 = type1->elsize;
    size2 = type2->elsize;

    if (size1 != size2) {
        return NPY_FALSE;
    }
    if (PyArray_ISNBO(type1->byteorder) != PyArray_ISNBO(type2->byteorder)) {
        return NPY_FALSE;
    }
    if (type1->subarray || type2->subarray) {
        return ((type_num1 == type_num2)
                && _equivalent_subarrays(type1->subarray, type2->subarray));
    }
    if (type_num1 == NPY_VOID || type_num2 == NPY_VOID) {
        return ((type_num1 == type_num2) && _equivalent_fields(type1, type2));
    }
    if (type_num1 == NPY_DATETIME
            || type_num1 == NPY_TIMEDELTA
            || type_num2 == NPY_DATETIME
            || type_num2 == NPY_TIMEDELTA) {
        return ((type_num1 == type_num2)
                && has_equivalent_datetime_metadata(type1, type2));
    }
    return type1->kind == type2->kind;
}

/*NUMPY_API*/
NPY_NO_EXPORT unsigned char
PyArray_EquivTypenums(int typenum1, int typenum2)
{
    PyArray_Descr *d1, *d2;
    npy_bool ret;

    if (typenum1 == typenum2) {
        return NPY_SUCCEED;
    }

    d1 = PyArray_DescrFromType(typenum1);
    d2 = PyArray_DescrFromType(typenum2);
    ret = PyArray_EquivTypes(d1, d2);
    Py_DECREF(d1);
    Py_DECREF(d2);
    return ret;
}

/*** END C-API FUNCTIONS **/
/*
 * NPY_RELAXED_STRIDES_CHECKING: If the strides logic is changed, the
 * order specific stride setting is not necessary.
 */
static NPY_STEALS_REF_TO_ARG(1) PyObject *
_prepend_ones(PyArrayObject *arr, int nd, int ndmin, NPY_ORDER order)
{
    npy_intp newdims[NPY_MAXDIMS];
    npy_intp newstrides[NPY_MAXDIMS];
    npy_intp newstride;
    int i, k, num;
    PyObject *ret;
    PyArray_Descr *dtype;

    if (order == NPY_FORTRANORDER || PyArray_ISFORTRAN(arr) || PyArray_NDIM(arr) == 0) {
        newstride = PyArray_DESCR(arr)->elsize;
    }
    else {
        newstride = PyArray_STRIDES(arr)[0] * PyArray_DIMS(arr)[0];
    }

    num = ndmin - nd;
    for (i = 0; i < num; i++) {
        newdims[i] = 1;
        newstrides[i] = newstride;
    }
    for (i = num; i < ndmin; i++) {
        k = i - num;
        newdims[i] = PyArray_DIMS(arr)[k];
        newstrides[i] = PyArray_STRIDES(arr)[k];
    }
    dtype = PyArray_DESCR(arr);
    Py_INCREF(dtype);
    ret = PyArray_NewFromDescrAndBase(
            Py_TYPE(arr), dtype,
            ndmin, newdims, newstrides, PyArray_DATA(arr),
            PyArray_FLAGS(arr), (PyObject *)arr, (PyObject *)arr);
    Py_DECREF(arr);

    return ret;
}

#define STRIDING_OK(op, order) \
                ((order) == NPY_ANYORDER || \
                 (order) == NPY_KEEPORDER || \
                 ((order) == NPY_CORDER && PyArray_IS_C_CONTIGUOUS(op)) || \
                 ((order) == NPY_FORTRANORDER && PyArray_IS_F_CONTIGUOUS(op)))

static PyObject *
_array_fromobject(PyObject *NPY_UNUSED(ignored), PyObject *args, PyObject *kws)
{
    PyObject *op;
    PyArrayObject *oparr = NULL, *ret = NULL;
    npy_bool subok = NPY_FALSE;
    npy_bool copy = NPY_TRUE;
    int ndmin = 0, nd;
    PyArray_Descr *type = NULL;
    PyArray_Descr *oldtype = NULL;
    NPY_ORDER order = NPY_KEEPORDER;
    int flags = 0;

    static char *kwd[]= {"object", "dtype", "copy", "order", "subok",
                         "ndmin", NULL};

    if (PyTuple_GET_SIZE(args) > 2) {
        PyErr_Format(PyExc_TypeError,
                     "array() takes from 1 to 2 positional arguments but "
                     "%zd were given", PyTuple_GET_SIZE(args));
        return NULL;
    }

    /* super-fast path for ndarray argument calls */
    if (PyTuple_GET_SIZE(args) == 0) {
        goto full_path;
    }
    op = PyTuple_GET_ITEM(args, 0);
    if (PyArray_CheckExact(op)) {
        PyObject * dtype_obj = Py_None;
        oparr = (PyArrayObject *)op;
        /* get dtype which can be positional */
        if (PyTuple_GET_SIZE(args) == 2) {
            dtype_obj = PyTuple_GET_ITEM(args, 1);
        }
        else if (kws) {
            dtype_obj = PyDict_GetItemWithError(kws, npy_ma_str_dtype);
            if (dtype_obj == NULL && PyErr_Occurred()) {
                return NULL;
            }
            if (dtype_obj == NULL) {
                dtype_obj = Py_None;
            }
        }
        if (dtype_obj != Py_None) {
            goto full_path;
        }

        /* array(ndarray) */
        if (kws == NULL) {
            ret = (PyArrayObject *)PyArray_NewCopy(oparr, order);
            goto finish;
        }
        else {
            /* fast path for copy=False rest default (np.asarray) */
            PyObject * copy_obj, * order_obj, *ndmin_obj;
            copy_obj = PyDict_GetItemWithError(kws, npy_ma_str_copy);
            if (copy_obj == NULL && PyErr_Occurred()) {
                return NULL;
            }
            if (copy_obj != Py_False) {
                goto full_path;
            }
            copy = NPY_FALSE;

            /* order does not matter for contiguous 1d arrays */
            if (PyArray_NDIM((PyArrayObject*)op) > 1 ||
                !PyArray_IS_C_CONTIGUOUS((PyArrayObject*)op)) {
                order_obj = PyDict_GetItemWithError(kws, npy_ma_str_order);
                if (order_obj == NULL && PyErr_Occurred()) {
                    return NULL;
                }
                else if (order_obj != Py_None && order_obj != NULL) {
                    goto full_path;
                }
            }

            ndmin_obj = PyDict_GetItemWithError(kws, npy_ma_str_ndmin);
            if (ndmin_obj == NULL && PyErr_Occurred()) {
                return NULL;
            }
            else if (ndmin_obj) {
                long t = PyLong_AsLong(ndmin_obj);
                if (error_converting(t)) {
                    goto clean_type;
                }
                else if (t > NPY_MAXDIMS) {
                    goto full_path;
                }
                ndmin = t;
            }

            /* copy=False with default dtype, order (any is OK) and ndim */
            ret = oparr;
            Py_INCREF(ret);
            goto finish;
        }
    }

full_path:
    if (!PyArg_ParseTupleAndKeywords(args, kws, "O|O&O&O&O&i:array", kwd,
                &op,
                PyArray_DescrConverter2, &type,
                PyArray_BoolConverter, &copy,
                PyArray_OrderConverter, &order,
                PyArray_BoolConverter, &subok,
                &ndmin)) {
        goto clean_type;
    }

    if (ndmin > NPY_MAXDIMS) {
        PyErr_Format(PyExc_ValueError,
                "ndmin bigger than allowable number of dimensions "
                "NPY_MAXDIMS (=%d)", NPY_MAXDIMS);
        goto clean_type;
    }
    /* fast exit if simple call */
    if ((subok && PyArray_Check(op)) ||
        (!subok && PyArray_CheckExact(op))) {
        oparr = (PyArrayObject *)op;
        if (type == NULL) {
            if (!copy && STRIDING_OK(oparr, order)) {
                ret = oparr;
                Py_INCREF(ret);
                goto finish;
            }
            else {
                ret = (PyArrayObject *)PyArray_NewCopy(oparr, order);
                goto finish;
            }
        }
        /* One more chance */
        oldtype = PyArray_DESCR(oparr);
        if (PyArray_EquivTypes(oldtype, type)) {
            if (!copy && STRIDING_OK(oparr, order)) {
                Py_INCREF(op);
                ret = oparr;
                goto finish;
            }
            else {
                ret = (PyArrayObject *)PyArray_NewCopy(oparr, order);
                if (oldtype == type || ret == NULL) {
                    goto finish;
                }
                Py_INCREF(oldtype);
                Py_DECREF(PyArray_DESCR(ret));
                ((PyArrayObject_fields *)ret)->descr = oldtype;
                goto finish;
            }
        }
    }

    if (copy) {
        flags = NPY_ARRAY_ENSURECOPY;
    }
    if (order == NPY_CORDER) {
        flags |= NPY_ARRAY_C_CONTIGUOUS;
    }
    else if ((order == NPY_FORTRANORDER)
                 /* order == NPY_ANYORDER && */
                 || (PyArray_Check(op) &&
                     PyArray_ISFORTRAN((PyArrayObject *)op))) {
        flags |= NPY_ARRAY_F_CONTIGUOUS;
    }
    if (!subok) {
        flags |= NPY_ARRAY_ENSUREARRAY;
    }

    flags |= NPY_ARRAY_FORCECAST;
    Py_XINCREF(type);
    ret = (PyArrayObject *)PyArray_CheckFromAny(op, type,
                                                0, 0, flags, NULL);

 finish:
    Py_XDECREF(type);
    if (ret == NULL) {
        return NULL;
    }

    nd = PyArray_NDIM(ret);
    if (nd >= ndmin) {
        return (PyObject *)ret;
    }
    /*
     * create a new array from the same data with ones in the shape
     * steals a reference to ret
     */
    return _prepend_ones(ret, nd, ndmin, order);

clean_type:
    Py_XDECREF(type);
    return NULL;
}

static PyObject *
array_copyto(PyObject *NPY_UNUSED(ignored), PyObject *args, PyObject *kwds)
{

    static char *kwlist[] = {"dst", "src", "casting", "where", NULL};
    PyObject *wheremask_in = NULL;
    PyArrayObject *dst = NULL, *src = NULL, *wheremask = NULL;
    NPY_CASTING casting = NPY_SAME_KIND_CASTING;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!O&|O&O:copyto", kwlist,
                &PyArray_Type, &dst,
                &PyArray_Converter, &src,
                &PyArray_CastingConverter, &casting,
                &wheremask_in)) {
        goto fail;
    }

    if (wheremask_in != NULL) {
        /* Get the boolean where mask */
        PyArray_Descr *dtype = PyArray_DescrFromType(NPY_BOOL);
        if (dtype == NULL) {
            goto fail;
        }
        wheremask = (PyArrayObject *)PyArray_FromAny(wheremask_in,
                                        dtype, 0, 0, 0, NULL);
        if (wheremask == NULL) {
            goto fail;
        }
    }

    if (PyArray_AssignArray(dst, src, wheremask, casting) < 0) {
        goto fail;
    }

    Py_XDECREF(src);
    Py_XDECREF(wheremask);

    Py_RETURN_NONE;

fail:
    Py_XDECREF(src);
    Py_XDECREF(wheremask);
    return NULL;
}

static PyObject *
array_empty(PyObject *NPY_UNUSED(ignored), PyObject *args, PyObject *kwds)
{

    static char *kwlist[] = {"shape", "dtype", "order", NULL};
    PyArray_Descr *typecode = NULL;
    PyArray_Dims shape = {NULL, 0};
    NPY_ORDER order = NPY_CORDER;
    npy_bool is_f_order;
    PyArrayObject *ret = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|O&O&:empty", kwlist,
                PyArray_IntpConverter, &shape,
                PyArray_DescrConverter, &typecode,
                PyArray_OrderConverter, &order)) {
        goto fail;
    }

    switch (order) {
        case NPY_CORDER:
            is_f_order = NPY_FALSE;
            break;
        case NPY_FORTRANORDER:
            is_f_order = NPY_TRUE;
            break;
        default:
            PyErr_SetString(PyExc_ValueError,
                            "only 'C' or 'F' order is permitted");
            goto fail;
    }

    ret = (PyArrayObject *)PyArray_Empty(shape.len, shape.ptr,
                                            typecode, is_f_order);

    npy_free_cache_dim_obj(shape);
    return (PyObject *)ret;

fail:
    Py_XDECREF(typecode);
    npy_free_cache_dim_obj(shape);
    return NULL;
}

static PyObject *
array_empty_like(PyObject *NPY_UNUSED(ignored), PyObject *args, PyObject *kwds)
{

    static char *kwlist[] = {"prototype", "dtype", "order", "subok", "shape", NULL};
    PyArrayObject *prototype = NULL;
    PyArray_Descr *dtype = NULL;
    NPY_ORDER order = NPY_KEEPORDER;
    PyArrayObject *ret = NULL;
    int subok = 1;
    /* -1 is a special value meaning "not specified" */
    PyArray_Dims shape = {NULL, -1};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|O&O&iO&:empty_like", kwlist,
                &PyArray_Converter, &prototype,
                &PyArray_DescrConverter2, &dtype,
                &PyArray_OrderConverter, &order,
                &subok,
                &PyArray_OptionalIntpConverter, &shape)) {
        goto fail;
    }
    /* steals the reference to dtype if it's not NULL */
    ret = (PyArrayObject *)PyArray_NewLikeArrayWithShape(prototype, order, dtype,
                                                         shape.len, shape.ptr, subok);
    npy_free_cache_dim_obj(shape);
    if (!ret) {
        goto fail;
    }
    Py_DECREF(prototype);

    return (PyObject *)ret;

fail:
    Py_XDECREF(prototype);
    Py_XDECREF(dtype);
    return NULL;
}

/*
 * This function is needed for supporting Pickles of
 * numpy scalar objects.
 */
static PyObject *
array_scalar(PyObject *NPY_UNUSED(ignored), PyObject *args, PyObject *kwds)
{

    static char *kwlist[] = {"dtype", "obj", NULL};
    PyArray_Descr *typecode;
    PyObject *obj = NULL, *tmpobj = NULL;
    int alloc = 0;
    void *dptr;
    PyObject *ret;


    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|O:scalar", kwlist,
                &PyArrayDescr_Type, &typecode, &obj)) {
        return NULL;
    }
    if (PyDataType_FLAGCHK(typecode, NPY_LIST_PICKLE)) {
        if (!PySequence_Check(obj)) {
            PyErr_SetString(PyExc_TypeError,
                            "found non-sequence while unpickling scalar with "
                            "NPY_LIST_PICKLE set");
            return NULL;
        }
        dptr = &obj;
    }

    else if (PyDataType_FLAGCHK(typecode, NPY_ITEM_IS_POINTER)) {
        if (obj == NULL) {
            obj = Py_None;
        }
        dptr = &obj;
    }
    else {
        if (obj == NULL) {
            if (typecode->elsize == 0) {
                typecode->elsize = 1;
            }
            dptr = PyArray_malloc(typecode->elsize);
            if (dptr == NULL) {
                return PyErr_NoMemory();
            }
            memset(dptr, '\0', typecode->elsize);
            alloc = 1;
        }
        else {
            /* Backward compatibility with Python 2 NumPy pickles */
            if (PyUnicode_Check(obj)) {
                tmpobj = PyUnicode_AsLatin1String(obj);
                obj = tmpobj;
                if (tmpobj == NULL) {
                    /* More informative error message */
                    PyErr_SetString(PyExc_ValueError,
                            "Failed to encode Numpy scalar data string to "
                            "latin1,\npickle.load(a, encoding='latin1') is "
                            "assumed if unpickling.");
                    return NULL;
                }
            }
            if (!PyString_Check(obj)) {
                PyErr_SetString(PyExc_TypeError,
                        "initializing object must be a string");
                Py_XDECREF(tmpobj);
                return NULL;
            }
            if (PyString_GET_SIZE(obj) < typecode->elsize) {
                PyErr_SetString(PyExc_ValueError,
                        "initialization string is too small");
                Py_XDECREF(tmpobj);
                return NULL;
            }
            dptr = PyString_AS_STRING(obj);
        }
    }
    ret = PyArray_Scalar(dptr, typecode, NULL);

    /* free dptr which contains zeros */
    if (alloc) {
        PyArray_free(dptr);
    }
    Py_XDECREF(tmpobj);
    return ret;
}

static PyObject *
array_zeros(PyObject *NPY_UNUSED(ignored), PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"shape", "dtype", "order", NULL};
    PyArray_Descr *typecode = NULL;
    PyArray_Dims shape = {NULL, 0};
    NPY_ORDER order = NPY_CORDER;
    npy_bool is_f_order = NPY_FALSE;
    PyArrayObject *ret = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|O&O&:zeros", kwlist,
                PyArray_IntpConverter, &shape,
                PyArray_DescrConverter, &typecode,
                PyArray_OrderConverter, &order)) {
        goto fail;
    }

    switch (order) {
        case NPY_CORDER:
            is_f_order = NPY_FALSE;
            break;
        case NPY_FORTRANORDER:
            is_f_order = NPY_TRUE;
            break;
        default:
            PyErr_SetString(PyExc_ValueError,
                            "only 'C' or 'F' order is permitted");
            goto fail;
    }

    ret = (PyArrayObject *)PyArray_Zeros(shape.len, shape.ptr,
                                        typecode, (int) is_f_order);

    npy_free_cache_dim_obj(shape);
    return (PyObject *)ret;

fail:
    Py_XDECREF(typecode);
    npy_free_cache_dim_obj(shape);
    return (PyObject *)ret;
}

static PyObject *
array_count_nonzero(PyObject *NPY_UNUSED(self), PyObject *args, PyObject *kwds)
{
    PyArrayObject *array;
    npy_intp count;

    if (!PyArg_ParseTuple(args, "O&:count_nonzero", PyArray_Converter, &array)) {
        return NULL;
    }

    count =  PyArray_CountNonzero(array);

    Py_DECREF(array);

    if (count == -1) {
        return NULL;
    }
    return PyLong_FromSsize_t(count);
}

static PyObject *
array_fromstring(PyObject *NPY_UNUSED(ignored), PyObject *args, PyObject *keywds)
{
    char *data;
    Py_ssize_t nin = -1;
    char *sep = NULL;
    Py_ssize_t s;
    static char *kwlist[] = {"string", "dtype", "count", "sep", NULL};
    PyArray_Descr *descr = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, keywds,
                "s#|O&" NPY_SSIZE_T_PYFMT "s:fromstring", kwlist,
                &data, &s, PyArray_DescrConverter, &descr, &nin, &sep)) {
        Py_XDECREF(descr);
        return NULL;
    }

    /* binary mode, condition copied from PyArray_FromString */
    if (sep == NULL || strlen(sep) == 0) {
        /* Numpy 1.14, 2017-10-19 */
        if (DEPRECATE(
                "The binary mode of fromstring is deprecated, as it behaves "
                "surprisingly on unicode inputs. Use frombuffer instead") < 0) {
            Py_XDECREF(descr);
            return NULL;
        }
    }
    return PyArray_FromString(data, (npy_intp)s, descr, (npy_intp)nin, sep);
}



static PyObject *
array_fromfile(PyObject *NPY_UNUSED(ignored), PyObject *args, PyObject *keywds)
{
    PyObject *file = NULL, *ret = NULL;
    PyObject *err_type = NULL, *err_value = NULL, *err_traceback = NULL;
    char *sep = "";
    Py_ssize_t nin = -1;
    static char *kwlist[] = {"file", "dtype", "count", "sep", "offset", NULL};
    PyArray_Descr *type = NULL;
    int own;
    npy_off_t orig_pos = 0, offset = 0;
    FILE *fp;

    if (!PyArg_ParseTupleAndKeywords(args, keywds,
                "O|O&" NPY_SSIZE_T_PYFMT "s" NPY_OFF_T_PYFMT ":fromfile", kwlist,
                &file, PyArray_DescrConverter, &type, &nin, &sep, &offset)) {
        Py_XDECREF(type);
        return NULL;
    }

    file = NpyPath_PathlikeToFspath(file);
    if (file == NULL) {
        return NULL;
    }

    if (offset != 0 && strcmp(sep, "") != 0) {
        PyErr_SetString(PyExc_TypeError, "'offset' argument only permitted for binary files");
        Py_XDECREF(type);
        Py_DECREF(file);
        return NULL;
    }
    if (PyString_Check(file) || PyUnicode_Check(file)) {
        Py_SETREF(file, npy_PyFile_OpenFile(file, "rb"));
        if (file == NULL) {
            Py_XDECREF(type);
            return NULL;
        }
        own = 1;
    }
    else {
        own = 0;
    }
    fp = npy_PyFile_Dup2(file, "rb", &orig_pos);
    if (fp == NULL) {
        Py_DECREF(file);
        Py_XDECREF(type);
        return NULL;
    }
    if (npy_fseek(fp, offset, SEEK_CUR) != 0) {
        PyErr_SetFromErrno(PyExc_IOError);
        goto cleanup;
    }
    if (type == NULL) {
        type = PyArray_DescrFromType(NPY_DEFAULT_TYPE);
    }
    ret = PyArray_FromFile(fp, type, (npy_intp) nin, sep);

    /* If an exception is thrown in the call to PyArray_FromFile
     * we need to clear it, and restore it later to ensure that
     * we can cleanup the duplicated file descriptor properly.
     */
cleanup:
    PyErr_Fetch(&err_type, &err_value, &err_traceback);
    if (npy_PyFile_DupClose2(file, fp, orig_pos) < 0) {
        npy_PyErr_ChainExceptions(err_type, err_value, err_traceback);
        goto fail;
    }
    if (own && npy_PyFile_CloseFile(file) < 0) {
        npy_PyErr_ChainExceptions(err_type, err_value, err_traceback);
        goto fail;
    }
    PyErr_Restore(err_type, err_value, err_traceback);
    Py_DECREF(file);
    return ret;

fail:
    Py_DECREF(file);
    Py_XDECREF(ret);
    return NULL;
}

static PyObject *
array_fromiter(PyObject *NPY_UNUSED(ignored), PyObject *args, PyObject *keywds)
{
    PyObject *iter;
    Py_ssize_t nin = -1;
    static char *kwlist[] = {"iter", "dtype", "count", NULL};
    PyArray_Descr *descr = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, keywds,
                "OO&|" NPY_SSIZE_T_PYFMT ":fromiter", kwlist,
                &iter, PyArray_DescrConverter, &descr, &nin)) {
        Py_XDECREF(descr);
        return NULL;
    }
    return PyArray_FromIter(iter, descr, (npy_intp)nin);
}

static PyObject *
array_frombuffer(PyObject *NPY_UNUSED(ignored), PyObject *args, PyObject *keywds)
{
    PyObject *obj = NULL;
    Py_ssize_t nin = -1, offset = 0;
    static char *kwlist[] = {"buffer", "dtype", "count", "offset", NULL};
    PyArray_Descr *type = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, keywds,
                "O|O&" NPY_SSIZE_T_PYFMT NPY_SSIZE_T_PYFMT ":frombuffer", kwlist,
                &obj, PyArray_DescrConverter, &type, &nin, &offset)) {
        Py_XDECREF(type);
        return NULL;
    }
    if (type == NULL) {
        type = PyArray_DescrFromType(NPY_DEFAULT_TYPE);
    }
    return PyArray_FromBuffer(obj, type, (npy_intp)nin, (npy_intp)offset);
}

static PyObject *
array_concatenate(PyObject *NPY_UNUSED(dummy), PyObject *args, PyObject *kwds)
{
    PyObject *a0;
    PyObject *out = NULL;
    int axis = 0;
    static char *kwlist[] = {"seq", "axis", "out", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O&O:concatenate", kwlist,
                &a0, PyArray_AxisConverter, &axis, &out)) {
        return NULL;
    }
    if (out != NULL) {
        if (out == Py_None) {
            out = NULL;
        }
        else if (!PyArray_Check(out)) {
            PyErr_SetString(PyExc_TypeError, "'out' must be an array");
            return NULL;
        }
    }
    return PyArray_ConcatenateInto(a0, axis, (PyArrayObject *)out);
}

static PyObject *
array_innerproduct(PyObject *NPY_UNUSED(dummy), PyObject *args)
{
    PyObject *b0, *a0;

    if (!PyArg_ParseTuple(args, "OO:innerproduct", &a0, &b0)) {
        return NULL;
    }
    return PyArray_Return((PyArrayObject *)PyArray_InnerProduct(a0, b0));
}

static PyObject *
array_matrixproduct(PyObject *NPY_UNUSED(dummy), PyObject *args, PyObject* kwds)
{
    PyObject *v, *a, *o = NULL;
    PyArrayObject *ret;
    char* kwlist[] = {"a", "b", "out", NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|O:matrixproduct",
                                     kwlist, &a, &v, &o)) {
        return NULL;
    }
    if (o != NULL) {
        if (o == Py_None) {
            o = NULL;
        }
        else if (!PyArray_Check(o)) {
            PyErr_SetString(PyExc_TypeError, "'out' must be an array");
            return NULL;
        }
    }
    ret = (PyArrayObject *)PyArray_MatrixProduct2(a, v, (PyArrayObject *)o);
    return PyArray_Return(ret);
}


static PyObject *
array_vdot(PyObject *NPY_UNUSED(dummy), PyObject *args)
{
    int typenum;
    char *ip1, *ip2, *op;
    npy_intp n, stride1, stride2;
    PyObject *op1, *op2;
    npy_intp newdimptr[1] = {-1};
    PyArray_Dims newdims = {newdimptr, 1};
    PyArrayObject *ap1 = NULL, *ap2  = NULL, *ret = NULL;
    PyArray_Descr *type;
    PyArray_DotFunc *vdot;
    NPY_BEGIN_THREADS_DEF;

    if (!PyArg_ParseTuple(args, "OO:vdot", &op1, &op2)) {
        return NULL;
    }

    /*
     * Conjugating dot product using the BLAS for vectors.
     * Flattens both op1 and op2 before dotting.
     */
    typenum = PyArray_ObjectType(op1, 0);
    typenum = PyArray_ObjectType(op2, typenum);

    type = PyArray_DescrFromType(typenum);
    Py_INCREF(type);
    ap1 = (PyArrayObject *)PyArray_FromAny(op1, type, 0, 0, 0, NULL);
    if (ap1 == NULL) {
        Py_DECREF(type);
        goto fail;
    }

    op1 = PyArray_Newshape(ap1, &newdims, NPY_CORDER);
    if (op1 == NULL) {
        Py_DECREF(type);
        goto fail;
    }
    Py_DECREF(ap1);
    ap1 = (PyArrayObject *)op1;

    ap2 = (PyArrayObject *)PyArray_FromAny(op2, type, 0, 0, 0, NULL);
    if (ap2 == NULL) {
        goto fail;
    }
    op2 = PyArray_Newshape(ap2, &newdims, NPY_CORDER);
    if (op2 == NULL) {
        goto fail;
    }
    Py_DECREF(ap2);
    ap2 = (PyArrayObject *)op2;

    if (PyArray_DIM(ap2, 0) != PyArray_DIM(ap1, 0)) {
        PyErr_SetString(PyExc_ValueError,
                "vectors have different lengths");
        goto fail;
    }

    /* array scalar output */
    ret = new_array_for_sum(ap1, ap2, NULL, 0, (npy_intp *)NULL, typenum, NULL);
    if (ret == NULL) {
        goto fail;
    }

    n = PyArray_DIM(ap1, 0);
    stride1 = PyArray_STRIDE(ap1, 0);
    stride2 = PyArray_STRIDE(ap2, 0);
    ip1 = PyArray_DATA(ap1);
    ip2 = PyArray_DATA(ap2);
    op = PyArray_DATA(ret);

    switch (typenum) {
        case NPY_CFLOAT:
            vdot = (PyArray_DotFunc *)CFLOAT_vdot;
            break;
        case NPY_CDOUBLE:
            vdot = (PyArray_DotFunc *)CDOUBLE_vdot;
            break;
        case NPY_CLONGDOUBLE:
            vdot = (PyArray_DotFunc *)CLONGDOUBLE_vdot;
            break;
        case NPY_OBJECT:
            vdot = (PyArray_DotFunc *)OBJECT_vdot;
            break;
        default:
            vdot = type->f->dotfunc;
            if (vdot == NULL) {
                PyErr_SetString(PyExc_ValueError,
                        "function not available for this data type");
                goto fail;
            }
    }

    if (n < 500) {
        vdot(ip1, stride1, ip2, stride2, op, n, NULL);
    }
    else {
        NPY_BEGIN_THREADS_DESCR(type);
        vdot(ip1, stride1, ip2, stride2, op, n, NULL);
        NPY_END_THREADS_DESCR(type);
    }

    Py_XDECREF(ap1);
    Py_XDECREF(ap2);
    return PyArray_Return(ret);
fail:
    Py_XDECREF(ap1);
    Py_XDECREF(ap2);
    Py_XDECREF(ret);
    return NULL;
}

static int
einsum_sub_op_from_str(PyObject *args, PyObject **str_obj, char **subscripts,
                       PyArrayObject **op)
{
    int i, nop;
    PyObject *subscripts_str;

    nop = PyTuple_GET_SIZE(args) - 1;
    if (nop <= 0) {
        PyErr_SetString(PyExc_ValueError,
                        "must specify the einstein sum subscripts string "
                        "and at least one operand");
        return -1;
    }
    else if (nop >= NPY_MAXARGS) {
        PyErr_SetString(PyExc_ValueError, "too many operands");
        return -1;
    }

    /* Get the subscripts string */
    subscripts_str = PyTuple_GET_ITEM(args, 0);
    if (PyUnicode_Check(subscripts_str)) {
        *str_obj = PyUnicode_AsASCIIString(subscripts_str);
        if (*str_obj == NULL) {
            return -1;
        }
        subscripts_str = *str_obj;
    }

    *subscripts = PyBytes_AsString(subscripts_str);
    if (*subscripts == NULL) {
        Py_XDECREF(*str_obj);
        *str_obj = NULL;
        return -1;
    }

    /* Set the operands to NULL */
    for (i = 0; i < nop; ++i) {
        op[i] = NULL;
    }

    /* Get the operands */
    for (i = 0; i < nop; ++i) {
        PyObject *obj = PyTuple_GET_ITEM(args, i+1);

        op[i] = (PyArrayObject *)PyArray_FROM_OF(obj, NPY_ARRAY_ENSUREARRAY);
        if (op[i] == NULL) {
            goto fail;
        }
    }

    return nop;

fail:
    for (i = 0; i < nop; ++i) {
        Py_XDECREF(op[i]);
        op[i] = NULL;
    }

    return -1;
}

/*
 * Converts a list of subscripts to a string.
 *
 * Returns -1 on error, the number of characters placed in subscripts
 * otherwise.
 */
static int
einsum_list_to_subscripts(PyObject *obj, char *subscripts, int subsize)
{
    int ellipsis = 0, subindex = 0;
    npy_intp i, size;
    PyObject *item;

    obj = PySequence_Fast(obj, "the subscripts for each operand must "
                               "be a list or a tuple");
    if (obj == NULL) {
        return -1;
    }
    size = PySequence_Size(obj);

    for (i = 0; i < size; ++i) {
        item = PySequence_Fast_GET_ITEM(obj, i);
        /* Ellipsis */
        if (item == Py_Ellipsis) {
            if (ellipsis) {
                PyErr_SetString(PyExc_ValueError,
                        "each subscripts list may have only one ellipsis");
                Py_DECREF(obj);
                return -1;
            }
            if (subindex + 3 >= subsize) {
                PyErr_SetString(PyExc_ValueError,
                        "subscripts list is too long");
                Py_DECREF(obj);
                return -1;
            }
            subscripts[subindex++] = '.';
            subscripts[subindex++] = '.';
            subscripts[subindex++] = '.';
            ellipsis = 1;
        }
        /* Subscript */
        else {
            npy_intp s = PyArray_PyIntAsIntp(item);
            /* Invalid */
            if (error_converting(s)) {
                PyErr_SetString(PyExc_TypeError,
                        "each subscript must be either an integer "
                        "or an ellipsis");
                Py_DECREF(obj);
                return -1;
            }
            npy_bool bad_input = 0;

            if (subindex + 1 >= subsize) {
                PyErr_SetString(PyExc_ValueError,
                        "subscripts list is too long");
                Py_DECREF(obj);
                return -1;
            }

            if (s < 0) {
                bad_input = 1;
            }
            else if (s < 26) {
                subscripts[subindex++] = 'A' + (char)s;
            }
            else if (s < 2*26) {
                subscripts[subindex++] = 'a' + (char)s - 26;
            }
            else {
                bad_input = 1;
            }

            if (bad_input) {
                PyErr_SetString(PyExc_ValueError,
                        "subscript is not within the valid range [0, 52)");
                Py_DECREF(obj);
                return -1;
            }              
        }
        
    }

    Py_DECREF(obj);

    return subindex;
}

/*
 * Fills in the subscripts, with maximum size subsize, and op,
 * with the values in the tuple 'args'.
 *
 * Returns -1 on error, number of operands placed in op otherwise.
 */
static int
einsum_sub_op_from_lists(PyObject *args,
                char *subscripts, int subsize, PyArrayObject **op)
{
    int subindex = 0;
    npy_intp i, nop;

    nop = PyTuple_Size(args)/2;

    if (nop == 0) {
        PyErr_SetString(PyExc_ValueError, "must provide at least an "
                        "operand and a subscripts list to einsum");
        return -1;
    }
    else if (nop >= NPY_MAXARGS) {
        PyErr_SetString(PyExc_ValueError, "too many operands");
        return -1;
    }

    /* Set the operands to NULL */
    for (i = 0; i < nop; ++i) {
        op[i] = NULL;
    }

    /* Get the operands and build the subscript string */
    for (i = 0; i < nop; ++i) {
        PyObject *obj = PyTuple_GET_ITEM(args, 2*i);
        int n;

        /* Comma between the subscripts for each operand */
        if (i != 0) {
            subscripts[subindex++] = ',';
            if (subindex >= subsize) {
                PyErr_SetString(PyExc_ValueError,
                        "subscripts list is too long");
                goto fail;
            }
        }

        op[i] = (PyArrayObject *)PyArray_FROM_OF(obj, NPY_ARRAY_ENSUREARRAY);
        if (op[i] == NULL) {
            goto fail;
        }

        obj = PyTuple_GET_ITEM(args, 2*i+1);
        n = einsum_list_to_subscripts(obj, subscripts+subindex,
                                      subsize-subindex);
        if (n < 0) {
            goto fail;
        }
        subindex += n;
    }

    /* Add the '->' to the string if provided */
    if (PyTuple_Size(args) == 2*nop+1) {
        PyObject *obj;
        int n;

        if (subindex + 2 >= subsize) {
            PyErr_SetString(PyExc_ValueError,
                    "subscripts list is too long");
            goto fail;
        }
        subscripts[subindex++] = '-';
        subscripts[subindex++] = '>';

        obj = PyTuple_GET_ITEM(args, 2*nop);
        n = einsum_list_to_subscripts(obj, subscripts+subindex,
                                      subsize-subindex);
        if (n < 0) {
            goto fail;
        }
        subindex += n;
    }

    /* NULL-terminate the subscripts string */
    subscripts[subindex] = '\0';

    return nop;

fail:
    for (i = 0; i < nop; ++i) {
        Py_XDECREF(op[i]);
        op[i] = NULL;
    }

    return -1;
}

static PyObject *
array_einsum(PyObject *NPY_UNUSED(dummy), PyObject *args, PyObject *kwds)
{
    char *subscripts = NULL, subscripts_buffer[256];
    PyObject *str_obj = NULL, *str_key_obj = NULL;
    PyObject *arg0;
    int i, nop;
    PyArrayObject *op[NPY_MAXARGS];
    NPY_ORDER order = NPY_KEEPORDER;
    NPY_CASTING casting = NPY_SAFE_CASTING;
    PyArrayObject *out = NULL;
    PyArray_Descr *dtype = NULL;
    PyObject *ret = NULL;

    if (PyTuple_GET_SIZE(args) < 1) {
        PyErr_SetString(PyExc_ValueError,
                        "must specify the einstein sum subscripts string "
                        "and at least one operand, or at least one operand "
                        "and its corresponding subscripts list");
        return NULL;
    }
    arg0 = PyTuple_GET_ITEM(args, 0);

    /* einsum('i,j', a, b), einsum('i,j->ij', a, b) */
    if (PyString_Check(arg0) || PyUnicode_Check(arg0)) {
        nop = einsum_sub_op_from_str(args, &str_obj, &subscripts, op);
    }
    /* einsum(a, [0], b, [1]), einsum(a, [0], b, [1], [0,1]) */
    else {
        nop = einsum_sub_op_from_lists(args, subscripts_buffer,
                                    sizeof(subscripts_buffer), op);
        subscripts = subscripts_buffer;
    }
    if (nop <= 0) {
        goto finish;
    }

    /* Get the keyword arguments */
    if (kwds != NULL) {
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        while (PyDict_Next(kwds, &pos, &key, &value)) {
            char *str = NULL;

            Py_XDECREF(str_key_obj);
            str_key_obj = PyUnicode_AsASCIIString(key);
            if (str_key_obj != NULL) {
                key = str_key_obj;
            }

            str = PyBytes_AsString(key);

            if (str == NULL) {
                PyErr_Clear();
                PyErr_SetString(PyExc_TypeError, "invalid keyword");
                goto finish;
            }

            if (strcmp(str,"out") == 0) {
                if (PyArray_Check(value)) {
                    out = (PyArrayObject *)value;
                }
                else {
                    PyErr_SetString(PyExc_TypeError,
                                "keyword parameter out must be an "
                                "array for einsum");
                    goto finish;
                }
            }
            else if (strcmp(str,"order") == 0) {
                if (!PyArray_OrderConverter(value, &order)) {
                    goto finish;
                }
            }
            else if (strcmp(str,"casting") == 0) {
                if (!PyArray_CastingConverter(value, &casting)) {
                    goto finish;
                }
            }
            else if (strcmp(str,"dtype") == 0) {
                if (!PyArray_DescrConverter2(value, &dtype)) {
                    goto finish;
                }
            }
            else {
                PyErr_Format(PyExc_TypeError,
                            "'%s' is an invalid keyword for einsum",
                            str);
                goto finish;
            }
        }
    }

    ret = (PyObject *)PyArray_EinsteinSum(subscripts, nop, op, dtype,
                                        order, casting, out);

    /* If no output was supplied, possibly convert to a scalar */
    if (ret != NULL && out == NULL) {
        ret = PyArray_Return((PyArrayObject *)ret);
    }

finish:
    for (i = 0; i < nop; ++i) {
        Py_XDECREF(op[i]);
    }
    Py_XDECREF(dtype);
    Py_XDECREF(str_obj);
    Py_XDECREF(str_key_obj);
    /* out is a borrowed reference */

    return ret;
}

static PyObject *
array_fastCopyAndTranspose(PyObject *NPY_UNUSED(dummy), PyObject *args)
{
    PyObject *a0;

    if (!PyArg_ParseTuple(args, "O:_fastCopyAndTranspose", &a0)) {
        return NULL;
    }
    return PyArray_Return((PyArrayObject *)PyArray_CopyAndTranspose(a0));
}

static PyObject *
array_correlate(PyObject *NPY_UNUSED(dummy), PyObject *args, PyObject *kwds)
{
    PyObject *shape, *a0;
    int mode = 0;
    static char *kwlist[] = {"a", "v", "mode", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|i:correlate", kwlist,
                &a0, &shape, &mode)) {
        return NULL;
    }
    return PyArray_Correlate(a0, shape, mode);
}

static PyObject*
array_correlate2(PyObject *NPY_UNUSED(dummy), PyObject *args, PyObject *kwds)
{
    PyObject *shape, *a0;
    int mode = 0;
    static char *kwlist[] = {"a", "v", "mode", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|i:correlate2", kwlist,
                &a0, &shape, &mode)) {
        return NULL;
    }
    return PyArray_Correlate2(a0, shape, mode);
}

static PyObject *
array_arange(PyObject *NPY_UNUSED(ignored), PyObject *args, PyObject *kws) {
    PyObject *o_start = NULL, *o_stop = NULL, *o_step = NULL, *range=NULL;
    static char *kwd[]= {"start", "stop", "step", "dtype", NULL};
    PyArray_Descr *typecode = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kws, "O|OOO&:arange", kwd,
                &o_start,
                &o_stop,
                &o_step,
                PyArray_DescrConverter2, &typecode)) {
        Py_XDECREF(typecode);
        return NULL;
    }
    range = PyArray_ArangeObj(o_start, o_stop, o_step, typecode);
    Py_XDECREF(typecode);

    return range;
}

/*NUMPY_API
 *
 * Included at the very first so not auto-grabbed and thus not labeled.
 */
NPY_NO_EXPORT unsigned int
PyArray_GetNDArrayCVersion(void)
{
    return (unsigned int)NPY_ABI_VERSION;
}

/*NUMPY_API
 * Returns the built-in (at compilation time) C API version
 */
NPY_NO_EXPORT unsigned int
PyArray_GetNDArrayCFeatureVersion(void)
{
    return (unsigned int)NPY_API_VERSION;
}

static PyObject *
array__get_ndarray_c_version(PyObject *NPY_UNUSED(dummy), PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "", kwlist )) {
        return NULL;
    }
    return PyInt_FromLong( (long) PyArray_GetNDArrayCVersion() );
}

/*NUMPY_API
*/
NPY_NO_EXPORT int
PyArray_GetEndianness(void)
{
    const union {
        npy_uint32 i;
        char c[4];
    } bint = {0x01020304};

    if (bint.c[0] == 1) {
        return NPY_CPU_BIG;
    }
    else if (bint.c[0] == 4) {
        return NPY_CPU_LITTLE;
    }
    else {
        return NPY_CPU_UNKNOWN_ENDIAN;
    }
}

static PyObject *
array__reconstruct(PyObject *NPY_UNUSED(dummy), PyObject *args)
{

    PyObject *ret;
    PyTypeObject *subtype;
    PyArray_Dims shape = {NULL, 0};
    PyArray_Descr *dtype = NULL;

    evil_global_disable_warn_O4O8_flag = 1;

    if (!PyArg_ParseTuple(args, "O!O&O&:_reconstruct",
                &PyType_Type, &subtype,
                PyArray_IntpConverter, &shape,
                PyArray_DescrConverter, &dtype)) {
        goto fail;
    }
    if (!PyType_IsSubtype(subtype, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError,
                "_reconstruct: First argument must be a sub-type of ndarray");
        goto fail;
    }
    ret = PyArray_NewFromDescr(subtype, dtype,
            (int)shape.len, shape.ptr, NULL, NULL, 0, NULL);
    npy_free_cache_dim_obj(shape);

    evil_global_disable_warn_O4O8_flag = 0;

    return ret;

fail:
    evil_global_disable_warn_O4O8_flag = 0;

    Py_XDECREF(dtype);
    npy_free_cache_dim_obj(shape);
    return NULL;
}

static PyObject *
array_set_string_function(PyObject *NPY_UNUSED(self), PyObject *args,
        PyObject *kwds)
{
    PyObject *op = NULL;
    int repr = 1;
    static char *kwlist[] = {"f", "repr", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|Oi:set_string_function", kwlist, &op, &repr)) {
        return NULL;
    }
    /* reset the array_repr function to built-in */
    if (op == Py_None) {
        op = NULL;
    }
    if (op != NULL && !PyCallable_Check(op)) {
        PyErr_SetString(PyExc_TypeError,
                "Argument must be callable.");
        return NULL;
    }
    PyArray_SetStringFunction(op, repr);
    Py_RETURN_NONE;
}

static PyObject *
array_set_ops_function(PyObject *NPY_UNUSED(self), PyObject *NPY_UNUSED(args),
        PyObject *kwds)
{
    PyObject *oldops = NULL;

    if ((oldops = _PyArray_GetNumericOps()) == NULL) {
        return NULL;
    }
    /*
     * Should probably ensure that objects are at least callable
     *  Leave this to the caller for now --- error will be raised
     *  later when use is attempted
     */
    if (kwds && PyArray_SetNumericOps(kwds) == -1) {
        Py_DECREF(oldops);
        if (PyErr_Occurred() == NULL) {
            PyErr_SetString(PyExc_ValueError,
                "one or more objects not callable");
        }
        return NULL;
    }
    return oldops;
}

static PyObject *
array_set_datetimeparse_function(PyObject *NPY_UNUSED(self),
        PyObject *NPY_UNUSED(args), PyObject *NPY_UNUSED(kwds))
{
    PyErr_SetString(PyExc_RuntimeError, "This function has been removed");
    return NULL;
}

/*
 * inner loop with constant size memcpy arguments
 * this allows the compiler to replace function calls while still handling the
 * alignment requirements of the platform.
 */
#define INNER_WHERE_LOOP(size) \
    do { \
        npy_intp i; \
        for (i = 0; i < n; i++) { \
            if (*csrc) { \
                memcpy(dst, xsrc, size); \
            } \
            else { \
                memcpy(dst, ysrc, size); \
            } \
            dst += size; \
            xsrc += xstride; \
            ysrc += ystride; \
            csrc += cstride; \
        } \
    } while(0)


/*NUMPY_API
 * Where
 */
NPY_NO_EXPORT PyObject *
PyArray_Where(PyObject *condition, PyObject *x, PyObject *y)
{
    PyArrayObject *arr, *ax, *ay;
    PyObject *ret = NULL;

    arr = (PyArrayObject *)PyArray_FROM_O(condition);
    if (arr == NULL) {
        return NULL;
    }
    if ((x == NULL) && (y == NULL)) {
        ret = PyArray_Nonzero(arr);
        Py_DECREF(arr);
        return ret;
    }
    if ((x == NULL) || (y == NULL)) {
        Py_DECREF(arr);
        PyErr_SetString(PyExc_ValueError,
                "either both or neither of x and y should be given");
        return NULL;
    }

    ax = (PyArrayObject*)PyArray_FROM_O(x);
    ay = (PyArrayObject*)PyArray_FROM_O(y);
    if (ax == NULL || ay == NULL) {
        goto fail;
    }
    else {
        npy_uint32 flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_BUFFERED |
                           NPY_ITER_REFS_OK | NPY_ITER_ZEROSIZE_OK;
        PyArrayObject * op_in[4] = {
            NULL, arr, ax, ay
        };
        npy_uint32 op_flags[4] = {
            NPY_ITER_WRITEONLY | NPY_ITER_ALLOCATE | NPY_ITER_NO_SUBTYPE,
            NPY_ITER_READONLY, NPY_ITER_READONLY, NPY_ITER_READONLY
        };
        PyArray_Descr * common_dt = PyArray_ResultType(2, &op_in[0] + 2,
                                                       0, NULL);
        PyArray_Descr * op_dt[4] = {common_dt, PyArray_DescrFromType(NPY_BOOL),
                                    common_dt, common_dt};
        NpyIter * iter;
        int needs_api;
        NPY_BEGIN_THREADS_DEF;

        if (common_dt == NULL || op_dt[1] == NULL) {
            Py_XDECREF(op_dt[1]);
            Py_XDECREF(common_dt);
            goto fail;
        }
        iter =  NpyIter_MultiNew(4, op_in, flags,
                                 NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                 op_flags, op_dt);
        Py_DECREF(op_dt[1]);
        Py_DECREF(common_dt);
        if (iter == NULL) {
            goto fail;
        }

        needs_api = NpyIter_IterationNeedsAPI(iter);

        /* Get the result from the iterator object array */
        ret = (PyObject*)NpyIter_GetOperandArray(iter)[0];

        NPY_BEGIN_THREADS_NDITER(iter);

        if (NpyIter_GetIterSize(iter) != 0) {
            NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
            npy_intp * innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);
            char **dataptrarray = NpyIter_GetDataPtrArray(iter);

            do {
                PyArray_Descr * dtx = NpyIter_GetDescrArray(iter)[2];
                PyArray_Descr * dty = NpyIter_GetDescrArray(iter)[3];
                int axswap = PyDataType_ISBYTESWAPPED(dtx);
                int ayswap = PyDataType_ISBYTESWAPPED(dty);
                PyArray_CopySwapFunc *copyswapx = dtx->f->copyswap;
                PyArray_CopySwapFunc *copyswapy = dty->f->copyswap;
                int native = (axswap == ayswap) && (axswap == 0) && !needs_api;
                npy_intp n = (*innersizeptr);
                npy_intp itemsize = NpyIter_GetDescrArray(iter)[0]->elsize;
                npy_intp cstride = NpyIter_GetInnerStrideArray(iter)[1];
                npy_intp xstride = NpyIter_GetInnerStrideArray(iter)[2];
                npy_intp ystride = NpyIter_GetInnerStrideArray(iter)[3];
                char * dst = dataptrarray[0];
                char * csrc = dataptrarray[1];
                char * xsrc = dataptrarray[2];
                char * ysrc = dataptrarray[3];

                /* constant sizes so compiler replaces memcpy */
                if (native && itemsize == 16) {
                    INNER_WHERE_LOOP(16);
                }
                else if (native && itemsize == 8) {
                    INNER_WHERE_LOOP(8);
                }
                else if (native && itemsize == 4) {
                    INNER_WHERE_LOOP(4);
                }
                else if (native && itemsize == 2) {
                    INNER_WHERE_LOOP(2);
                }
                else if (native && itemsize == 1) {
                    INNER_WHERE_LOOP(1);
                }
                else {
                    /* copyswap is faster than memcpy even if we are native */
                    npy_intp i;
                    for (i = 0; i < n; i++) {
                        if (*csrc) {
                            copyswapx(dst, xsrc, axswap, ret);
                        }
                        else {
                            copyswapy(dst, ysrc, ayswap, ret);
                        }
                        dst += itemsize;
                        xsrc += xstride;
                        ysrc += ystride;
                        csrc += cstride;
                    }
                }
            } while (iternext(iter));
        }

        NPY_END_THREADS;

        Py_INCREF(ret);
        Py_DECREF(arr);
        Py_DECREF(ax);
        Py_DECREF(ay);

        if (NpyIter_Deallocate(iter) != NPY_SUCCEED) {
            Py_DECREF(ret);
            return NULL;
        }

        return ret;
    }

fail:
    Py_DECREF(arr);
    Py_XDECREF(ax);
    Py_XDECREF(ay);
    return NULL;
}

#undef INNER_WHERE_LOOP

static PyObject *
array_where(PyObject *NPY_UNUSED(ignored), PyObject *args)
{
    PyObject *obj = NULL, *x = NULL, *y = NULL;

    if (!PyArg_ParseTuple(args, "O|OO:where", &obj, &x, &y)) {
        return NULL;
    }
    return PyArray_Where(obj, x, y);
}

static PyObject *
array_lexsort(PyObject *NPY_UNUSED(ignored), PyObject *args, PyObject *kwds)
{
    int axis = -1;
    PyObject *obj;
    static char *kwlist[] = {"keys", "axis", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|i:lexsort", kwlist, &obj, &axis)) {
        return NULL;
    }
    return PyArray_Return((PyArrayObject *)PyArray_LexSort(obj, axis));
}

static PyObject *
array_can_cast_safely(PyObject *NPY_UNUSED(self), PyObject *args,
        PyObject *kwds)
{
    PyObject *from_obj = NULL;
    PyArray_Descr *d1 = NULL;
    PyArray_Descr *d2 = NULL;
    npy_bool ret;
    PyObject *retobj = NULL;
    NPY_CASTING casting = NPY_SAFE_CASTING;
    static char *kwlist[] = {"from_", "to", "casting", NULL};

    if(!PyArg_ParseTupleAndKeywords(args, kwds, "OO&|O&:can_cast", kwlist,
                &from_obj,
                PyArray_DescrConverter2, &d2,
                PyArray_CastingConverter, &casting)) {
        goto finish;
    }
    if (d2 == NULL) {
        PyErr_SetString(PyExc_TypeError,
                "did not understand one of the types; 'None' not accepted");
        goto finish;
    }

    /* If the first parameter is an object or scalar, use CanCastArrayTo */
    if (PyArray_Check(from_obj)) {
        ret = PyArray_CanCastArrayTo((PyArrayObject *)from_obj, d2, casting);
    }
    else if (PyArray_IsScalar(from_obj, Generic) ||
                                PyArray_IsPythonNumber(from_obj)) {
        PyArrayObject *arr;
        arr = (PyArrayObject *)PyArray_FROM_O(from_obj);
        if (arr == NULL) {
            goto finish;
        }
        ret = PyArray_CanCastArrayTo(arr, d2, casting);
        Py_DECREF(arr);
    }
    /* Otherwise use CanCastTypeTo */
    else {
        if (!PyArray_DescrConverter2(from_obj, &d1) || d1 == NULL) {
            PyErr_SetString(PyExc_TypeError,
                    "did not understand one of the types; 'None' not accepted");
            goto finish;
        }
        ret = PyArray_CanCastTypeTo(d1, d2, casting);
    }

    retobj = ret ? Py_True : Py_False;
    Py_INCREF(retobj);

 finish:
    Py_XDECREF(d1);
    Py_XDECREF(d2);
    return retobj;
}

static PyObject *
array_promote_types(PyObject *NPY_UNUSED(dummy), PyObject *args)
{
    PyArray_Descr *d1 = NULL;
    PyArray_Descr *d2 = NULL;
    PyObject *ret = NULL;
    if (!PyArg_ParseTuple(args, "O&O&:promote_types",
                PyArray_DescrConverter2, &d1, PyArray_DescrConverter2, &d2)) {
        goto finish;
    }

    if (d1 == NULL || d2 == NULL) {
        PyErr_SetString(PyExc_TypeError,
                "did not understand one of the types");
        goto finish;
    }

    ret = (PyObject *)PyArray_PromoteTypes(d1, d2);

 finish:
    Py_XDECREF(d1);
    Py_XDECREF(d2);
    return ret;
}

static PyObject *
array_min_scalar_type(PyObject *NPY_UNUSED(dummy), PyObject *args)
{
    PyObject *array_in = NULL;
    PyArrayObject *array;
    PyObject *ret = NULL;

    if (!PyArg_ParseTuple(args, "O:min_scalar_type", &array_in)) {
        return NULL;
    }

    array = (PyArrayObject *)PyArray_FROM_O(array_in);
    if (array == NULL) {
        return NULL;
    }

    ret = (PyObject *)PyArray_MinScalarType(array);
    Py_DECREF(array);
    return ret;
}

static PyObject *
array_result_type(PyObject *NPY_UNUSED(dummy), PyObject *args)
{
    npy_intp i, len, narr = 0, ndtypes = 0;
    PyArrayObject **arr = NULL;
    PyArray_Descr **dtypes = NULL;
    PyObject *ret = NULL;

    len = PyTuple_GET_SIZE(args);
    if (len == 0) {
        PyErr_SetString(PyExc_ValueError,
                        "at least one array or dtype is required");
        goto finish;
    }

    arr = PyArray_malloc(2 * len * sizeof(void *));
    if (arr == NULL) {
        return PyErr_NoMemory();
    }
    dtypes = (PyArray_Descr**)&arr[len];

    for (i = 0; i < len; ++i) {
        PyObject *obj = PyTuple_GET_ITEM(args, i);
        if (PyArray_Check(obj)) {
            Py_INCREF(obj);
            arr[narr] = (PyArrayObject *)obj;
            ++narr;
        }
        else if (PyArray_IsScalar(obj, Generic) ||
                                    PyArray_IsPythonNumber(obj)) {
            arr[narr] = (PyArrayObject *)PyArray_FROM_O(obj);
            if (arr[narr] == NULL) {
                goto finish;
            }
            ++narr;
        }
        else {
            if (!PyArray_DescrConverter(obj, &dtypes[ndtypes])) {
                goto finish;
            }
            ++ndtypes;
        }
    }

    ret = (PyObject *)PyArray_ResultType(narr, arr, ndtypes, dtypes);

finish:
    for (i = 0; i < narr; ++i) {
        Py_DECREF(arr[i]);
    }
    for (i = 0; i < ndtypes; ++i) {
        Py_DECREF(dtypes[i]);
    }
    PyArray_free(arr);
    return ret;
}

static PyObject *
array_datetime_data(PyObject *NPY_UNUSED(dummy), PyObject *args)
{
    PyArray_Descr *dtype;
    PyArray_DatetimeMetaData *meta;

    if (!PyArg_ParseTuple(args, "O&:datetime_data",
                PyArray_DescrConverter, &dtype)) {
        return NULL;
    }

    meta = get_datetime_metadata_from_dtype(dtype);
    if (meta == NULL) {
        Py_DECREF(dtype);
        return NULL;
    }

    PyObject *res = convert_datetime_metadata_to_tuple(meta);
    Py_DECREF(dtype);
    return res;
}

/*
 * Prints floating-point scalars using the Dragon4 algorithm, scientific mode.
 * See docstring of `np.format_float_scientific` for description of arguments.
 * The differences is that a value of -1 is valid for pad_left, exp_digits,
 * precision, which is equivalent to `None`.
 */
static PyObject *
dragon4_scientific(PyObject *NPY_UNUSED(dummy), PyObject *args, PyObject *kwds)
{
    PyObject *obj;
    static char *kwlist[] = {"x", "precision", "unique", "sign", "trim",
                             "pad_left", "exp_digits", NULL};
    int precision=-1, pad_left=-1, exp_digits=-1;
    char *trimstr=NULL;
    DigitMode digit_mode;
    TrimMode trim = TrimMode_None;
    int sign=0, unique=1;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|iiisii:dragon4_scientific",
                kwlist, &obj, &precision, &unique, &sign, &trimstr, &pad_left,
                &exp_digits)) {
        return NULL;
    }

    if (trimstr != NULL) {
        if (strcmp(trimstr, "k") == 0) {
            trim = TrimMode_None;
        }
        else if (strcmp(trimstr, ".") == 0) {
            trim = TrimMode_Zeros;
        }
        else if (strcmp(trimstr, "0") == 0) {
            trim = TrimMode_LeaveOneZero;
        }
        else if (strcmp(trimstr, "-") == 0) {
            trim = TrimMode_DptZeros;
        }
        else {
            PyErr_SetString(PyExc_TypeError,
                "if supplied, trim must be 'k', '.', '0' or '-'");
            return NULL;
        }
    }

    digit_mode = unique ? DigitMode_Unique : DigitMode_Exact;

    if (unique == 0 && precision < 0) {
        PyErr_SetString(PyExc_TypeError,
            "in non-unique mode `precision` must be supplied");
        return NULL;
    }

    return Dragon4_Scientific(obj, digit_mode, precision, sign, trim,
                              pad_left, exp_digits);
}

/*
 * Prints floating-point scalars using the Dragon4 algorithm, positional mode.
 * See docstring of `np.format_float_positional` for description of arguments.
 * The differences is that a value of -1 is valid for pad_left, pad_right,
 * precision, which is equivalent to `None`.
 */
static PyObject *
dragon4_positional(PyObject *NPY_UNUSED(dummy), PyObject *args, PyObject *kwds)
{
    PyObject *obj;
    static char *kwlist[] = {"x", "precision", "unique", "fractional",
                             "sign", "trim", "pad_left", "pad_right", NULL};
    int precision=-1, pad_left=-1, pad_right=-1;
    char *trimstr=NULL;
    CutoffMode cutoff_mode;
    DigitMode digit_mode;
    TrimMode trim = TrimMode_None;
    int sign=0, unique=1, fractional=0;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|iiiisii:dragon4_positional",
                kwlist, &obj, &precision, &unique, &fractional, &sign, &trimstr,
                &pad_left, &pad_right)) {
        return NULL;
    }

    if (trimstr != NULL) {
        if (strcmp(trimstr, "k") == 0) {
            trim = TrimMode_None;
        }
        else if (strcmp(trimstr, ".") == 0) {
            trim = TrimMode_Zeros;
        }
        else if (strcmp(trimstr, "0") == 0) {
            trim = TrimMode_LeaveOneZero;
        }
        else if (strcmp(trimstr, "-") == 0) {
            trim = TrimMode_DptZeros;
        }
        else {
            PyErr_SetString(PyExc_TypeError,
                "if supplied, trim must be 'k', '.', '0' or '-'");
            return NULL;
        }
    }

    digit_mode = unique ? DigitMode_Unique : DigitMode_Exact;
    cutoff_mode = fractional ? CutoffMode_FractionLength :
                               CutoffMode_TotalLength;

    if (unique == 0 && precision < 0) {
        PyErr_SetString(PyExc_TypeError,
            "in non-unique mode `precision` must be supplied");
        return NULL;
    }

    return Dragon4_Positional(obj, digit_mode, cutoff_mode, precision, sign,
                              trim, pad_left, pad_right);
}

static PyObject *
format_longfloat(PyObject *NPY_UNUSED(dummy), PyObject *args, PyObject *kwds)
{
    PyObject *obj;
    unsigned int precision;
    static char *kwlist[] = {"x", "precision", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OI:format_longfloat", kwlist,
                &obj, &precision)) {
        return NULL;
    }
    if (!PyArray_IsScalar(obj, LongDouble)) {
        PyErr_SetString(PyExc_TypeError,
                "not a longfloat");
        return NULL;
    }
    return Dragon4_Scientific(obj, DigitMode_Unique, precision, 0,
                              TrimMode_LeaveOneZero, -1, -1);
}

static PyObject *
compare_chararrays(PyObject *NPY_UNUSED(dummy), PyObject *args, PyObject *kwds)
{
    PyObject *array;
    PyObject *other;
    PyArrayObject *newarr, *newoth;
    int cmp_op;
    npy_bool rstrip;
    char *cmp_str;
    Py_ssize_t strlength;
    PyObject *res = NULL;
    static char msg[] = "comparison must be '==', '!=', '<', '>', '<=', '>='";
    static char *kwlist[] = {"a1", "a2", "cmp", "rstrip", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOs#O&:compare_chararrays",
                kwlist,
                &array, &other, &cmp_str, &strlength,
                PyArray_BoolConverter, &rstrip)) {
        return NULL;
    }
    if (strlength < 1 || strlength > 2) {
        goto err;
    }
    if (strlength > 1) {
        if (cmp_str[1] != '=') {
            goto err;
        }
        if (cmp_str[0] == '=') {
            cmp_op = Py_EQ;
        }
        else if (cmp_str[0] == '!') {
            cmp_op = Py_NE;
        }
        else if (cmp_str[0] == '<') {
            cmp_op = Py_LE;
        }
        else if (cmp_str[0] == '>') {
            cmp_op = Py_GE;
        }
        else {
            goto err;
        }
    }
    else {
        if (cmp_str[0] == '<') {
            cmp_op = Py_LT;
        }
        else if (cmp_str[0] == '>') {
            cmp_op = Py_GT;
        }
        else {
            goto err;
        }
    }

    newarr = (PyArrayObject *)PyArray_FROM_O(array);
    if (newarr == NULL) {
        return NULL;
    }
    newoth = (PyArrayObject *)PyArray_FROM_O(other);
    if (newoth == NULL) {
        Py_DECREF(newarr);
        return NULL;
    }
    if (PyArray_ISSTRING(newarr) && PyArray_ISSTRING(newoth)) {
        res = _strings_richcompare(newarr, newoth, cmp_op, rstrip != 0);
    }
    else {
        PyErr_SetString(PyExc_TypeError,
                "comparison of non-string arrays");
    }
    Py_DECREF(newarr);
    Py_DECREF(newoth);
    return res;

 err:
    PyErr_SetString(PyExc_ValueError, msg);
    return NULL;
}

static PyObject *
_vec_string_with_args(PyArrayObject* char_array, PyArray_Descr* type,
                      PyObject* method, PyObject* args)
{
    PyObject* broadcast_args[NPY_MAXARGS];
    PyArrayMultiIterObject* in_iter = NULL;
    PyArrayObject* result = NULL;
    PyArrayIterObject* out_iter = NULL;
    Py_ssize_t i, n, nargs;

    nargs = PySequence_Size(args) + 1;
    if (nargs == -1 || nargs > NPY_MAXARGS) {
        PyErr_Format(PyExc_ValueError,
                "len(args) must be < %d", NPY_MAXARGS - 1);
        Py_DECREF(type);
        goto err;
    }

    broadcast_args[0] = (PyObject*)char_array;
    for (i = 1; i < nargs; i++) {
        PyObject* item = PySequence_GetItem(args, i-1);
        if (item == NULL) {
            Py_DECREF(type);
            goto err;
        }
        broadcast_args[i] = item;
        Py_DECREF(item);
    }
    in_iter = (PyArrayMultiIterObject*)PyArray_MultiIterFromObjects
        (broadcast_args, nargs, 0);
    if (in_iter == NULL) {
        Py_DECREF(type);
        goto err;
    }
    n = in_iter->numiter;

    result = (PyArrayObject*)PyArray_SimpleNewFromDescr(in_iter->nd,
            in_iter->dimensions, type);
    if (result == NULL) {
        goto err;
    }

    out_iter = (PyArrayIterObject*)PyArray_IterNew((PyObject*)result);
    if (out_iter == NULL) {
        goto err;
    }

    while (PyArray_MultiIter_NOTDONE(in_iter)) {
        PyObject* item_result;
        PyObject* args_tuple = PyTuple_New(n);
        if (args_tuple == NULL) {
            goto err;
        }

        for (i = 0; i < n; i++) {
            PyArrayIterObject* it = in_iter->iters[i];
            PyObject* arg = PyArray_ToScalar(PyArray_ITER_DATA(it), it->ao);
            if (arg == NULL) {
                Py_DECREF(args_tuple);
                goto err;
            }
            /* Steals ref to arg */
            PyTuple_SetItem(args_tuple, i, arg);
        }

        item_result = PyObject_CallObject(method, args_tuple);
        Py_DECREF(args_tuple);
        if (item_result == NULL) {
            goto err;
        }

        if (PyArray_SETITEM(result, PyArray_ITER_DATA(out_iter), item_result)) {
            Py_DECREF(item_result);
            PyErr_SetString( PyExc_TypeError,
                    "result array type does not match underlying function");
            goto err;
        }
        Py_DECREF(item_result);

        PyArray_MultiIter_NEXT(in_iter);
        PyArray_ITER_NEXT(out_iter);
    }

    Py_DECREF(in_iter);
    Py_DECREF(out_iter);

    return (PyObject*)result;

 err:
    Py_XDECREF(in_iter);
    Py_XDECREF(out_iter);
    Py_XDECREF(result);

    return 0;
}

static PyObject *
_vec_string_no_args(PyArrayObject* char_array,
                                   PyArray_Descr* type, PyObject* method)
{
    /*
     * This is a faster version of _vec_string_args to use when there
     * are no additional arguments to the string method.  This doesn't
     * require a broadcast iterator (and broadcast iterators don't work
     * with 1 argument anyway).
     */
    PyArrayIterObject* in_iter = NULL;
    PyArrayObject* result = NULL;
    PyArrayIterObject* out_iter = NULL;

    in_iter = (PyArrayIterObject*)PyArray_IterNew((PyObject*)char_array);
    if (in_iter == NULL) {
        Py_DECREF(type);
        goto err;
    }

    result = (PyArrayObject*)PyArray_SimpleNewFromDescr(
            PyArray_NDIM(char_array), PyArray_DIMS(char_array), type);
    if (result == NULL) {
        goto err;
    }

    out_iter = (PyArrayIterObject*)PyArray_IterNew((PyObject*)result);
    if (out_iter == NULL) {
        goto err;
    }

    while (PyArray_ITER_NOTDONE(in_iter)) {
        PyObject* item_result;
        PyObject* item = PyArray_ToScalar(in_iter->dataptr, in_iter->ao);
        if (item == NULL) {
            goto err;
        }

        item_result = PyObject_CallFunctionObjArgs(method, item, NULL);
        Py_DECREF(item);
        if (item_result == NULL) {
            goto err;
        }

        if (PyArray_SETITEM(result, PyArray_ITER_DATA(out_iter), item_result)) {
            Py_DECREF(item_result);
            PyErr_SetString( PyExc_TypeError,
                "result array type does not match underlying function");
            goto err;
        }
        Py_DECREF(item_result);

        PyArray_ITER_NEXT(in_iter);
        PyArray_ITER_NEXT(out_iter);
    }

    Py_DECREF(in_iter);
    Py_DECREF(out_iter);

    return (PyObject*)result;

 err:
    Py_XDECREF(in_iter);
    Py_XDECREF(out_iter);
    Py_XDECREF(result);

    return 0;
}

static PyObject *
_vec_string(PyObject *NPY_UNUSED(dummy), PyObject *args, PyObject *NPY_UNUSED(kwds))
{
    PyArrayObject* char_array = NULL;
    PyArray_Descr *type;
    PyObject* method_name;
    PyObject* args_seq = NULL;

    PyObject* method = NULL;
    PyObject* result = NULL;

    if (!PyArg_ParseTuple(args, "O&O&O|O",
                PyArray_Converter, &char_array,
                PyArray_DescrConverter, &type,
                &method_name, &args_seq)) {
        goto err;
    }

    if (PyArray_TYPE(char_array) == NPY_STRING) {
        method = PyObject_GetAttr((PyObject *)&PyString_Type, method_name);
    }
    else if (PyArray_TYPE(char_array) == NPY_UNICODE) {
        method = PyObject_GetAttr((PyObject *)&PyUnicode_Type, method_name);
    }
    else {
        PyErr_SetString(PyExc_TypeError,
                "string operation on non-string array");
        Py_DECREF(type);
        goto err;
    }
    if (method == NULL) {
        Py_DECREF(type);
        goto err;
    }

    if (args_seq == NULL
            || (PySequence_Check(args_seq) && PySequence_Size(args_seq) == 0)) {
        result = _vec_string_no_args(char_array, type, method);
    }
    else if (PySequence_Check(args_seq)) {
        result = _vec_string_with_args(char_array, type, method, args_seq);
    }
    else {
        Py_DECREF(type);
        PyErr_SetString(PyExc_TypeError,
                "'args' must be a sequence of arguments");
        goto err;
    }
    if (result == NULL) {
        goto err;
    }

    Py_DECREF(char_array);
    Py_DECREF(method);

    return (PyObject*)result;

 err:
    Py_XDECREF(char_array);
    Py_XDECREF(method);

    return 0;
}

#ifndef __NPY_PRIVATE_NO_SIGNAL

static NPY_TLS int sigint_buf_init = 0;
static NPY_TLS NPY_SIGJMP_BUF _NPY_SIGINT_BUF;

/*NUMPY_API
 */
NPY_NO_EXPORT void
_PyArray_SigintHandler(int signum)
{
    PyOS_setsig(signum, SIG_IGN);
    /*
     * jump buffer may be uninitialized as SIGINT allowing functions are usually
     * run in other threads than the master thread that receives the signal
     */
    if (sigint_buf_init > 0) {
        NPY_SIGLONGJMP(_NPY_SIGINT_BUF, signum);
    }
    /*
     * sending SIGINT to the worker threads to cancel them is job of the
     * application
     */
}

/*NUMPY_API
 */
NPY_NO_EXPORT void*
_PyArray_GetSigintBuf(void)
{
    sigint_buf_init = 1;
    return (void *)&_NPY_SIGINT_BUF;
}

#else

NPY_NO_EXPORT void
_PyArray_SigintHandler(int signum)
{
    return;
}

NPY_NO_EXPORT void*
_PyArray_GetSigintBuf(void)
{
    return NULL;
}

#endif


static PyObject *
array_shares_memory_impl(PyObject *args, PyObject *kwds, Py_ssize_t default_max_work,
                         int raise_exceptions)
{
    PyObject * self_obj = NULL;
    PyObject * other_obj = NULL;
    PyArrayObject * self = NULL;
    PyArrayObject * other = NULL;
    PyObject *max_work_obj = NULL;
    static char *kwlist[] = {"self", "other", "max_work", NULL};

    mem_overlap_t result;
    static PyObject *too_hard_cls = NULL;
    Py_ssize_t max_work;
    NPY_BEGIN_THREADS_DEF;

    max_work = default_max_work;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|O:shares_memory_impl", kwlist,
                                     &self_obj, &other_obj, &max_work_obj)) {
        return NULL;
    }

    if (PyArray_Check(self_obj)) {
        self = (PyArrayObject*)self_obj;
        Py_INCREF(self);
    }
    else {
        /* Use FromAny to enable checking overlap for objects exposing array
           interfaces etc. */
        self = (PyArrayObject*)PyArray_FROM_O(self_obj);
        if (self == NULL) {
            goto fail;
        }
    }

    if (PyArray_Check(other_obj)) {
        other = (PyArrayObject*)other_obj;
        Py_INCREF(other);
    }
    else {
        other = (PyArrayObject*)PyArray_FROM_O(other_obj);
        if (other == NULL) {
            goto fail;
        }
    }

    if (max_work_obj == NULL || max_work_obj == Py_None) {
        /* noop */
    }
    else if (PyLong_Check(max_work_obj)) {
        max_work = PyLong_AsSsize_t(max_work_obj);
        if (PyErr_Occurred()) {
            goto fail;
        }
    }
    else {
        PyErr_SetString(PyExc_ValueError, "max_work must be an integer");
        goto fail;
    }

    if (max_work < -2) {
        PyErr_SetString(PyExc_ValueError, "Invalid value for max_work");
        goto fail;
    }

    NPY_BEGIN_THREADS;
    result = solve_may_share_memory(self, other, max_work);
    NPY_END_THREADS;

    Py_XDECREF(self);
    Py_XDECREF(other);

    if (result == MEM_OVERLAP_NO) {
        Py_RETURN_FALSE;
    }
    else if (result == MEM_OVERLAP_YES) {
        Py_RETURN_TRUE;
    }
    else if (result == MEM_OVERLAP_OVERFLOW) {
        if (raise_exceptions) {
            PyErr_SetString(PyExc_OverflowError,
                            "Integer overflow in computing overlap");
            return NULL;
        }
        else {
            /* Don't know, so say yes */
            Py_RETURN_TRUE;
        }
    }
    else if (result == MEM_OVERLAP_TOO_HARD) {
        if (raise_exceptions) {
            npy_cache_import("numpy.core._exceptions", "TooHardError",
                             &too_hard_cls);
            if (too_hard_cls) {
                PyErr_SetString(too_hard_cls, "Exceeded max_work");
            }
            return NULL;
        }
        else {
            /* Don't know, so say yes */
            Py_RETURN_TRUE;
        }
    }
    else {
        /* Doesn't happen usually */
        PyErr_SetString(PyExc_RuntimeError,
                        "Error in computing overlap");
        return NULL;
    }

fail:
    Py_XDECREF(self);
    Py_XDECREF(other);
    return NULL;
}


static PyObject *
array_shares_memory(PyObject *NPY_UNUSED(ignored), PyObject *args, PyObject *kwds)
{
    return array_shares_memory_impl(args, kwds, NPY_MAY_SHARE_EXACT, 1);
}


static PyObject *
array_may_share_memory(PyObject *NPY_UNUSED(ignored), PyObject *args, PyObject *kwds)
{
    return array_shares_memory_impl(args, kwds, NPY_MAY_SHARE_BOUNDS, 0);
}

static PyObject *
normalize_axis_index(PyObject *NPY_UNUSED(self), PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"axis", "ndim", "msg_prefix", NULL};
    int axis;
    int ndim;
    PyObject *msg_prefix = Py_None;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "ii|O:normalize_axis_index",
                                     kwlist, &axis, &ndim, &msg_prefix)) {
        return NULL;
    }
    if (check_and_adjust_axis_msg(&axis, ndim, msg_prefix) < 0) {
        return NULL;
    }

    return PyInt_FromLong(axis);
}

static struct PyMethodDef array_module_methods[] = {
    {"_get_implementing_args",
        (PyCFunction)array__get_implementing_args,
        METH_VARARGS, NULL},
    {"_get_ndarray_c_version",
        (PyCFunction)array__get_ndarray_c_version,
        METH_VARARGS|METH_KEYWORDS, NULL},
    {"_reconstruct",
        (PyCFunction)array__reconstruct,
        METH_VARARGS, NULL},
    {"set_string_function",
        (PyCFunction)array_set_string_function,
        METH_VARARGS|METH_KEYWORDS, NULL},
    {"set_numeric_ops",
        (PyCFunction)array_set_ops_function,
        METH_VARARGS|METH_KEYWORDS, NULL},
    {"set_datetimeparse_function",
        (PyCFunction)array_set_datetimeparse_function,
        METH_VARARGS|METH_KEYWORDS, NULL},
    {"set_typeDict",
        (PyCFunction)array_set_typeDict,
        METH_VARARGS, NULL},
    {"array",
        (PyCFunction)_array_fromobject,
        METH_VARARGS|METH_KEYWORDS, NULL},
    {"copyto",
        (PyCFunction)array_copyto,
        METH_VARARGS|METH_KEYWORDS, NULL},
    {"nested_iters",
        (PyCFunction)NpyIter_NestedIters,
        METH_VARARGS|METH_KEYWORDS, NULL},
    {"arange",
        (PyCFunction)array_arange,
        METH_VARARGS|METH_KEYWORDS, NULL},
    {"zeros",
        (PyCFunction)array_zeros,
        METH_VARARGS|METH_KEYWORDS, NULL},
    {"count_nonzero",
        (PyCFunction)array_count_nonzero,
        METH_VARARGS|METH_KEYWORDS, NULL},
    {"empty",
        (PyCFunction)array_empty,
        METH_VARARGS|METH_KEYWORDS, NULL},
    {"empty_like",
        (PyCFunction)array_empty_like,
        METH_VARARGS|METH_KEYWORDS, NULL},
    {"scalar",
        (PyCFunction)array_scalar,
        METH_VARARGS|METH_KEYWORDS, NULL},
    {"where",
        (PyCFunction)array_where,
        METH_VARARGS, NULL},
    {"lexsort",
        (PyCFunction)array_lexsort,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"putmask",
        (PyCFunction)array_putmask,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"fromstring",
        (PyCFunction)array_fromstring,
        METH_VARARGS|METH_KEYWORDS, NULL},
    {"fromiter",
        (PyCFunction)array_fromiter,
        METH_VARARGS|METH_KEYWORDS, NULL},
    {"concatenate",
        (PyCFunction)array_concatenate,
        METH_VARARGS|METH_KEYWORDS, NULL},
    {"inner",
        (PyCFunction)array_innerproduct,
        METH_VARARGS, NULL},
    {"dot",
        (PyCFunction)array_matrixproduct,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"vdot",
        (PyCFunction)array_vdot,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"c_einsum",
        (PyCFunction)array_einsum,
        METH_VARARGS|METH_KEYWORDS, NULL},
    {"_fastCopyAndTranspose",
        (PyCFunction)array_fastCopyAndTranspose,
        METH_VARARGS, NULL},
    {"correlate",
        (PyCFunction)array_correlate,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"correlate2",
        (PyCFunction)array_correlate2,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"frombuffer",
        (PyCFunction)array_frombuffer,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"fromfile",
        (PyCFunction)array_fromfile,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"can_cast",
        (PyCFunction)array_can_cast_safely,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"promote_types",
        (PyCFunction)array_promote_types,
        METH_VARARGS, NULL},
    {"min_scalar_type",
        (PyCFunction)array_min_scalar_type,
        METH_VARARGS, NULL},
    {"result_type",
        (PyCFunction)array_result_type,
        METH_VARARGS, NULL},
    {"shares_memory",
        (PyCFunction)array_shares_memory,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"may_share_memory",
        (PyCFunction)array_may_share_memory,
        METH_VARARGS | METH_KEYWORDS, NULL},
    /* Datetime-related functions */
    {"datetime_data",
        (PyCFunction)array_datetime_data,
        METH_VARARGS, NULL},
    {"datetime_as_string",
        (PyCFunction)array_datetime_as_string,
        METH_VARARGS | METH_KEYWORDS, NULL},
    /* Datetime business-day API */
    {"busday_offset",
        (PyCFunction)array_busday_offset,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"busday_count",
        (PyCFunction)array_busday_count,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"is_busday",
        (PyCFunction)array_is_busday,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"format_longfloat",
        (PyCFunction)format_longfloat,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"dragon4_positional",
        (PyCFunction)dragon4_positional,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"dragon4_scientific",
        (PyCFunction)dragon4_scientific,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"compare_chararrays",
        (PyCFunction)compare_chararrays,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"_vec_string",
        (PyCFunction)_vec_string,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"_insert", (PyCFunction)arr_insert,
        METH_VARARGS | METH_KEYWORDS,
        "Insert vals sequentially into equivalent 1-d positions "
        "indicated by mask."},
    {"bincount", (PyCFunction)arr_bincount,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"_monotonicity", (PyCFunction)arr__monotonicity,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"implement_array_function",
        (PyCFunction)array_implement_array_function,
        METH_VARARGS, NULL},
    {"interp", (PyCFunction)arr_interp,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"interp_complex", (PyCFunction)arr_interp_complex,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"ravel_multi_index", (PyCFunction)arr_ravel_multi_index,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"unravel_index", (PyCFunction)arr_unravel_index,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"add_docstring", (PyCFunction)arr_add_docstring,
        METH_VARARGS, NULL},
    {"packbits", (PyCFunction)io_pack,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"unpackbits", (PyCFunction)io_unpack,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"normalize_axis_index", (PyCFunction)normalize_axis_index,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"set_legacy_print_mode", (PyCFunction)set_legacy_print_mode,
        METH_VARARGS, NULL},
    /* from umath */
    {"frompyfunc",
        (PyCFunction) ufunc_frompyfunc,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"seterrobj",
        (PyCFunction) ufunc_seterr,
        METH_VARARGS, NULL},
    {"geterrobj",
        (PyCFunction) ufunc_geterr,
        METH_VARARGS, NULL},
    {"_add_newdoc_ufunc", (PyCFunction)add_newdoc_ufunc,
        METH_VARARGS, NULL},
    {"_set_madvise_hugepage", (PyCFunction)_set_madvise_hugepage,
        METH_O, NULL},
    {NULL, NULL, 0, NULL}                /* sentinel */
};

#include "__multiarray_api.c"

/* Establish scalar-type hierarchy
 *
 *  For dual inheritance we need to make sure that the objects being
 *  inherited from have the tp->mro object initialized.  This is
 *  not necessarily true for the basic type objects of Python (it is
 *  checked for single inheritance but not dual in PyType_Ready).
 *
 *  Thus, we call PyType_Ready on the standard Python Types, here.
 */
static int
setup_scalartypes(PyObject *NPY_UNUSED(dict))
{
    if (PyType_Ready(&PyBool_Type) < 0) {
        return -1;
    }
    if (PyType_Ready(&PyFloat_Type) < 0) {
        return -1;
    }
    if (PyType_Ready(&PyComplex_Type) < 0) {
        return -1;
    }
    if (PyType_Ready(&PyString_Type) < 0) {
        return -1;
    }
    if (PyType_Ready(&PyUnicode_Type) < 0) {
        return -1;
    }

#define SINGLE_INHERIT(child, parent)                                   \
    Py##child##ArrType_Type.tp_base = &Py##parent##ArrType_Type;        \
    if (PyType_Ready(&Py##child##ArrType_Type) < 0) {                   \
        PyErr_Print();                                                  \
        PyErr_Format(PyExc_SystemError,                                 \
                     "could not initialize Py%sArrType_Type",           \
                     #child);                                           \
        return -1;                                                      \
    }

    if (PyType_Ready(&PyGenericArrType_Type) < 0) {
        return -1;
    }
    SINGLE_INHERIT(Number, Generic);
    SINGLE_INHERIT(Integer, Number);
    SINGLE_INHERIT(Inexact, Number);
    SINGLE_INHERIT(SignedInteger, Integer);
    SINGLE_INHERIT(UnsignedInteger, Integer);
    SINGLE_INHERIT(Floating, Inexact);
    SINGLE_INHERIT(ComplexFloating, Inexact);
    SINGLE_INHERIT(Flexible, Generic);
    SINGLE_INHERIT(Character, Flexible);

#define DUAL_INHERIT(child, parent1, parent2)                           \
    Py##child##ArrType_Type.tp_base = &Py##parent2##ArrType_Type;       \
    Py##child##ArrType_Type.tp_bases =                                  \
        Py_BuildValue("(OO)", &Py##parent2##ArrType_Type,               \
                      &Py##parent1##_Type);                             \
    Py##child##ArrType_Type.tp_hash = Py##parent1##_Type.tp_hash;       \
    if (PyType_Ready(&Py##child##ArrType_Type) < 0) {                   \
        PyErr_Print();                                                  \
        PyErr_Format(PyExc_SystemError,                                 \
                     "could not initialize Py%sArrType_Type",           \
                     #child);                                           \
        return -1;                                                      \
    }

#define DUAL_INHERIT2(child, parent1, parent2)                          \
    Py##child##ArrType_Type.tp_base = &Py##parent1##_Type;              \
    Py##child##ArrType_Type.tp_bases =                                  \
        Py_BuildValue("(OO)", &Py##parent1##_Type,                      \
                      &Py##parent2##ArrType_Type);                      \
    Py##child##ArrType_Type.tp_richcompare =                            \
        Py##parent1##_Type.tp_richcompare;                              \
    Py##child##ArrType_Type.tp_hash = Py##parent1##_Type.tp_hash;       \
    if (PyType_Ready(&Py##child##ArrType_Type) < 0) {                   \
        PyErr_Print();                                                  \
        PyErr_Format(PyExc_SystemError,                                 \
                     "could not initialize Py%sArrType_Type",           \
                     #child);                                           \
        return -1;                                                      \
    }

    SINGLE_INHERIT(Bool, Generic);
    SINGLE_INHERIT(Byte, SignedInteger);
    SINGLE_INHERIT(Short, SignedInteger);
    SINGLE_INHERIT(Int, SignedInteger);
    SINGLE_INHERIT(Long, SignedInteger);
    SINGLE_INHERIT(LongLong, SignedInteger);

    /* Datetime doesn't fit in any category */
    SINGLE_INHERIT(Datetime, Generic);
    /* Timedelta is an integer with an associated unit */
    SINGLE_INHERIT(Timedelta, SignedInteger);

    /*
       fprintf(stderr,
        "tp_free = %p, PyObject_Del = %p, int_tp_free = %p, base.tp_free = %p\n",
         PyIntArrType_Type.tp_free, PyObject_Del, PyInt_Type.tp_free,
         PySignedIntegerArrType_Type.tp_free);
     */
    SINGLE_INHERIT(UByte, UnsignedInteger);
    SINGLE_INHERIT(UShort, UnsignedInteger);
    SINGLE_INHERIT(UInt, UnsignedInteger);
    SINGLE_INHERIT(ULong, UnsignedInteger);
    SINGLE_INHERIT(ULongLong, UnsignedInteger);

    SINGLE_INHERIT(Half, Floating);
    SINGLE_INHERIT(Float, Floating);
    DUAL_INHERIT(Double, Float, Floating);
    SINGLE_INHERIT(LongDouble, Floating);

    SINGLE_INHERIT(CFloat, ComplexFloating);
    DUAL_INHERIT(CDouble, Complex, ComplexFloating);
    SINGLE_INHERIT(CLongDouble, ComplexFloating);

    DUAL_INHERIT2(String, String, Character);
    DUAL_INHERIT2(Unicode, Unicode, Character);

    SINGLE_INHERIT(Void, Flexible);

    SINGLE_INHERIT(Object, Generic);

    return 0;

#undef SINGLE_INHERIT
#undef DUAL_INHERIT
#undef DUAL_INHERIT2

    /*
     * Clean up string and unicode array types so they act more like
     * strings -- get their tables from the standard types.
     */
}

/* place a flag dictionary in d */

static void
set_flaginfo(PyObject *d)
{
    PyObject *s;
    PyObject *newd;

    newd = PyDict_New();

#define _addnew(key, val, one)                                       \
    PyDict_SetItemString(newd, #key, s=PyInt_FromLong(val));    \
    Py_DECREF(s);                                               \
    PyDict_SetItemString(newd, #one, s=PyInt_FromLong(val));    \
    Py_DECREF(s)

#define _addone(key, val)                                            \
    PyDict_SetItemString(newd, #key, s=PyInt_FromLong(val));    \
    Py_DECREF(s)

    _addnew(OWNDATA, NPY_ARRAY_OWNDATA, O);
    _addnew(FORTRAN, NPY_ARRAY_F_CONTIGUOUS, F);
    _addnew(CONTIGUOUS, NPY_ARRAY_C_CONTIGUOUS, C);
    _addnew(ALIGNED, NPY_ARRAY_ALIGNED, A);
    _addnew(UPDATEIFCOPY, NPY_ARRAY_UPDATEIFCOPY, U);
    _addnew(WRITEBACKIFCOPY, NPY_ARRAY_WRITEBACKIFCOPY, X);
    _addnew(WRITEABLE, NPY_ARRAY_WRITEABLE, W);
    _addone(C_CONTIGUOUS, NPY_ARRAY_C_CONTIGUOUS);
    _addone(F_CONTIGUOUS, NPY_ARRAY_F_CONTIGUOUS);

#undef _addone
#undef _addnew

    PyDict_SetItemString(d, "_flagdict", newd);
    Py_DECREF(newd);
    return;
}

NPY_VISIBILITY_HIDDEN PyObject * npy_ma_str_array = NULL;
NPY_VISIBILITY_HIDDEN PyObject * npy_ma_str_array_prepare = NULL;
NPY_VISIBILITY_HIDDEN PyObject * npy_ma_str_array_wrap = NULL;
NPY_VISIBILITY_HIDDEN PyObject * npy_ma_str_array_finalize = NULL;
NPY_VISIBILITY_HIDDEN PyObject * npy_ma_str_ufunc = NULL;
NPY_VISIBILITY_HIDDEN PyObject * npy_ma_str_implementation = NULL;
NPY_VISIBILITY_HIDDEN PyObject * npy_ma_str_order = NULL;
NPY_VISIBILITY_HIDDEN PyObject * npy_ma_str_copy = NULL;
NPY_VISIBILITY_HIDDEN PyObject * npy_ma_str_dtype = NULL;
NPY_VISIBILITY_HIDDEN PyObject * npy_ma_str_ndmin = NULL;
NPY_VISIBILITY_HIDDEN PyObject * npy_ma_str_axis1 = NULL;
NPY_VISIBILITY_HIDDEN PyObject * npy_ma_str_axis2 = NULL;

static int
intern_strings(void)
{
    npy_ma_str_array = PyUString_InternFromString("__array__");
    npy_ma_str_array_prepare = PyUString_InternFromString("__array_prepare__");
    npy_ma_str_array_wrap = PyUString_InternFromString("__array_wrap__");
    npy_ma_str_array_finalize = PyUString_InternFromString("__array_finalize__");
    npy_ma_str_ufunc = PyUString_InternFromString("__array_ufunc__");
    npy_ma_str_implementation = PyUString_InternFromString("_implementation");
    npy_ma_str_order = PyUString_InternFromString("order");
    npy_ma_str_copy = PyUString_InternFromString("copy");
    npy_ma_str_dtype = PyUString_InternFromString("dtype");
    npy_ma_str_ndmin = PyUString_InternFromString("ndmin");
    npy_ma_str_axis1 = PyUString_InternFromString("axis1");
    npy_ma_str_axis2 = PyUString_InternFromString("axis2");

    return npy_ma_str_array && npy_ma_str_array_prepare &&
           npy_ma_str_array_wrap && npy_ma_str_array_finalize &&
           npy_ma_str_ufunc && npy_ma_str_implementation &&
           npy_ma_str_order && npy_ma_str_copy && npy_ma_str_dtype &&
           npy_ma_str_ndmin && npy_ma_str_axis1 && npy_ma_str_axis2;
}

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_multiarray_umath",
        NULL,
        -1,
        array_module_methods,
        NULL,
        NULL,
        NULL,
        NULL
};

/* Initialization function for the module */
PyMODINIT_FUNC PyInit__multiarray_umath(void) {
    PyObject *m, *d, *s;
    PyObject *c_api;

    /* Initialize CPU features */
    if (npy_cpu_init() < 0) {
        goto err;
    }

    /* Create the module and add the functions */
    m = PyModule_Create(&moduledef);
    if (!m) {
        goto err;
    }

#if defined(MS_WIN64) && defined(__GNUC__)
  PyErr_WarnEx(PyExc_Warning,
        "Numpy built with MINGW-W64 on Windows 64 bits is experimental, " \
        "and only available for \n" \
        "testing. You are advised not to use it for production. \n\n" \
        "CRASHES ARE TO BE EXPECTED - PLEASE REPORT THEM TO NUMPY DEVELOPERS",
        1);
#endif

    /* Initialize access to the PyDateTime API */
    numpy_pydatetime_import();

    if (PyErr_Occurred()) {
        goto err;
    }

    /* Add some symbolic constants to the module */
    d = PyModule_GetDict(m);
    if (!d) {
        goto err;
    }

    /*
     * Before calling PyType_Ready, initialize the tp_hash slot in
     * PyArray_Type to work around mingw32 not being able initialize
     * static structure slots with functions from the Python C_API.
     */
    PyArray_Type.tp_hash = PyObject_HashNotImplemented;

    if (PyType_Ready(&PyUFunc_Type) < 0) {
        goto err;
    }

    /* Load the ufunc operators into the array module's namespace */
    if (InitOperators(d) < 0) {
        goto err;
    }

    if (set_matmul_flags(d) < 0) {
        goto err;
    }
    initialize_casting_tables();
    initialize_numeric_types();
    if (initscalarmath(m) < 0) {
        goto err;
    }

    if (PyType_Ready(&PyArray_Type) < 0) {
        goto err;
    }
    if (setup_scalartypes(d) < 0) {
        goto err;
    }
    PyArrayIter_Type.tp_iter = PyObject_SelfIter;
    NpyIter_Type.tp_iter = PyObject_SelfIter;
    PyArrayMultiIter_Type.tp_iter = PyObject_SelfIter;
    PyArrayMultiIter_Type.tp_free = PyArray_free;
    if (PyType_Ready(&PyArrayIter_Type) < 0) {
        goto err;
    }
    if (PyType_Ready(&PyArrayMapIter_Type) < 0) {
        goto err;
    }
    if (PyType_Ready(&PyArrayMultiIter_Type) < 0) {
        goto err;
    }
    PyArrayNeighborhoodIter_Type.tp_new = PyType_GenericNew;
    if (PyType_Ready(&PyArrayNeighborhoodIter_Type) < 0) {
        goto err;
    }
    if (PyType_Ready(&NpyIter_Type) < 0) {
        goto err;
    }

    PyArrayDescr_Type.tp_hash = PyArray_DescrHash;
    if (PyType_Ready(&PyArrayDescr_Type) < 0) {
        goto err;
    }
    if (PyType_Ready(&PyArrayFlags_Type) < 0) {
        goto err;
    }
    NpyBusDayCalendar_Type.tp_new = PyType_GenericNew;
    if (PyType_Ready(&NpyBusDayCalendar_Type) < 0) {
        goto err;
    }

    c_api = NpyCapsule_FromVoidPtr((void *)PyArray_API, NULL);
    if (c_api == NULL) {
        goto err;
    }
    PyDict_SetItemString(d, "_ARRAY_API", c_api);
    Py_DECREF(c_api);

    c_api = NpyCapsule_FromVoidPtr((void *)PyUFunc_API, NULL);
    if (c_api == NULL) {
        goto err;
    }
    PyDict_SetItemString(d, "_UFUNC_API", c_api);
    Py_DECREF(c_api);
    if (PyErr_Occurred()) {
        goto err;
    }

    /*
     * PyExc_Exception should catch all the standard errors that are
     * now raised instead of the string exception "multiarray.error"

     * This is for backward compatibility with existing code.
     */
    PyDict_SetItemString (d, "error", PyExc_Exception);

    s = PyInt_FromLong(NPY_TRACE_DOMAIN);
    PyDict_SetItemString(d, "tracemalloc_domain", s);
    Py_DECREF(s);

    s = PyUString_FromString("3.1");
    PyDict_SetItemString(d, "__version__", s);
    Py_DECREF(s);

    s = npy_cpu_features_dict();
    if (s == NULL) {
        goto err;
    }
    if (PyDict_SetItemString(d, "__cpu_features__", s) < 0) {
        Py_DECREF(s);
        goto err;
    }
    Py_DECREF(s);

    s = NpyCapsule_FromVoidPtr((void *)_datetime_strings, NULL);
    if (s == NULL) {
        goto err;
    }
    PyDict_SetItemString(d, "DATETIMEUNITS", s);
    Py_DECREF(s);

#define ADDCONST(NAME)                          \
    s = PyInt_FromLong(NPY_##NAME);             \
    PyDict_SetItemString(d, #NAME, s);          \
    Py_DECREF(s)


    ADDCONST(ALLOW_THREADS);
    ADDCONST(BUFSIZE);
    ADDCONST(CLIP);

    ADDCONST(ITEM_HASOBJECT);
    ADDCONST(LIST_PICKLE);
    ADDCONST(ITEM_IS_POINTER);
    ADDCONST(NEEDS_INIT);
    ADDCONST(NEEDS_PYAPI);
    ADDCONST(USE_GETITEM);
    ADDCONST(USE_SETITEM);

    ADDCONST(RAISE);
    ADDCONST(WRAP);
    ADDCONST(MAXDIMS);

    ADDCONST(MAY_SHARE_BOUNDS);
    ADDCONST(MAY_SHARE_EXACT);
#undef ADDCONST

    PyDict_SetItemString(d, "ndarray", (PyObject *)&PyArray_Type);
    PyDict_SetItemString(d, "flatiter", (PyObject *)&PyArrayIter_Type);
    PyDict_SetItemString(d, "nditer", (PyObject *)&NpyIter_Type);
    PyDict_SetItemString(d, "broadcast",
                         (PyObject *)&PyArrayMultiIter_Type);
    PyDict_SetItemString(d, "dtype", (PyObject *)&PyArrayDescr_Type);
    PyDict_SetItemString(d, "flagsobj", (PyObject *)&PyArrayFlags_Type);

    /* Business day calendar object */
    PyDict_SetItemString(d, "busdaycalendar",
                            (PyObject *)&NpyBusDayCalendar_Type);
    set_flaginfo(d);

    /* Create the typeinfo types */
    if (typeinfo_init_structsequences(d) < 0) {
        goto err;
    }

    if (!intern_strings()) {
        goto err;
    }

    if (set_typeinfo(d) != 0) {
        goto err;
    }
    if (initumath(m) != 0) {
        goto err;
    }
    return m;

 err:
    if (!PyErr_Occurred()) {
        PyErr_SetString(PyExc_RuntimeError,
                        "cannot load multiarray module.");
    }
    return NULL;
}
