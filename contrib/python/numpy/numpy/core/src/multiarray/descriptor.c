/* Array Descr Object */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"

#include "npy_config.h"
#include "npy_ctypes.h"
#include "npy_pycompat.h"

#include "_datetime.h"
#include "common.h"
#include "templ_common.h" /* for npy_mul_with_overflow_intp */
#include "descriptor.h"
#include "alloc.h"
#include "assert.h"
#include "buffer.h"

/*
 * offset:    A starting offset.
 * alignment: A power-of-two alignment.
 *
 * This macro returns the smallest value >= 'offset'
 * that is divisible by 'alignment'. Because 'alignment'
 * is a power of two and integers are twos-complement,
 * it is possible to use some simple bit-fiddling to do this.
 */
#define NPY_NEXT_ALIGNED_OFFSET(offset, alignment) \
                (((offset) + (alignment) - 1) & (-(alignment)))

#ifndef PyDictProxy_Check
#define PyDictProxy_Check(obj) (Py_TYPE(obj) == &PyDictProxy_Type)
#endif

static PyObject *typeDict = NULL;   /* Must be explicitly loaded */

static PyArray_Descr *
_use_inherit(PyArray_Descr *type, PyObject *newobj, int *errflag);

static PyArray_Descr *
_arraydescr_from_ctypes_type(PyTypeObject *type)
{
    PyObject *_numpy_dtype_ctypes;
    PyObject *res;

    /* Call the python function of the same name. */
    _numpy_dtype_ctypes = PyImport_ImportModule("numpy.core._dtype_ctypes");
    if (_numpy_dtype_ctypes == NULL) {
        return NULL;
    }
    res = PyObject_CallMethod(_numpy_dtype_ctypes, "dtype_from_ctypes_type", "O", (PyObject *)type);
    Py_DECREF(_numpy_dtype_ctypes);
    if (res == NULL) {
        return NULL;
    }

    /*
     * sanity check that dtype_from_ctypes_type returned the right type,
     * since getting it wrong would give segfaults.
     */
    if (!PyObject_TypeCheck(res, &PyArrayDescr_Type)) {
        Py_DECREF(res);
        PyErr_BadInternalCall();
        return NULL;
    }

    return (PyArray_Descr *)res;
}

/*
 * This function creates a dtype object when the object has a "dtype" attribute,
 * and it can be converted to a dtype object.
 *
 * Returns a new reference to a dtype object, or NULL
 * if this is not possible. When it returns NULL, it does
 * not set a Python exception.
 */
NPY_NO_EXPORT PyArray_Descr *
_arraydescr_from_dtype_attr(PyObject *obj)
{
    PyObject *dtypedescr;
    PyArray_Descr *newdescr = NULL;
    int ret;

    /* For arbitrary objects that have a "dtype" attribute */
    dtypedescr = PyObject_GetAttrString(obj, "dtype");
    PyErr_Clear();
    if (dtypedescr == NULL) {
        return NULL;
    }

    ret = PyArray_DescrConverter(dtypedescr, &newdescr);
    Py_DECREF(dtypedescr);
    if (ret != NPY_SUCCEED) {
        PyErr_Clear();
        return NULL;
    }

    return newdescr;
}

/*
 * Sets the global typeDict object, which is a dictionary mapping
 * dtype names to numpy scalar types.
 */
NPY_NO_EXPORT PyObject *
array_set_typeDict(PyObject *NPY_UNUSED(ignored), PyObject *args)
{
    PyObject *dict;

    if (!PyArg_ParseTuple(args, "O:set_typeDict", &dict)) {
        return NULL;
    }
    /* Decrement old reference (if any)*/
    Py_XDECREF(typeDict);
    typeDict = dict;
    /* Create an internal reference to it */
    Py_INCREF(dict);
    Py_RETURN_NONE;
}

#define _chk_byteorder(arg) (arg == '>' || arg == '<' ||        \
                             arg == '|' || arg == '=')

static int
_check_for_commastring(char *type, Py_ssize_t len)
{
    Py_ssize_t i;
    int sqbracket;

    /* Check for ints at start of string */
    if ((type[0] >= '0'
                && type[0] <= '9')
            || ((len > 1)
                && _chk_byteorder(type[0])
                && (type[1] >= '0'
                && type[1] <= '9'))) {
        return 1;
    }
    /* Check for empty tuple */
    if (((len > 1)
                && (type[0] == '('
                && type[1] == ')'))
            || ((len > 3)
                && _chk_byteorder(type[0])
                && (type[1] == '('
                && type[2] == ')'))) {
        return 1;
    }
    /*
     * Check for presence of commas outside square [] brackets. This
     * allows commas inside of [], for parameterized dtypes to use.
     */
    sqbracket = 0;
    for (i = 0; i < len; i++) {
        switch (type[i]) {
            case ',':
                if (sqbracket == 0) {
                    return 1;
                }
                break;
            case '[':
                ++sqbracket;
                break;
            case ']':
                --sqbracket;
                break;
        }
    }
    return 0;
}

#undef _chk_byteorder

static int
is_datetime_typestr(char *type, Py_ssize_t len)
{
    if (len < 2) {
        return 0;
    }
    if (type[1] == '8' && (type[0] == 'M' || type[0] == 'm')) {
        return 1;
    }
    if (len < 10) {
        return 0;
    }
    if (strncmp(type, "datetime64", 10) == 0) {
        return 1;
    }
    if (len < 11) {
        return 0;
    }
    if (strncmp(type, "timedelta64", 11) == 0) {
        return 1;
    }
    return 0;
}

static PyArray_Descr *
_convert_from_tuple(PyObject *obj, int align)
{
    PyArray_Descr *type, *res;
    PyObject *val;
    int errflag;

    if (PyTuple_GET_SIZE(obj) != 2) {
        return NULL;
    }
    if (align) {
        if (!PyArray_DescrAlignConverter(PyTuple_GET_ITEM(obj, 0), &type)) {
            return NULL;
        }
    }
    else {
        if (!PyArray_DescrConverter(PyTuple_GET_ITEM(obj, 0), &type)) {
            return NULL;
        }
    }    
    val = PyTuple_GET_ITEM(obj,1);
    /* try to interpret next item as a type */
    res = _use_inherit(type, val, &errflag);
    if (res || errflag) {
        Py_DECREF(type);
        return res;
    }
    PyErr_Clear();
    /*
     * We get here if res was NULL but errflag wasn't set
     * --- i.e. the conversion to a data-descr failed in _use_inherit
     */
    if (PyDataType_ISUNSIZED(type)) {
        /* interpret next item as a typesize */
        int itemsize = PyArray_PyIntAsInt(PyTuple_GET_ITEM(obj,1));

        if (error_converting(itemsize)) {
            PyErr_SetString(PyExc_ValueError,
                    "invalid itemsize in generic type tuple");
            Py_DECREF(type);
            return NULL;
        }
        PyArray_DESCR_REPLACE(type);
        if (type == NULL) {
            return NULL;
        }
        if (type->type_num == NPY_UNICODE) {
            type->elsize = itemsize << 2;
        }
        else {
            type->elsize = itemsize;
        }
        return type;
    }
    else if (type->metadata && (PyDict_Check(val) || PyDictProxy_Check(val))) {
        /* Assume it's a metadata dictionary */
        if (PyDict_Merge(type->metadata, val, 0) == -1) {
            Py_DECREF(type);
            return NULL;
        }
        return type;
    }
    else {
        /*
         * interpret next item as shape (if it's a tuple)
         * and reset the type to NPY_VOID with
         * a new fields attribute.
         */
        PyArray_Dims shape = {NULL, -1};
        PyArray_Descr *newdescr = NULL;
        npy_intp items;
        int i, overflowed;
        int nbytes;

        if (!(PyArray_IntpConverter(val, &shape)) || (shape.len > NPY_MAXDIMS)) {
            PyErr_SetString(PyExc_ValueError,
                    "invalid shape in fixed-type tuple.");
            goto fail;
        }
        /*
         * If (type, 1) was given, it is equivalent to type...
         * or (type, ()) was given it is equivalent to type...
         */
        if ((shape.len == 1
                    && shape.ptr[0] == 1
                    && PyNumber_Check(val))
                || (shape.len == 0
                    && PyTuple_Check(val))) {
            npy_free_cache_dim_obj(shape);
            return type;
        }

        /* validate and set shape */
        for (i=0; i < shape.len; i++) {
            if (shape.ptr[i] < 0) {
                PyErr_SetString(PyExc_ValueError,
                                "invalid shape in fixed-type tuple: "
                                "dimension smaller then zero.");
                goto fail;
            }
            if (shape.ptr[i] > NPY_MAX_INT) {
                PyErr_SetString(PyExc_ValueError,
                                "invalid shape in fixed-type tuple: "
                                "dimension does not fit into a C int.");
                goto fail;
            }
        }
        items = PyArray_OverflowMultiplyList(shape.ptr, shape.len);
        if (items < 0 || items > NPY_MAX_INT) {
            overflowed = 1;
        }
        else {
            overflowed = npy_mul_with_overflow_int(
                &nbytes, type->elsize, (int) items);
        }
        if (overflowed) {
            PyErr_SetString(PyExc_ValueError,
                            "invalid shape in fixed-type tuple: dtype size in "
                            "bytes must fit into a C int.");
            goto fail;
        }
        newdescr = PyArray_DescrNewFromType(NPY_VOID);
        if (newdescr == NULL) {
            goto fail;
        }
        newdescr->elsize = nbytes;
        newdescr->subarray = PyArray_malloc(sizeof(PyArray_ArrayDescr));
        if (newdescr->subarray == NULL) {
            PyErr_NoMemory();
            goto fail;
        }
        newdescr->flags = type->flags;
        newdescr->alignment = type->alignment;
        newdescr->subarray->base = type;
        type = NULL;
        Py_XDECREF(newdescr->fields);
        Py_XDECREF(newdescr->names);
        newdescr->fields = NULL;
        newdescr->names = NULL;

        /*
         * Create a new subarray->shape tuple (it can be an arbitrary
         * sequence of integer like objects, neither of which is safe.
         */
        newdescr->subarray->shape = PyTuple_New(shape.len);
        if (newdescr->subarray->shape == NULL) {
            goto fail;
        }
        for (i=0; i < shape.len; i++) {
            PyTuple_SET_ITEM(newdescr->subarray->shape, i,
                             PyInt_FromLong((long)shape.ptr[i]));

            if (PyTuple_GET_ITEM(newdescr->subarray->shape, i) == NULL) {
                goto fail;
            }
        }

        npy_free_cache_dim_obj(shape);
        return newdescr;

    fail:
        Py_XDECREF(type);
        Py_XDECREF(newdescr);
        npy_free_cache_dim_obj(shape);
        return NULL;
    }
}

/*
 * obj is a list.  Each item is a tuple with
 *
 * (field-name, data-type (either a list or a string), and an optional
 * shape parameter).
 *
 * field-name can be a string or a 2-tuple
 * data-type can now be a list, string, or 2-tuple
 *          (string, metadata dictionary)
 */
static PyArray_Descr *
_convert_from_array_descr(PyObject *obj, int align)
{
    int n, i, totalsize;
    int ret;
    PyObject *fields, *item, *newobj;
    PyObject *name, *tup, *title;
    PyObject *nameslist;
    PyArray_Descr *new;
    PyArray_Descr *conv;
    /* Types with fields need the Python C API for field access */
    char dtypeflags = NPY_NEEDS_PYAPI;
    int maxalign = 0;

    n = PyList_GET_SIZE(obj);
    nameslist = PyTuple_New(n);
    if (!nameslist) {
        return NULL;
    }
    totalsize = 0;
    fields = PyDict_New();
    for (i = 0; i < n; i++) {
        item = PyList_GET_ITEM(obj, i);
        if (!PyTuple_Check(item) || (PyTuple_GET_SIZE(item) < 2)) {
            goto fail;
        }
        name = PyTuple_GET_ITEM(item, 0);
        if (PyBaseString_Check(name)) {
            title = NULL;
        }
        else if (PyTuple_Check(name)) {
            if (PyTuple_GET_SIZE(name) != 2) {
                goto fail;
            }
            title = PyTuple_GET_ITEM(name, 0);
            name = PyTuple_GET_ITEM(name, 1);
            if (!PyBaseString_Check(name)) {
                goto fail;
            }
        }
        else {
            goto fail;
        }

        /* Insert name into nameslist */
        Py_INCREF(name);

#if !defined(NPY_PY3K)
        /* convert unicode name to ascii on Python 2 if possible */ 
        if (PyUnicode_Check(name)) {
            PyObject *tmp = PyUnicode_AsASCIIString(name);
            Py_DECREF(name);
            if (tmp == NULL) { 
                goto fail;
            }
            name = tmp;
        }
#endif
        if (PyUString_GET_SIZE(name) == 0) {
            Py_DECREF(name);
            if (title == NULL) {
                name = PyUString_FromFormat("f%d", i);
            }
#if defined(NPY_PY3K)
            /* On Py3, allow only non-empty Unicode strings as field names */
            else if (PyUString_Check(title) && PyUString_GET_SIZE(title) > 0) {
                name = title;
                Py_INCREF(name);
            }
            else {
                goto fail;
            }
#else
            else {
                name = title;
                Py_INCREF(name);
            }
#endif
        }
        PyTuple_SET_ITEM(nameslist, i, name);

        /* Process rest */

        if (PyTuple_GET_SIZE(item) == 2) {
            if (align) {
                ret = PyArray_DescrAlignConverter(PyTuple_GET_ITEM(item, 1),
                                                        &conv);
            }
            else {
                ret = PyArray_DescrConverter(PyTuple_GET_ITEM(item, 1), &conv);
            }
        }
        else if (PyTuple_GET_SIZE(item) == 3) {
            newobj = PyTuple_GetSlice(item, 1, 3);
            if (align) {
                ret = PyArray_DescrAlignConverter(newobj, &conv);
            }
            else {
                ret = PyArray_DescrConverter(newobj, &conv);
            }
            Py_DECREF(newobj);
        }
        else {
            goto fail;
        }
        if (ret == NPY_FAIL) {
            goto fail;
        }

        if ((PyDict_GetItem(fields, name) != NULL)
             || (title
                 && PyBaseString_Check(title)
                 && (PyDict_GetItem(fields, title) != NULL))) {
#if defined(NPY_PY3K)
            name = PyUnicode_AsUTF8String(name);
#endif
            PyErr_Format(PyExc_ValueError,
                    "field '%s' occurs more than once", PyString_AsString(name));
#if defined(NPY_PY3K)
            Py_DECREF(name);
#endif
            Py_DECREF(conv);
            goto fail;
        }
        dtypeflags |= (conv->flags & NPY_FROM_FIELDS);
        if (align) {
            int _align;

            _align = conv->alignment;
            if (_align > 1) {
                totalsize = NPY_NEXT_ALIGNED_OFFSET(totalsize, _align);
            }
            maxalign = PyArray_MAX(maxalign, _align);
        }
        tup = PyTuple_New((title == NULL ? 2 : 3));
        PyTuple_SET_ITEM(tup, 0, (PyObject *)conv);
        PyTuple_SET_ITEM(tup, 1, PyInt_FromLong((long) totalsize));

        /*
         * Title can be "meta-data".  Only insert it
         * into the fields dictionary if it is a string
         * and if it is not the same as the name.
         */
        if (title != NULL) {
            Py_INCREF(title);
            PyTuple_SET_ITEM(tup, 2, title);
            PyDict_SetItem(fields, name, tup);
            if (PyBaseString_Check(title)) {
                if (PyDict_GetItem(fields, title) != NULL) {
                    PyErr_SetString(PyExc_ValueError,
                            "title already used as a name or title.");
                    Py_DECREF(tup);
                    goto fail;
                }
                PyDict_SetItem(fields, title, tup);
            }
        }
        else {
            PyDict_SetItem(fields, name, tup);
        }

        totalsize += conv->elsize;
        Py_DECREF(tup);
    }

    if (maxalign > 1) {
        totalsize = NPY_NEXT_ALIGNED_OFFSET(totalsize, maxalign);
    }

    new = PyArray_DescrNewFromType(NPY_VOID);
    if (new == NULL) {
        Py_XDECREF(fields);
        Py_XDECREF(nameslist);
        return NULL;
    }
    new->fields = fields;
    new->names = nameslist;
    new->elsize = totalsize;
    new->flags = dtypeflags;

    /* Structured arrays get a sticky aligned bit */
    if (align) {
        new->flags |= NPY_ALIGNED_STRUCT;
        new->alignment = maxalign;
    }
    return new;

 fail:
    Py_DECREF(fields);
    Py_DECREF(nameslist);
    return NULL;

}

/*
 * a list specifying a data-type can just be
 * a list of formats.  The names for the fields
 * will default to f0, f1, f2, and so forth.
 */
static PyArray_Descr *
_convert_from_list(PyObject *obj, int align)
{
    int n, i;
    int totalsize;
    PyObject *fields;
    PyArray_Descr *conv = NULL;
    PyArray_Descr *new;
    PyObject *key, *tup;
    PyObject *nameslist = NULL;
    int ret;
    int maxalign = 0;
    /* Types with fields need the Python C API for field access */
    char dtypeflags = NPY_NEEDS_PYAPI;

    n = PyList_GET_SIZE(obj);
    /*
     * Ignore any empty string at end which _internal._commastring
     * can produce
     */
    key = PyList_GET_ITEM(obj, n-1);
    if (PyBytes_Check(key) && PyBytes_GET_SIZE(key) == 0) {
        n = n - 1;
    }
    /* End ignore code.*/
    totalsize = 0;
    if (n == 0) {
        return NULL;
    }
    nameslist = PyTuple_New(n);
    if (!nameslist) {
        return NULL;
    }
    fields = PyDict_New();
    for (i = 0; i < n; i++) {
        tup = PyTuple_New(2);
        key = PyUString_FromFormat("f%d", i);
        if (align) {
            ret = PyArray_DescrAlignConverter(PyList_GET_ITEM(obj, i), &conv);
        }
        else {
            ret = PyArray_DescrConverter(PyList_GET_ITEM(obj, i), &conv);
        }
        if (ret == NPY_FAIL) {
            Py_DECREF(tup);
            Py_DECREF(key);
            goto fail;
        }
        dtypeflags |= (conv->flags & NPY_FROM_FIELDS);
        PyTuple_SET_ITEM(tup, 0, (PyObject *)conv);
        if (align) {
            int _align;

            _align = conv->alignment;
            if (_align > 1) {
                totalsize = NPY_NEXT_ALIGNED_OFFSET(totalsize, _align);
            }
            maxalign = PyArray_MAX(maxalign, _align);
        }
        PyTuple_SET_ITEM(tup, 1, PyInt_FromLong((long) totalsize));
        PyDict_SetItem(fields, key, tup);
        Py_DECREF(tup);
        PyTuple_SET_ITEM(nameslist, i, key);
        totalsize += conv->elsize;
    }
    new = PyArray_DescrNewFromType(NPY_VOID);
    new->fields = fields;
    new->names = nameslist;
    new->flags = dtypeflags;
    if (maxalign > 1) {
        totalsize = NPY_NEXT_ALIGNED_OFFSET(totalsize, maxalign);
    }
    /* Structured arrays get a sticky aligned bit */
    if (align) {
        new->flags |= NPY_ALIGNED_STRUCT;
        new->alignment = maxalign;
    }
    new->elsize = totalsize;
    return new;

 fail:
    Py_DECREF(nameslist);
    Py_DECREF(fields);
    return NULL;
}


/*
 * comma-separated string
 * this is the format developed by the numarray records module and implemented
 * by the format parser in that module this is an alternative implementation
 * found in the _internal.py file patterned after that one -- the approach is
 * to try to convert to a list (with tuples if any repeat information is
 * present) and then call the _convert_from_list)
 *
 * TODO: Calling Python from C like this in critical-path code is not
 *       a good idea. This should all be converted to C code.
 */
static PyArray_Descr *
_convert_from_commastring(PyObject *obj, int align)
{
    PyObject *listobj;
    PyArray_Descr *res;
    PyObject *_numpy_internal;

    if (!PyBytes_Check(obj)) {
        return NULL;
    }
    _numpy_internal = PyImport_ImportModule("numpy.core._internal");
    if (_numpy_internal == NULL) {
        return NULL;
    }
    listobj = PyObject_CallMethod(_numpy_internal, "_commastring", "O", obj);
    Py_DECREF(_numpy_internal);
    if (listobj == NULL) {
        return NULL;
    }
    if (!PyList_Check(listobj) || PyList_GET_SIZE(listobj) < 1) {
        PyErr_SetString(PyExc_RuntimeError,
                "_commastring is not returning a list with len >= 1");
        Py_DECREF(listobj);
        return NULL;
    }
    if (PyList_GET_SIZE(listobj) == 1) {
        int retcode;
        retcode = PyArray_DescrConverter(PyList_GET_ITEM(listobj, 0),
                                                &res);
        if (retcode == NPY_FAIL) {
            res = NULL;
        }
    }
    else {
        res = _convert_from_list(listobj, align);
    }
    Py_DECREF(listobj);
    if (!res && !PyErr_Occurred()) {
        PyErr_SetString(PyExc_ValueError,
                "invalid data-type");
        return NULL;
    }
    return res;
}

static int
_is_tuple_of_integers(PyObject *obj)
{
    int i;

    if (!PyTuple_Check(obj)) {
        return 0;
    }
    for (i = 0; i < PyTuple_GET_SIZE(obj); i++) {
        if (!PyArray_IsIntegerScalar(PyTuple_GET_ITEM(obj, i))) {
            return 0;
        }
    }
    return 1;
}

/*
 * helper function for _use_inherit to disallow dtypes of the form
 * (old_dtype, new_dtype) where either of the dtypes contains python
 * objects - these dtypes are not useful and can be a source of segfaults,
 * when an attempt is made to interpret a python object as a different dtype
 * or vice versa
 * an exception is made for dtypes of the form ('O', [('name', 'O')]), which
 * people have been using to add a field to an object array without fields
 */
static int
invalid_union_object_dtype(PyArray_Descr *new, PyArray_Descr *conv)
{
    PyObject *name, *tup;
    PyArray_Descr *dtype;

    if (!PyDataType_REFCHK(new) && !PyDataType_REFCHK(conv)) {
        return 0;
    }
    if (PyDataType_HASFIELDS(new) || new->kind != 'O') {
        goto fail;
    }
    if (!PyDataType_HASFIELDS(conv) || PyTuple_GET_SIZE(conv->names) != 1) {
        goto fail;
    }
    name = PyTuple_GET_ITEM(conv->names, 0);
    if (name == NULL) {
        return -1;
    }
    tup = PyDict_GetItem(conv->fields, name);
    if (tup == NULL) {
        return -1;
    }
    dtype = (PyArray_Descr *)PyTuple_GET_ITEM(tup, 0);
    if (dtype == NULL) {
        return -1;
    }
    if (dtype->kind != 'O') {
        goto fail;
    }
    return 0;

fail:
    PyErr_SetString(PyExc_ValueError,
            "dtypes of the form (old_dtype, new_dtype) containing the object "
            "dtype are not supported");
    return -1;
}

/*
 * A tuple type would be either (generic typeobject, typesize)
 * or (fixed-length data-type, shape)
 *
 * or (inheriting data-type, new-data-type)
 * The new data-type must have the same itemsize as the inheriting data-type
 * unless the latter is 0
 *
 * Thus (int32, {'real':(int16,0),'imag',(int16,2)})
 *
 * is one way to specify a descriptor that will give
 * a['real'] and a['imag'] to an int32 array.
 *
 * leave type reference alone
 */
static PyArray_Descr *
_use_inherit(PyArray_Descr *type, PyObject *newobj, int *errflag)
{
    PyArray_Descr *new;
    PyArray_Descr *conv;

    *errflag = 0;
    if (PyArray_IsScalar(newobj, Integer)
            || _is_tuple_of_integers(newobj)
            || !PyArray_DescrConverter(newobj, &conv)) {
        return NULL;
    }
    *errflag = 1;
    new = PyArray_DescrNew(type);
    if (new == NULL) {
        goto fail;
    }
    if (PyDataType_ISUNSIZED(new)) {
        new->elsize = conv->elsize;
    }
    else if (new->elsize != conv->elsize) {
        PyErr_SetString(PyExc_ValueError,
                "mismatch in size of old and new data-descriptor");
        Py_DECREF(new);
        goto fail;
    }
    else if (invalid_union_object_dtype(new, conv)) {
        Py_DECREF(new);
        goto fail;
    }

    if (PyDataType_HASFIELDS(conv)) {
        Py_XDECREF(new->fields);
        new->fields = conv->fields;
        Py_XINCREF(new->fields);

        Py_XDECREF(new->names);
        new->names = conv->names;
        Py_XINCREF(new->names);
    }
    if (conv->metadata != NULL) {
        Py_XDECREF(new->metadata);
        new->metadata = conv->metadata;
        Py_XINCREF(new->metadata);
    }
    new->flags = conv->flags;
    Py_DECREF(conv);
    *errflag = 0;
    return new;

 fail:
    Py_DECREF(conv);
    return NULL;
}

/*
 * Validates that any field of the structured array 'dtype' which has
 * the NPY_ITEM_HASOBJECT flag set does not overlap with another field.
 *
 * This algorithm is worst case O(n^2). It could be done with a sort
 * and sweep algorithm, but the structured dtype representation is
 * rather ugly right now, so writing something better can wait until
 * that representation is made sane.
 *
 * Returns 0 on success, -1 if an exception is raised.
 */
static int
validate_object_field_overlap(PyArray_Descr *dtype)
{
    PyObject *names, *fields, *key, *tup, *title;
    Py_ssize_t i, j, names_size;
    PyArray_Descr *fld_dtype, *fld2_dtype;
    int fld_offset, fld2_offset;

    /* Get some properties from the dtype */
    names = dtype->names;
    names_size = PyTuple_GET_SIZE(names);
    fields = dtype->fields;

    for (i = 0; i < names_size; ++i) {
        key = PyTuple_GET_ITEM(names, i);
        if (key == NULL) {
            return -1;
        }
        tup = PyDict_GetItem(fields, key);
        if (tup == NULL) {
            return -1;
        }
        if (!PyArg_ParseTuple(tup, "Oi|O", &fld_dtype, &fld_offset, &title)) {
            return -1;
        }

        /* If this field has objects, check for overlaps */
        if (PyDataType_REFCHK(fld_dtype)) {
            for (j = 0; j < names_size; ++j) {
                if (i != j) {
                    key = PyTuple_GET_ITEM(names, j);
                    if (key == NULL) {
                        return -1;
                    }
                    tup = PyDict_GetItem(fields, key);
                    if (tup == NULL) {
                        return -1;
                    }
                    if (!PyArg_ParseTuple(tup, "Oi|O", &fld2_dtype,
                                                &fld2_offset, &title)) {
                        return -1;
                    }
                    /* Raise an exception if it overlaps */
                    if (fld_offset < fld2_offset + fld2_dtype->elsize &&
                                fld2_offset < fld_offset + fld_dtype->elsize) {
                        PyErr_SetString(PyExc_TypeError,
                                "Cannot create a NumPy dtype with overlapping "
                                "object fields");
                        return -1;
                    }
                }
            }
        }
    }

    /* It passed all the overlap tests */
    return 0;
}

/*
 * a dictionary specifying a data-type
 * must have at least two and up to four
 * keys These must all be sequences of the same length.
 *
 * can also have an additional key called "metadata" which can be any dictionary
 *
 * "names" --- field names
 * "formats" --- the data-type descriptors for the field.
 *
 * Optional:
 *
 * "offsets" --- integers indicating the offset into the
 * record of the start of the field.
 * if not given, then "consecutive offsets"
 * will be assumed and placed in the dictionary.
 *
 * "titles" --- Allows the use of an additional key
 * for the fields dictionary.(if these are strings
 * or unicode objects) or
 * this can also be meta-data to
 * be passed around with the field description.
 *
 * Attribute-lookup-based field names merely has to query the fields
 * dictionary of the data-descriptor.  Any result present can be used
 * to return the correct field.
 *
 * So, the notion of what is a name and what is a title is really quite
 * arbitrary.
 *
 * What does distinguish a title, however, is that if it is not None,
 * it will be placed at the end of the tuple inserted into the
 * fields dictionary.and can therefore be used to carry meta-data around.
 *
 * If the dictionary does not have "names" and "formats" entries,
 * then it will be checked for conformity and used directly.
 */
static PyArray_Descr *
_use_fields_dict(PyObject *obj, int align)
{
    PyObject *_numpy_internal;
    PyArray_Descr *res;

    _numpy_internal = PyImport_ImportModule("numpy.core._internal");
    if (_numpy_internal == NULL) {
        return NULL;
    }
    res = (PyArray_Descr *)PyObject_CallMethod(_numpy_internal,
            "_usefields", "Oi", obj, align);
    Py_DECREF(_numpy_internal);
    return res;
}

/*
 * Creates a struct dtype object from a Python dictionary.
 */
static PyArray_Descr *
_convert_from_dict(PyObject *obj, int align)
{
    PyArray_Descr *new;
    PyObject *fields = NULL;
    PyObject *names  = NULL;
    PyObject *offsets= NULL;
    PyObject *descrs = NULL;
    PyObject *titles = NULL;
    PyObject *metadata, *tmp;
    int n, i;
    int totalsize, itemsize;
    int maxalign = 0;
    /* Types with fields need the Python C API for field access */
    char dtypeflags = NPY_NEEDS_PYAPI;
    int has_out_of_order_fields = 0;

    fields = PyDict_New();
    if (fields == NULL) {
        return (PyArray_Descr *)PyErr_NoMemory();
    }
    /*
     * Use PyMapping_GetItemString to support dictproxy objects as well.
     */
    names = PyMapping_GetItemString(obj, "names");
    if (names == NULL) {
        Py_DECREF(fields);
        /* XXX should check this is a KeyError */
        PyErr_Clear();
        return _use_fields_dict(obj, align);
    }
    descrs = PyMapping_GetItemString(obj, "formats");
    if (descrs == NULL) {
        Py_DECREF(fields);
        /* XXX should check this is a KeyError */
        PyErr_Clear();
        Py_DECREF(names);
        return _use_fields_dict(obj, align);
    }
    n = PyObject_Length(names);
    offsets = PyMapping_GetItemString(obj, "offsets");
    if (!offsets) {
        PyErr_Clear();
    }
    titles = PyMapping_GetItemString(obj, "titles");
    if (!titles) {
        PyErr_Clear();
    }

    if ((n > PyObject_Length(descrs))
        || (offsets && (n > PyObject_Length(offsets)))
        || (titles && (n > PyObject_Length(titles)))) {
        PyErr_SetString(PyExc_ValueError,
                "'names', 'formats', 'offsets', and 'titles' dict "
                "entries must have the same length");
        goto fail;
    }

    /*
     * If a property 'aligned' is in the dict, it overrides the align flag
     * to be True if it not already true.
     */
    tmp = PyMapping_GetItemString(obj, "aligned");
    if (tmp == NULL) {
        PyErr_Clear();
    } else {
        if (tmp == Py_True) {
            align = 1;
        }
        else if (tmp != Py_False) {
            Py_DECREF(tmp);
            PyErr_SetString(PyExc_ValueError,
                    "NumPy dtype descriptor includes 'aligned' entry, "
                    "but its value is neither True nor False");
            goto fail;
        }
        Py_DECREF(tmp);
    }

    totalsize = 0;
    for (i = 0; i < n; i++) {
        PyObject *tup, *descr, *ind, *title, *name, *off;
        int len, ret, _align = 1;
        PyArray_Descr *newdescr;

        /* Build item to insert (descr, offset, [title])*/
        len = 2;
        title = NULL;
        ind = PyInt_FromLong(i);
        if (titles) {
            title=PyObject_GetItem(titles, ind);
            if (title && title != Py_None) {
                len = 3;
            }
            else {
                Py_XDECREF(title);
            }
            PyErr_Clear();
        }
        tup = PyTuple_New(len);
        descr = PyObject_GetItem(descrs, ind);
        if (!descr) {
            Py_DECREF(tup);
            Py_DECREF(ind);
            goto fail;
        }
        if (align) {
            ret = PyArray_DescrAlignConverter(descr, &newdescr);
        }
        else {
            ret = PyArray_DescrConverter(descr, &newdescr);
        }
        Py_DECREF(descr);
        if (ret == NPY_FAIL) {
            Py_DECREF(tup);
            Py_DECREF(ind);
            goto fail;
        }
        PyTuple_SET_ITEM(tup, 0, (PyObject *)newdescr);
        if (align) {
            _align = newdescr->alignment;
            maxalign = PyArray_MAX(maxalign,_align);
        }
        if (offsets) {
            long offset;
            off = PyObject_GetItem(offsets, ind);
            if (!off) {
                Py_DECREF(tup);
                Py_DECREF(ind);
                goto fail;
            }
            offset = PyArray_PyIntAsInt(off);
            if (error_converting(offset)) {
                Py_DECREF(off);
                Py_DECREF(tup);
                Py_DECREF(ind);
                goto fail;
            }
            Py_DECREF(off);
            if (offset < 0) {
                PyErr_Format(PyExc_ValueError, "offset %d cannot be negative",
                             (int)offset);
                Py_DECREF(tup);
                Py_DECREF(ind);
                goto fail;
            }

            PyTuple_SET_ITEM(tup, 1, PyInt_FromLong(offset));
            /* Flag whether the fields are specified out of order */
            if (offset < totalsize) {
                has_out_of_order_fields = 1;
            }
            /* If align=True, enforce field alignment */
            if (align && offset % newdescr->alignment != 0) {
                PyErr_Format(PyExc_ValueError,
                        "offset %d for NumPy dtype with fields is "
                        "not divisible by the field alignment %d "
                        "with align=True",
                        (int)offset, (int)newdescr->alignment);
                ret = NPY_FAIL;
            }
            else if (offset + newdescr->elsize > totalsize) {
                totalsize = offset + newdescr->elsize;
            }
        }
        else {
            if (align && _align > 1) {
                totalsize = NPY_NEXT_ALIGNED_OFFSET(totalsize, _align);
            }
            PyTuple_SET_ITEM(tup, 1, PyInt_FromLong(totalsize));
            totalsize += newdescr->elsize;
        }
        if (ret == NPY_FAIL) {
            Py_DECREF(ind);
            Py_DECREF(tup);
            goto fail;
        }
        if (len == 3) {
            PyTuple_SET_ITEM(tup, 2, title);
        }
        name = PyObject_GetItem(names, ind);
        Py_DECREF(ind);
        if (!name) {
            Py_DECREF(tup);
            goto fail;
        }
        if (!PyBaseString_Check(name)) {
            PyErr_SetString(PyExc_ValueError,
                    "field names must be strings");
            Py_DECREF(tup);
            goto fail;
        }

        /* Insert into dictionary */
        if (PyDict_GetItem(fields, name) != NULL) {
            PyErr_SetString(PyExc_ValueError,
                    "name already used as a name or title");
            Py_DECREF(tup);
            goto fail;
        }
        PyDict_SetItem(fields, name, tup);
        Py_DECREF(name);
        if (len == 3) {
            if (PyBaseString_Check(title)) {
                if (PyDict_GetItem(fields, title) != NULL) {
                    PyErr_SetString(PyExc_ValueError,
                            "title already used as a name or title.");
                    Py_DECREF(tup);
                    goto fail;
                }
                PyDict_SetItem(fields, title, tup);
            }
        }
        Py_DECREF(tup);
        if (ret == NPY_FAIL) {
            goto fail;
        }
        dtypeflags |= (newdescr->flags & NPY_FROM_FIELDS);
    }

    new = PyArray_DescrNewFromType(NPY_VOID);
    if (new == NULL) {
        goto fail;
    }
    if (maxalign > 1) {
        totalsize = NPY_NEXT_ALIGNED_OFFSET(totalsize, maxalign);
    }
    if (align) {
        new->alignment = maxalign;
    }
    new->elsize = totalsize;
    if (!PyTuple_Check(names)) {
        Py_SETREF(names, PySequence_Tuple(names));
        if (names == NULL) {
            Py_DECREF(new);
            goto fail;
        }
    }
    new->names = names;
    new->fields = fields;
    new->flags = dtypeflags;
    /* new takes responsibility for DECREFing names, fields */
    names = NULL;
    fields = NULL;

    /*
     * If the fields weren't in order, and there was an OBJECT type,
     * need to verify that no OBJECT types overlap with something else.
     */
    if (has_out_of_order_fields && PyDataType_REFCHK(new)) {
        if (validate_object_field_overlap(new) < 0) {
            Py_DECREF(new);
            goto fail;
        }
    }

    /* Structured arrays get a sticky aligned bit */
    if (align) {
        new->flags |= NPY_ALIGNED_STRUCT;
    }

    /* Override the itemsize if provided */
    tmp = PyMapping_GetItemString(obj, "itemsize");
    if (tmp == NULL) {
        PyErr_Clear();
    } else {
        itemsize = (int)PyArray_PyIntAsInt(tmp);
        Py_DECREF(tmp);
        if (error_converting(itemsize)) {
            Py_DECREF(new);
            goto fail;
        }
        /* Make sure the itemsize isn't made too small */
        if (itemsize < new->elsize) {
            PyErr_Format(PyExc_ValueError,
                    "NumPy dtype descriptor requires %d bytes, "
                    "cannot override to smaller itemsize of %d",
                    (int)new->elsize, (int)itemsize);
            Py_DECREF(new);
            goto fail;
        }
        /* If align is set, make sure the alignment divides into the size */
        if (align && itemsize % new->alignment != 0) {
            PyErr_Format(PyExc_ValueError,
                    "NumPy dtype descriptor requires alignment of %d bytes, "
                    "which is not divisible into the specified itemsize %d",
                    (int)new->alignment, (int)itemsize);
            Py_DECREF(new);
            goto fail;
        }
        /* Set the itemsize */
        new->elsize = itemsize;
    }

    /* Add the metadata if provided */
    metadata = PyMapping_GetItemString(obj, "metadata");

    if (metadata == NULL) {
        PyErr_Clear();
    }
    else if (new->metadata == NULL) {
        new->metadata = metadata;
    }
    else {
        int ret = PyDict_Merge(new->metadata, metadata, 0);
        Py_DECREF(metadata);
        if (ret < 0) {
            Py_DECREF(new);
            goto fail;
        }
    }

    Py_XDECREF(fields);
    Py_XDECREF(names);
    Py_XDECREF(descrs);
    Py_XDECREF(offsets);
    Py_XDECREF(titles);
    return new;

 fail:
    Py_XDECREF(fields);
    Py_XDECREF(names);
    Py_XDECREF(descrs);
    Py_XDECREF(offsets);
    Py_XDECREF(titles);
    return NULL;
}


/*NUMPY_API*/
NPY_NO_EXPORT PyArray_Descr *
PyArray_DescrNewFromType(int type_num)
{
    PyArray_Descr *old;
    PyArray_Descr *new;

    old = PyArray_DescrFromType(type_num);
    new = PyArray_DescrNew(old);
    Py_DECREF(old);
    return new;
}

/*NUMPY_API
 * Get typenum from an object -- None goes to NULL
 */
NPY_NO_EXPORT int
PyArray_DescrConverter2(PyObject *obj, PyArray_Descr **at)
{
    if (obj == Py_None) {
        *at = NULL;
        return NPY_SUCCEED;
    }
    else {
        return PyArray_DescrConverter(obj, at);
    }
}

/*NUMPY_API
 * Get typenum from an object -- None goes to NPY_DEFAULT_TYPE
 * This function takes a Python object representing a type and converts it
 * to a the correct PyArray_Descr * structure to describe the type.
 *
 * Many objects can be used to represent a data-type which in NumPy is
 * quite a flexible concept.
 *
 * This is the central code that converts Python objects to
 * Type-descriptor objects that are used throughout numpy.
 *
 * Returns a new reference in *at, but the returned should not be
 * modified as it may be one of the canonical immutable objects or
 * a reference to the input obj.
 */
NPY_NO_EXPORT int
PyArray_DescrConverter(PyObject *obj, PyArray_Descr **at)
{
    int check_num = NPY_NOTYPE + 10;
    PyObject *item;
    int elsize = 0;
    char endian = '=';

    *at = NULL;

    /* default */
    if (obj == Py_None) {
        *at = PyArray_DescrFromType(NPY_DEFAULT_TYPE);
        return NPY_SUCCEED;
    }

    if (PyArray_DescrCheck(obj)) {
        *at = (PyArray_Descr *)obj;
        Py_INCREF(*at);
        return NPY_SUCCEED;
    }

    if (PyType_Check(obj)) {
        if (PyType_IsSubtype((PyTypeObject *)obj, &PyGenericArrType_Type)) {
            *at = PyArray_DescrFromTypeObject(obj);
            return (*at) ? NPY_SUCCEED : NPY_FAIL;
        }
        check_num = NPY_OBJECT;
#if !defined(NPY_PY3K)
        if (obj == (PyObject *)(&PyInt_Type)) {
            check_num = NPY_LONG;
        }
        else if (obj == (PyObject *)(&PyLong_Type)) {
            check_num = NPY_LONGLONG;
        }
#else
        if (obj == (PyObject *)(&PyLong_Type)) {
            check_num = NPY_LONG;
        }
#endif
        else if (obj == (PyObject *)(&PyFloat_Type)) {
            check_num = NPY_DOUBLE;
        }
        else if (obj == (PyObject *)(&PyComplex_Type)) {
            check_num = NPY_CDOUBLE;
        }
        else if (obj == (PyObject *)(&PyBool_Type)) {
            check_num = NPY_BOOL;
        }
        else if (obj == (PyObject *)(&PyBytes_Type)) {
            check_num = NPY_STRING;
        }
        else if (obj == (PyObject *)(&PyUnicode_Type)) {
            check_num = NPY_UNICODE;
        }
#if defined(NPY_PY3K)
        else if (obj == (PyObject *)(&PyMemoryView_Type)) {
#else
        else if (obj == (PyObject *)(&PyBuffer_Type)) {
#endif
            check_num = NPY_VOID;
        }
        else {
            *at = _arraydescr_from_dtype_attr(obj);
            if (*at) {
                return NPY_SUCCEED;
            }

            /*
             * Note: this comes after _arraydescr_from_dtype_attr because the ctypes
             * type might override the dtype if numpy does not otherwise
             * support it.
             */
            if (npy_ctypes_check((PyTypeObject *)obj)) {
                *at = _arraydescr_from_ctypes_type((PyTypeObject *)obj);
                return *at ? NPY_SUCCEED : NPY_FAIL;
            }
        }
        goto finish;
    }

    /* or a typecode string */

    if (PyUnicode_Check(obj)) {
        /* Allow unicode format strings: convert to bytes */
        int retval;
        PyObject *obj2;
        obj2 = PyUnicode_AsASCIIString(obj);
        if (obj2 == NULL) {
            /* Convert the exception into a TypeError */
            PyObject *err = PyErr_Occurred();
            if (PyErr_GivenExceptionMatches(err, PyExc_UnicodeEncodeError)) {
                PyErr_SetString(PyExc_TypeError,
                        "data type not understood");
            }
            return NPY_FAIL;
        }
        retval = PyArray_DescrConverter(obj2, at);
        Py_DECREF(obj2);
        return retval;
    }

    if (PyBytes_Check(obj)) {
        char *type = NULL;
        Py_ssize_t len = 0;

        /* Check for a string typecode. */
        if (PyBytes_AsStringAndSize(obj, &type, &len) < 0) {
            goto error;
        }

        /* Empty string is invalid */
        if (len == 0) {
            goto fail;
        }

        /* check for commas present or first (or second) element a digit */
        if (_check_for_commastring(type, len)) {
            *at = _convert_from_commastring(obj, 0);
            return (*at) ? NPY_SUCCEED : NPY_FAIL;
        }

        /* Process the endian character. '|' is replaced by '='*/
        switch (type[0]) {
            case '>':
            case '<':
            case '=':
                endian = type[0];
                ++type;
                --len;
                break;

            case '|':
                endian = '=';
                ++type;
                --len;
                break;
        }

        /* Just an endian character is invalid */
        if (len == 0) {
            goto fail;
        }

        /* Check for datetime format */
        if (is_datetime_typestr(type, len)) {
            *at = parse_dtype_from_datetime_typestr(type, len);
            if (*at == NULL) {
                return NPY_FAIL;
            }
            /* *at has byte order '=' at this point */
            if (!PyArray_ISNBO(endian)) {
                (*at)->byteorder = endian;
            }
            return NPY_SUCCEED;
        }

        /* A typecode like 'd' */
        if (len == 1) {
            check_num = type[0];
        }
        /* A kind + size like 'f8' */
        else {
            char *typeend = NULL;
            int kind;

            /* Parse the integer, make sure it's the rest of the string */
            elsize = (int)strtol(type + 1, &typeend, 10);
            if (typeend - type == len) {

                kind = type[0];
                switch (kind) {
                    case NPY_STRINGLTR:
                    case NPY_STRINGLTR2:
                        check_num = NPY_STRING;
                        break;

                    /*
                     * When specifying length of UNICODE
                     * the number of characters is given to match
                     * the STRING interface.  Each character can be
                     * more than one byte and itemsize must be
                     * the number of bytes.
                     */
                    case NPY_UNICODELTR:
                        check_num = NPY_UNICODE;
                        elsize <<= 2;
                        break;

                    case NPY_VOIDLTR:
                        check_num = NPY_VOID;
                        break;

                    default:
                        if (elsize == 0) {
                            check_num = NPY_NOTYPE+10;
                        }
                        /* Support for generic processing c8, i4, f8, etc...*/
                        else {
                            check_num = PyArray_TypestrConvert(elsize, kind);
                            if (check_num == NPY_NOTYPE) {
                                check_num += 10;
                            }
                            elsize = 0;
                        }
                }
            }
        }
    }
    else if (PyTuple_Check(obj)) {
        /* or a tuple */
        *at = _convert_from_tuple(obj, 0);
        if (*at == NULL){
            if (PyErr_Occurred()) {
                return NPY_FAIL;
            }
            goto fail;
        }
        return NPY_SUCCEED;
    }
    else if (PyList_Check(obj)) {
        /* or a list */
        *at = _convert_from_array_descr(obj,0);
        if (*at == NULL) {
            if (PyErr_Occurred()) {
                return NPY_FAIL;
            }
            goto fail;
        }
        return NPY_SUCCEED;
    }
    else if (PyDict_Check(obj) || PyDictProxy_Check(obj)) {
        /* or a dictionary */
        *at = _convert_from_dict(obj,0);
        if (*at == NULL) {
            if (PyErr_Occurred()) {
                return NPY_FAIL;
            }
            goto fail;
        }
        return NPY_SUCCEED;
    }
    else if (PyArray_Check(obj)) {
        goto fail;
    }
    else {
        *at = _arraydescr_from_dtype_attr(obj);
        if (*at) {
            return NPY_SUCCEED;
        }
        if (PyErr_Occurred()) {
            return NPY_FAIL;
        }

        /*
         * Note: this comes after _arraydescr_from_dtype_attr because the ctypes
         * type might override the dtype if numpy does not otherwise
         * support it.
         */
        if (npy_ctypes_check(Py_TYPE(obj))) {
            *at = _arraydescr_from_ctypes_type(Py_TYPE(obj));
            return *at ? NPY_SUCCEED : NPY_FAIL;
        }
        goto fail;
    }
    if (PyErr_Occurred()) {
        goto fail;
    }

finish:
    if ((check_num == NPY_NOTYPE + 10) ||
            (*at = PyArray_DescrFromType(check_num)) == NULL) {
        PyErr_Clear();
        /* Now check to see if the object is registered in typeDict */
        if (typeDict != NULL) {
            item = PyDict_GetItem(typeDict, obj);
#if defined(NPY_PY3K)
            if (!item && PyBytes_Check(obj)) {
                PyObject *tmp;
                tmp = PyUnicode_FromEncodedObject(obj, "ascii", "strict");
                if (tmp != NULL) {
                    item = PyDict_GetItem(typeDict, tmp);
                    Py_DECREF(tmp);
                }
            }
#endif
            if (item) {
                /* Check for a deprecated Numeric-style typecode */
                if (PyBytes_Check(obj)) {
                    char *type = NULL;
                    Py_ssize_t len = 0;
                    char *dep_tps[] = {"Bool", "Complex", "Float", "Int",
                                       "Object0", "String0", "Timedelta64",
                                       "Unicode0", "UInt", "Void0"};
                    int ndep_tps = sizeof(dep_tps) / sizeof(dep_tps[0]);
                    int i;

                    if (PyBytes_AsStringAndSize(obj, &type, &len) < 0) {
                        goto error;
                    }
                    for (i = 0; i < ndep_tps; ++i) {
                        char *dep_tp = dep_tps[i];

                        if (strncmp(type, dep_tp, strlen(dep_tp)) == 0) {
                            if (DEPRECATE("Numeric-style type codes are "
                                          "deprecated and will result in "
                                          "an error in the future.") < 0) {
                                goto fail;
                            }
                        }
                    }
                }
                return PyArray_DescrConverter(item, at);
            }
        }
        goto fail;
    }

    if (PyDataType_ISUNSIZED(*at) && (*at)->elsize != elsize) {
        PyArray_DESCR_REPLACE(*at);
        if (*at == NULL) {
            goto error;
        }
        (*at)->elsize = elsize;
    }
    if (endian != '=' && PyArray_ISNBO(endian)) {
        endian = '=';
    }
    if (endian != '=' && (*at)->byteorder != '|'
        && (*at)->byteorder != endian) {
        PyArray_DESCR_REPLACE(*at);
        if (*at == NULL) {
            goto error;
        }
        (*at)->byteorder = endian;
    }
    return NPY_SUCCEED;

fail:
    if (PyBytes_Check(obj)) {
        PyErr_Format(PyExc_TypeError,
                "data type \"%s\" not understood", PyBytes_AS_STRING(obj));
    }
    else {
        PyErr_SetString(PyExc_TypeError,
                "data type not understood");
    }

error:
    *at = NULL;
    return NPY_FAIL;
}

/** Array Descr Objects for dynamic types **/

/*
 * There are some statically-defined PyArray_Descr objects corresponding
 * to the basic built-in types.
 * These can and should be DECREF'd and INCREF'd as appropriate, anyway.
 * If a mistake is made in reference counting, deallocation on these
 * builtins will be attempted leading to problems.
 *
 * This lets us deal with all PyArray_Descr objects using reference
 * counting (regardless of whether they are statically or dynamically
 * allocated).
 */

/*NUMPY_API
 * base cannot be NULL
 */
NPY_NO_EXPORT PyArray_Descr *
PyArray_DescrNew(PyArray_Descr *base)
{
    PyArray_Descr *newdescr = PyObject_New(PyArray_Descr, &PyArrayDescr_Type);

    if (newdescr == NULL) {
        return NULL;
    }
    /* Don't copy PyObject_HEAD part */
    memcpy((char *)newdescr + sizeof(PyObject),
           (char *)base + sizeof(PyObject),
           sizeof(PyArray_Descr) - sizeof(PyObject));

    /*
     * The c_metadata has a by-value ownership model, need to clone it
     * (basically a deep copy, but the auxdata clone function has some
     * flexibility still) so the new PyArray_Descr object owns
     * a copy of the data. Having both 'base' and 'newdescr' point to
     * the same auxdata pointer would cause a double-free of memory.
     */
    if (base->c_metadata != NULL) {
        newdescr->c_metadata = NPY_AUXDATA_CLONE(base->c_metadata);
        if (newdescr->c_metadata == NULL) {
            PyErr_NoMemory();
            /* TODO: This seems wrong, as the old fields get decref'd? */
            Py_DECREF(newdescr);
            return NULL;
        }
    }

    if (newdescr->fields == Py_None) {
        newdescr->fields = NULL;
    }
    Py_XINCREF(newdescr->fields);
    Py_XINCREF(newdescr->names);
    if (newdescr->subarray) {
        newdescr->subarray = PyArray_malloc(sizeof(PyArray_ArrayDescr));
        if (newdescr->subarray == NULL) {
            Py_DECREF(newdescr);
            return (PyArray_Descr *)PyErr_NoMemory();
        }
        memcpy(newdescr->subarray, base->subarray, sizeof(PyArray_ArrayDescr));
        Py_INCREF(newdescr->subarray->shape);
        Py_INCREF(newdescr->subarray->base);
    }
    Py_XINCREF(newdescr->typeobj);
    Py_XINCREF(newdescr->metadata);
    newdescr->hash = -1;

    return newdescr;
}

/*
 * should never be called for builtin-types unless
 * there is a reference-count problem
 */
static void
arraydescr_dealloc(PyArray_Descr *self)
{
    if (self->fields == Py_None) {
        fprintf(stderr, "*** Reference count error detected: \n" \
                "an attempt was made to deallocate %d (%c) ***\n",
                self->type_num, self->type);
        Py_INCREF(self);
        Py_INCREF(self);
        return;
    }
    _dealloc_cached_buffer_info((PyObject*)self);
    Py_XDECREF(self->typeobj);
    Py_XDECREF(self->names);
    Py_XDECREF(self->fields);
    if (self->subarray) {
        Py_XDECREF(self->subarray->shape);
        Py_DECREF(self->subarray->base);
        PyArray_free(self->subarray);
    }
    Py_XDECREF(self->metadata);
    NPY_AUXDATA_FREE(self->c_metadata);
    self->c_metadata = NULL;
    Py_TYPE(self)->tp_free((PyObject *)self);
}

/*
 * we need to be careful about setting attributes because these
 * objects are pointed to by arrays that depend on them for interpreting
 * data.  Currently no attributes of data-type objects can be set
 * directly except names.
 */
static PyMemberDef arraydescr_members[] = {
    {"type",
        T_OBJECT, offsetof(PyArray_Descr, typeobj), READONLY, NULL},
    {"kind",
        T_CHAR, offsetof(PyArray_Descr, kind), READONLY, NULL},
    {"char",
        T_CHAR, offsetof(PyArray_Descr, type), READONLY, NULL},
    {"num",
        T_INT, offsetof(PyArray_Descr, type_num), READONLY, NULL},
    {"byteorder",
        T_CHAR, offsetof(PyArray_Descr, byteorder), READONLY, NULL},
    {"itemsize",
        T_INT, offsetof(PyArray_Descr, elsize), READONLY, NULL},
    {"alignment",
        T_INT, offsetof(PyArray_Descr, alignment), READONLY, NULL},
    {"flags",
        T_BYTE, offsetof(PyArray_Descr, flags), READONLY, NULL},
    {NULL, 0, 0, 0, NULL},
};

static PyObject *
arraydescr_subdescr_get(PyArray_Descr *self)
{
    if (!PyDataType_HASSUBARRAY(self)) {
        Py_RETURN_NONE;
    }
    return Py_BuildValue("OO",
            (PyObject *)self->subarray->base, self->subarray->shape);
}

NPY_NO_EXPORT PyObject *
arraydescr_protocol_typestr_get(PyArray_Descr *self)
{
    char basic_ = self->kind;
    char endian = self->byteorder;
    int size = self->elsize;
    PyObject *ret;

    if (endian == '=') {
        endian = '<';
        if (!PyArray_IsNativeByteOrder(endian)) {
            endian = '>';
        }
    }
    if (self->type_num == NPY_UNICODE) {
        size >>= 2;
    }
    if (self->type_num == NPY_OBJECT) {
        ret = PyUString_FromFormat("%c%c", endian, basic_);
    }
    else {
        ret = PyUString_FromFormat("%c%c%d", endian, basic_, size);
    }
    if (PyDataType_ISDATETIME(self)) {
        PyArray_DatetimeMetaData *meta;

        meta = get_datetime_metadata_from_dtype(self);
        if (meta == NULL) {
            Py_DECREF(ret);
            return NULL;
        }

        ret = append_metastr_to_string(meta, 0, ret);
    }

    return ret;
}

static PyObject *
arraydescr_name_get(PyArray_Descr *self)
{
    /* let python handle this */
    PyObject *_numpy_dtype;
    PyObject *res;
    _numpy_dtype = PyImport_ImportModule("numpy.core._dtype");
    if (_numpy_dtype == NULL) {
        return NULL;
    }
    res = PyObject_CallMethod(_numpy_dtype, "_name_get", "O", self);
    Py_DECREF(_numpy_dtype);
    return res;
}

static PyObject *
arraydescr_base_get(PyArray_Descr *self)
{
    if (!PyDataType_HASSUBARRAY(self)) {
        Py_INCREF(self);
        return (PyObject *)self;
    }
    Py_INCREF(self->subarray->base);
    return (PyObject *)(self->subarray->base);
}

static PyObject *
arraydescr_shape_get(PyArray_Descr *self)
{
    if (!PyDataType_HASSUBARRAY(self)) {
        return PyTuple_New(0);
    }
    assert(PyTuple_Check(self->subarray->shape));
    Py_INCREF(self->subarray->shape);
    return self->subarray->shape;
}

static PyObject *
arraydescr_ndim_get(PyArray_Descr *self)
{
    Py_ssize_t ndim;

    if (!PyDataType_HASSUBARRAY(self)) {
        return PyInt_FromLong(0);
    }

    /*
     * PyTuple_Size has built in check
     * for tuple argument
     */
    ndim = PyTuple_Size(self->subarray->shape);
    return PyInt_FromLong(ndim);
}


NPY_NO_EXPORT PyObject *
arraydescr_protocol_descr_get(PyArray_Descr *self)
{
    PyObject *dobj, *res;
    PyObject *_numpy_internal;

    if (!PyDataType_HASFIELDS(self)) {
        /* get default */
        dobj = PyTuple_New(2);
        if (dobj == NULL) {
            return NULL;
        }
        PyTuple_SET_ITEM(dobj, 0, PyUString_FromString(""));
        PyTuple_SET_ITEM(dobj, 1, arraydescr_protocol_typestr_get(self));
        res = PyList_New(1);
        if (res == NULL) {
            Py_DECREF(dobj);
            return NULL;
        }
        PyList_SET_ITEM(res, 0, dobj);
        return res;
    }

    _numpy_internal = PyImport_ImportModule("numpy.core._internal");
    if (_numpy_internal == NULL) {
        return NULL;
    }
    res = PyObject_CallMethod(_numpy_internal, "_array_descr", "O", self);
    Py_DECREF(_numpy_internal);
    return res;
}

/*
 * returns 1 for a builtin type
 * and 2 for a user-defined data-type descriptor
 * return 0 if neither (i.e. it's a copy of one)
 */
static PyObject *
arraydescr_isbuiltin_get(PyArray_Descr *self)
{
    long val;
    val = 0;
    if (self->fields == Py_None) {
        val = 1;
    }
    if (PyTypeNum_ISUSERDEF(self->type_num)) {
        val = 2;
    }
    return PyInt_FromLong(val);
}

static int
_arraydescr_isnative(PyArray_Descr *self)
{
    if (!PyDataType_HASFIELDS(self)) {
        return PyArray_ISNBO(self->byteorder);
    }
    else {
        PyObject *key, *value, *title = NULL;
        PyArray_Descr *new;
        int offset;
        Py_ssize_t pos = 0;
        while (PyDict_Next(self->fields, &pos, &key, &value)) {
            if NPY_TITLE_KEY(key, value) {
                continue;
            }
            if (!PyArg_ParseTuple(value, "Oi|O", &new, &offset, &title)) {
                return -1;
            }
            if (!_arraydescr_isnative(new)) {
                return 0;
            }
        }
    }
    return 1;
}

/*
 * return Py_True if this data-type descriptor
 * has native byteorder if no fields are defined
 *
 * or if all sub-fields have native-byteorder if
 * fields are defined
 */
static PyObject *
arraydescr_isnative_get(PyArray_Descr *self)
{
    PyObject *ret;
    int retval;
    retval = _arraydescr_isnative(self);
    if (retval == -1) {
        return NULL;
    }
    ret = retval ? Py_True : Py_False;
    Py_INCREF(ret);
    return ret;
}

static PyObject *
arraydescr_isalignedstruct_get(PyArray_Descr *self)
{
    PyObject *ret;
    ret = (self->flags&NPY_ALIGNED_STRUCT) ? Py_True : Py_False;
    Py_INCREF(ret);
    return ret;
}

static PyObject *
arraydescr_fields_get(PyArray_Descr *self)
{
    if (!PyDataType_HASFIELDS(self)) {
        Py_RETURN_NONE;
    }
    return PyDictProxy_New(self->fields);
}

static PyObject *
arraydescr_metadata_get(PyArray_Descr *self)
{
    if (self->metadata == NULL) {
        Py_RETURN_NONE;
    }
    return PyDictProxy_New(self->metadata);
}

static PyObject *
arraydescr_hasobject_get(PyArray_Descr *self)
{
    if (PyDataType_FLAGCHK(self, NPY_ITEM_HASOBJECT)) {
        Py_RETURN_TRUE;
    }
    else {
        Py_RETURN_FALSE;
    }
}

static PyObject *
arraydescr_names_get(PyArray_Descr *self)
{
    if (!PyDataType_HASFIELDS(self)) {
        Py_RETURN_NONE;
    }
    Py_INCREF(self->names);
    return self->names;
}

static int
arraydescr_names_set(PyArray_Descr *self, PyObject *val)
{
    int N = 0;
    int i;
    PyObject *new_names;
    PyObject *new_fields;

    if (val == NULL) {
        PyErr_SetString(PyExc_AttributeError,
                "Cannot delete dtype names attribute");
        return -1;
    }
    if (!PyDataType_HASFIELDS(self)) {
        PyErr_SetString(PyExc_ValueError,
                "there are no fields defined");
        return -1;
    }

    /*
     * FIXME
     *
     * This deprecation has been temporarily removed for the NumPy 1.7
     * release. It should be re-added after the 1.7 branch is done,
     * and a convenience API to replace the typical use-cases for
     * mutable names should be implemented.
     *
     * if (DEPRECATE("Setting NumPy dtype names is deprecated, the dtype "
     *                "will become immutable in a future version") < 0) {
     *     return -1;
     * }
     */

    N = PyTuple_GET_SIZE(self->names);
    if (!PySequence_Check(val) || PyObject_Size((PyObject *)val) != N) {
        PyErr_Format(PyExc_ValueError,
                "must replace all names at once with a sequence of length %d",
                N);
        return -1;
    }
    /* Make sure all entries are strings */
    for (i = 0; i < N; i++) {
        PyObject *item;
        int valid = 1;
        item = PySequence_GetItem(val, i);
        valid = PyUString_Check(item);
        Py_DECREF(item);
        if (!valid) {
            PyErr_Format(PyExc_ValueError,
                    "item #%d of names is of type %s and not string",
                    i, Py_TYPE(item)->tp_name);
            return -1;
        }
    }
    /* Invalidate cached hash value */
    self->hash = -1;
    /* Update dictionary keys in fields */
    new_names = PySequence_Tuple(val);
    new_fields = PyDict_New();
    for (i = 0; i < N; i++) {
        PyObject *key;
        PyObject *item;
        PyObject *new_key;
        int ret;
        key = PyTuple_GET_ITEM(self->names, i);
        /* Borrowed references to item and new_key */
        item = PyDict_GetItem(self->fields, key);
        new_key = PyTuple_GET_ITEM(new_names, i);
        /* Check for duplicates */
        ret = PyDict_Contains(new_fields, new_key);
        if (ret != 0) {
            if (ret < 0) {
                PyErr_Clear();
            }
            PyErr_SetString(PyExc_ValueError, "Duplicate field names given.");
            Py_DECREF(new_names);
            Py_DECREF(new_fields);
            return -1;
        }
        PyDict_SetItem(new_fields, new_key, item);
    }

    /* Replace names */
    Py_DECREF(self->names);
    self->names = new_names;

    /* Replace fields */
    Py_DECREF(self->fields);
    self->fields = new_fields;

    return 0;
}

static PyGetSetDef arraydescr_getsets[] = {
    {"subdtype",
        (getter)arraydescr_subdescr_get,
        NULL, NULL, NULL},
    {"descr",
        (getter)arraydescr_protocol_descr_get,
        NULL, NULL, NULL},
    {"str",
        (getter)arraydescr_protocol_typestr_get,
        NULL, NULL, NULL},
    {"name",
        (getter)arraydescr_name_get,
        NULL, NULL, NULL},
    {"base",
        (getter)arraydescr_base_get,
        NULL, NULL, NULL},
    {"shape",
        (getter)arraydescr_shape_get,
        NULL, NULL, NULL},
    {"ndim",
        (getter)arraydescr_ndim_get,
        NULL, NULL, NULL},
    {"isbuiltin",
        (getter)arraydescr_isbuiltin_get,
        NULL, NULL, NULL},
    {"isnative",
        (getter)arraydescr_isnative_get,
        NULL, NULL, NULL},
    {"isalignedstruct",
        (getter)arraydescr_isalignedstruct_get,
        NULL, NULL, NULL},
    {"fields",
        (getter)arraydescr_fields_get,
        NULL, NULL, NULL},
    {"metadata",
        (getter)arraydescr_metadata_get,
        NULL, NULL, NULL},
    {"names",
        (getter)arraydescr_names_get,
        (setter)arraydescr_names_set,
        NULL, NULL},
    {"hasobject",
        (getter)arraydescr_hasobject_get,
        NULL, NULL, NULL},
    {NULL, NULL, NULL, NULL, NULL},
};

static PyObject *
arraydescr_new(PyTypeObject *NPY_UNUSED(subtype),
                PyObject *args, PyObject *kwds)
{
    PyObject *odescr, *metadata=NULL;
    PyArray_Descr *descr, *conv;
    npy_bool align = NPY_FALSE;
    npy_bool copy = NPY_FALSE;
    npy_bool copied = NPY_FALSE;

    static char *kwlist[] = {"dtype", "align", "copy", "metadata", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O&O&O!:dtype", kwlist,
                &odescr,
                PyArray_BoolConverter, &align,
                PyArray_BoolConverter, &copy,
                &PyDict_Type, &metadata)) {
        return NULL;
    }

    if (align) {
        if (!PyArray_DescrAlignConverter(odescr, &conv)) {
            return NULL;
        }
    }
    else if (!PyArray_DescrConverter(odescr, &conv)) {
        return NULL;
    }

    /* Get a new copy of it unless it's already a copy */
    if (copy && conv->fields == Py_None) {
        descr = PyArray_DescrNew(conv);
        Py_DECREF(conv);
        conv = descr;
        copied = NPY_TRUE;
    }

    if ((metadata != NULL)) {
        /*
         * We need to be sure to make a new copy of the data-type and any
         * underlying dictionary
         */
        if (!copied) {
            copied = NPY_TRUE;
            descr = PyArray_DescrNew(conv);
            Py_DECREF(conv);
            conv = descr;
        }
        if ((conv->metadata != NULL)) {
            /*
             * Make a copy of the metadata before merging with the
             * input metadata so that this data-type descriptor has
             * it's own copy
             */
            /* Save a reference */
            odescr = conv->metadata;
            conv->metadata = PyDict_Copy(odescr);
            /* Decrement the old reference */
            Py_DECREF(odescr);

            /*
             * Update conv->metadata with anything new in metadata
             * keyword, but do not over-write anything already there
             */
            if (PyDict_Merge(conv->metadata, metadata, 0) != 0) {
                Py_DECREF(conv);
                return NULL;
            }
        }
        else {
            /* Make a copy of the input dictionary */
            conv->metadata = PyDict_Copy(metadata);
        }
    }

    return (PyObject *)conv;
}

/*
 * Return a tuple of
 * (cleaned metadata dictionary, tuple with (str, num))
 */
static PyObject *
_get_pickleabletype_from_datetime_metadata(PyArray_Descr *dtype)
{
    PyObject *ret, *dt_tuple;
    PyArray_DatetimeMetaData *meta;

    /* Create the 2-item tuple to return */
    ret = PyTuple_New(2);
    if (ret == NULL) {
        return NULL;
    }

    /* Store the metadata dictionary */
    if (dtype->metadata != NULL) {
        Py_INCREF(dtype->metadata);
        PyTuple_SET_ITEM(ret, 0, dtype->metadata);
    } else {
        PyTuple_SET_ITEM(ret, 0, PyDict_New());
    }

    /* Convert the datetime metadata into a tuple */
    meta = get_datetime_metadata_from_dtype(dtype);
    if (meta == NULL) {
        Py_DECREF(ret);
        return NULL;
    }
    /* Use a 4-tuple that numpy 1.6 knows how to unpickle */
    dt_tuple = PyTuple_New(4);
    if (dt_tuple == NULL) {
        Py_DECREF(ret);
        return NULL;
    }
    PyTuple_SET_ITEM(dt_tuple, 0,
            PyBytes_FromString(_datetime_strings[meta->base]));
    PyTuple_SET_ITEM(dt_tuple, 1,
            PyInt_FromLong(meta->num));
    PyTuple_SET_ITEM(dt_tuple, 2,
            PyInt_FromLong(1));
    PyTuple_SET_ITEM(dt_tuple, 3,
            PyInt_FromLong(1));

    PyTuple_SET_ITEM(ret, 1, dt_tuple);

    return ret;
}

/*
 * return a tuple of (callable object, args, state).
 *
 * TODO: This method needs to change so that unpickling doesn't
 *       use __setstate__. This is required for the dtype
 *       to be an immutable object.
 */
static PyObject *
arraydescr_reduce(PyArray_Descr *self, PyObject *NPY_UNUSED(args))
{
    /*
     * version number of this pickle type. Increment if we need to
     * change the format. Be sure to handle the old versions in
     * arraydescr_setstate.
    */
    const int version = 4;
    PyObject *ret, *mod, *obj;
    PyObject *state;
    char endian;
    int elsize, alignment;

    ret = PyTuple_New(3);
    if (ret == NULL) {
        return NULL;
    }
    mod = PyImport_ImportModule("numpy.core._multiarray_umath");
    if (mod == NULL) {
        Py_DECREF(ret);
        return NULL;
    }
    obj = PyObject_GetAttrString(mod, "dtype");
    Py_DECREF(mod);
    if (obj == NULL) {
        Py_DECREF(ret);
        return NULL;
    }
    PyTuple_SET_ITEM(ret, 0, obj);
    if (PyTypeNum_ISUSERDEF(self->type_num)
            || ((self->type_num == NPY_VOID
                    && self->typeobj != &PyVoidArrType_Type))) {
        obj = (PyObject *)self->typeobj;
        Py_INCREF(obj);
    }
    else {
        elsize = self->elsize;
        if (self->type_num == NPY_UNICODE) {
            elsize >>= 2;
        }
        obj = PyUString_FromFormat("%c%d",self->kind, elsize);
    }
    PyTuple_SET_ITEM(ret, 1, Py_BuildValue("(Nii)", obj, 0, 1));

    /*
     * Now return the state which is at least byteorder,
     * subarray, and fields
     */
    endian = self->byteorder;
    if (endian == '=') {
        endian = '<';
        if (!PyArray_IsNativeByteOrder(endian)) {
            endian = '>';
        }
    }
    if (PyDataType_ISDATETIME(self)) {
        PyObject *newobj;
        state = PyTuple_New(9);
        PyTuple_SET_ITEM(state, 0, PyInt_FromLong(version));
        /*
         * newobj is a tuple of the Python metadata dictionary
         * and tuple of date_time info (str, num)
         */
        newobj = _get_pickleabletype_from_datetime_metadata(self);
        if (newobj == NULL) {
            Py_DECREF(state);
            Py_DECREF(ret);
            return NULL;
        }
        PyTuple_SET_ITEM(state, 8, newobj);
    }
    else if (self->metadata) {
        state = PyTuple_New(9);
        PyTuple_SET_ITEM(state, 0, PyInt_FromLong(version));
        Py_INCREF(self->metadata);
        PyTuple_SET_ITEM(state, 8, self->metadata);
    }
    else { /* Use version 3 pickle format */
        state = PyTuple_New(8);
        PyTuple_SET_ITEM(state, 0, PyInt_FromLong(3));
    }

    PyTuple_SET_ITEM(state, 1, PyUString_FromFormat("%c", endian));
    PyTuple_SET_ITEM(state, 2, arraydescr_subdescr_get(self));
    if (PyDataType_HASFIELDS(self)) {
        Py_INCREF(self->names);
        Py_INCREF(self->fields);
        PyTuple_SET_ITEM(state, 3, self->names);
        PyTuple_SET_ITEM(state, 4, self->fields);
    }
    else {
        PyTuple_SET_ITEM(state, 3, Py_None);
        PyTuple_SET_ITEM(state, 4, Py_None);
        Py_INCREF(Py_None);
        Py_INCREF(Py_None);
    }

    /* for extended types it also includes elsize and alignment */
    if (PyTypeNum_ISEXTENDED(self->type_num)) {
        elsize = self->elsize;
        alignment = self->alignment;
    }
    else {
        elsize = -1;
        alignment = -1;
    }
    PyTuple_SET_ITEM(state, 5, PyInt_FromLong(elsize));
    PyTuple_SET_ITEM(state, 6, PyInt_FromLong(alignment));
    PyTuple_SET_ITEM(state, 7, PyInt_FromLong(self->flags));

    PyTuple_SET_ITEM(ret, 2, state);
    return ret;
}

/*
 * returns NPY_OBJECT_DTYPE_FLAGS if this data-type has an object portion used
 * when setting the state because hasobject is not stored.
 */
static char
_descr_find_object(PyArray_Descr *self)
{
    if (self->flags
            || self->type_num == NPY_OBJECT
            || self->kind == 'O') {
        return NPY_OBJECT_DTYPE_FLAGS;
    }
    if (PyDataType_HASFIELDS(self)) {
        PyObject *key, *value, *title = NULL;
        PyArray_Descr *new;
        int offset;
        Py_ssize_t pos = 0;

        while (PyDict_Next(self->fields, &pos, &key, &value)) {
            if NPY_TITLE_KEY(key, value) {
                continue;
            }
            if (!PyArg_ParseTuple(value, "Oi|O", &new, &offset, &title)) {
                PyErr_Clear();
                return 0;
            }
            if (_descr_find_object(new)) {
                new->flags = NPY_OBJECT_DTYPE_FLAGS;
                return NPY_OBJECT_DTYPE_FLAGS;
            }
        }
    }
    return 0;
}

/*
 * state is at least byteorder, subarray, and fields but could include elsize
 * and alignment for EXTENDED arrays
 */
static PyObject *
arraydescr_setstate(PyArray_Descr *self, PyObject *args)
{
    int elsize = -1, alignment = -1;
    int version = 4;
    char endian;
    PyObject *endian_obj;
    PyObject *subarray, *fields, *names = NULL, *metadata=NULL;
    int incref_names = 1;
    int int_dtypeflags = 0;
    char dtypeflags;

    if (self->fields == Py_None) {
        Py_RETURN_NONE;
    }
    if (PyTuple_GET_SIZE(args) != 1
            || !(PyTuple_Check(PyTuple_GET_ITEM(args, 0)))) {
        PyErr_BadInternalCall();
        return NULL;
    }
    switch (PyTuple_GET_SIZE(PyTuple_GET_ITEM(args,0))) {
    case 9:
        if (!PyArg_ParseTuple(args, "(iOOOOiiiO):__setstate__",
                    &version, &endian_obj,
                    &subarray, &names, &fields, &elsize,
                    &alignment, &int_dtypeflags, &metadata)) {
            PyErr_Clear();
            return NULL;
        }
        break;
    case 8:
        if (!PyArg_ParseTuple(args, "(iOOOOiii):__setstate__",
                    &version, &endian_obj,
                    &subarray, &names, &fields, &elsize,
                    &alignment, &int_dtypeflags)) {
            return NULL;
        }
        break;
    case 7:
        if (!PyArg_ParseTuple(args, "(iOOOOii):__setstate__",
                    &version, &endian_obj,
                    &subarray, &names, &fields, &elsize,
                    &alignment)) {
            return NULL;
        }
        break;
    case 6:
        if (!PyArg_ParseTuple(args, "(iOOOii):__setstate__",
                    &version,
                    &endian_obj, &subarray, &fields,
                    &elsize, &alignment)) {
            return NULL;
        }
        break;
    case 5:
        version = 0;
        if (!PyArg_ParseTuple(args, "(OOOii):__setstate__",
                    &endian_obj, &subarray, &fields, &elsize,
                    &alignment)) {
            return NULL;
        }
        break;
    default:
        /* raise an error */
        if (PyTuple_GET_SIZE(PyTuple_GET_ITEM(args,0)) > 5) {
            version = PyInt_AsLong(PyTuple_GET_ITEM(args, 0));
        }
        else {
            version = -1;
        }
    }

    /*
     * If we ever need another pickle format, increment the version
     * number. But we should still be able to handle the old versions.
     */
    if (version < 0 || version > 4) {
        PyErr_Format(PyExc_ValueError,
                     "can't handle version %d of numpy.dtype pickle",
                     version);
        return NULL;
    }
    /* Invalidate cached hash value */
    self->hash = -1;

    if (version == 1 || version == 0) {
        if (fields != Py_None) {
            PyObject *key, *list;
            key = PyInt_FromLong(-1);
            list = PyDict_GetItem(fields, key);
            if (!list) {
                return NULL;
            }
            Py_INCREF(list);
            names = list;
            PyDict_DelItem(fields, key);
            incref_names = 0;
        }
        else {
            names = Py_None;
        }
    }

    /* Parse endian */
    if (PyUnicode_Check(endian_obj) || PyBytes_Check(endian_obj)) {
        PyObject *tmp = NULL;
        char *str;
        Py_ssize_t len;

        if (PyUnicode_Check(endian_obj)) {
            tmp = PyUnicode_AsASCIIString(endian_obj);
            if (tmp == NULL) {
                return NULL;
            }
            endian_obj = tmp;
        }

        if (PyBytes_AsStringAndSize(endian_obj, &str, &len) < 0) {
            Py_XDECREF(tmp);
            return NULL;
        }
        if (len != 1) {
            PyErr_SetString(PyExc_ValueError,
                            "endian is not 1-char string in Numpy dtype unpickling");
            Py_XDECREF(tmp);
            return NULL;
        }
        endian = str[0];
        Py_XDECREF(tmp);
    }
    else {
        PyErr_SetString(PyExc_ValueError,
                        "endian is not a string in Numpy dtype unpickling");
        return NULL;
    }

    if ((fields == Py_None && names != Py_None) ||
        (names == Py_None && fields != Py_None)) {
        PyErr_Format(PyExc_ValueError,
                "inconsistent fields and names in Numpy dtype unpickling");
        return NULL;
    }

    if (names != Py_None && !PyTuple_Check(names)) {
        PyErr_Format(PyExc_ValueError,
                "non-tuple names in Numpy dtype unpickling");
        return NULL;
    }

    if (fields != Py_None && !PyDict_Check(fields)) {
        PyErr_Format(PyExc_ValueError,
                "non-dict fields in Numpy dtype unpickling");
        return NULL;
    }

    if (endian != '|' && PyArray_IsNativeByteOrder(endian)) {
        endian = '=';
    }
    self->byteorder = endian;
    if (self->subarray) {
        Py_XDECREF(self->subarray->base);
        Py_XDECREF(self->subarray->shape);
        PyArray_free(self->subarray);
    }
    self->subarray = NULL;

    if (subarray != Py_None) {
        PyObject *subarray_shape;

        /*
         * Ensure that subarray[0] is an ArrayDescr and
         * that subarray_shape obtained from subarray[1] is a tuple of integers.
         */
        if (!(PyTuple_Check(subarray) &&
              PyTuple_Size(subarray) == 2 &&
              PyArray_DescrCheck(PyTuple_GET_ITEM(subarray, 0)))) {
            PyErr_Format(PyExc_ValueError,
                         "incorrect subarray in __setstate__");
            return NULL;
        }
        subarray_shape = PyTuple_GET_ITEM(subarray, 1);
        if (PyNumber_Check(subarray_shape)) {
            PyObject *tmp;
#if defined(NPY_PY3K)
            tmp = PyNumber_Long(subarray_shape);
#else
            tmp = PyNumber_Int(subarray_shape);
#endif
            if (tmp == NULL) {
                return NULL;
            }
            subarray_shape = Py_BuildValue("(O)", tmp);
            Py_DECREF(tmp);
            if (subarray_shape == NULL) {
                return NULL;
            }
        }
        else if (_is_tuple_of_integers(subarray_shape)) {
            Py_INCREF(subarray_shape);
        }
        else {
            PyErr_Format(PyExc_ValueError,
                         "incorrect subarray shape in __setstate__");
            return NULL;
        }

        self->subarray = PyArray_malloc(sizeof(PyArray_ArrayDescr));
        if (!PyDataType_HASSUBARRAY(self)) {
            return PyErr_NoMemory();
        }
        self->subarray->base = (PyArray_Descr *)PyTuple_GET_ITEM(subarray, 0);
        Py_INCREF(self->subarray->base);
        self->subarray->shape = subarray_shape;
    }

    if (fields != Py_None) {
        /*
         * Ensure names are of appropriate string type
         */
        Py_ssize_t i;
        int names_ok = 1;
        PyObject *name;

        for (i = 0; i < PyTuple_GET_SIZE(names); ++i) {
            name = PyTuple_GET_ITEM(names, i);
            if (!PyUString_Check(name)) {
                names_ok = 0;
                break;
            }
        }

        if (names_ok) {
            Py_XDECREF(self->fields);
            self->fields = fields;
            Py_INCREF(fields);
            Py_XDECREF(self->names);
            self->names = names;
            if (incref_names) {
                Py_INCREF(names);
            }
        }
        else {
            /*
             * At least one of the names is not of the expected type.
             * The difference might originate from pickles that were
             * created on another Python version (PY2/PY3). Go through
             * the names and convert if possible for compatibility
             */
            PyObject *tmp, *new_name, *field;

            tmp = PyDict_New();
            if (tmp == NULL) {
                return NULL;
            }
            Py_XDECREF(self->fields);
            self->fields = tmp;

            tmp = PyTuple_New(PyTuple_GET_SIZE(names));
            if (tmp == NULL) {
                return NULL;
            }
            Py_XDECREF(self->names);
            self->names = tmp;

            for (i = 0; i < PyTuple_GET_SIZE(names); ++i) {
                name = PyTuple_GET_ITEM(names, i);
                field = PyDict_GetItem(fields, name);
                if (!field) {
                    return NULL;
                }

#if defined(NPY_PY3K)
                /*
                 * To support pickle.load(f, encoding='bytes') for loading Py2
                 * generated pickles on Py3, we need to be more lenient and convert
                 * field names from byte strings to unicode.
                 */
                if (PyUnicode_Check(name)) {
                    // no transformation needed, keep it as is
                    new_name = name;
                    Py_INCREF(new_name);
                }
                else {
                    new_name = PyUnicode_FromEncodedObject(name, "ASCII", "strict");
                    if (new_name == NULL) {
                        return NULL;
                    }
                }
#else
                // PY2 should be able to read PY3 pickles. See gh-2407
                if (PyString_Check(name)) {
                    // It is a PY2 string, no transformation is needed
                    new_name = name;
                    Py_INCREF(new_name);
                }
                else if (PyUnicode_Check(name)) {
                    // The field names of a structured dtype were pickled in PY3 as unicode strings
                    // so, to unpickle them in PY2, we need to convert them to PY2 strings
                    new_name = PyUnicode_AsEncodedString(name, "ASCII", "strict");
                    if (new_name == NULL) {
                        return NULL;
                    }
                }
                else {
                    // The field name is not a string or a unicode object, we cannot process it
                    PyErr_Format(PyExc_ValueError,
                        "non-string/non-unicode names in Numpy dtype unpickling");
                    return NULL;
                }

#endif

                PyTuple_SET_ITEM(self->names, i, new_name);
                if (PyDict_SetItem(self->fields, new_name, field) != 0) {
                    return NULL;
                }
            }
        }
    }

    if (PyTypeNum_ISEXTENDED(self->type_num)) {
        self->elsize = elsize;
        self->alignment = alignment;
    }

    /*
     * We use an integer converted to char for backward compatibility with
     * pickled arrays. Pickled arrays created with previous versions encoded
     * flags as an int even though it actually was a char in the PyArray_Descr
     * structure
     */
    dtypeflags = int_dtypeflags;
    if (dtypeflags != int_dtypeflags) {
        PyErr_Format(PyExc_ValueError,
                     "incorrect value for flags variable (overflow)");
        return NULL;
    }
    else {
        self->flags = dtypeflags;
    }

    if (version < 3) {
        self->flags = _descr_find_object(self);
    }

    /*
     * We have a borrowed reference to metadata so no need
     * to alter reference count when throwing away Py_None.
     */
    if (metadata == Py_None) {
        metadata = NULL;
    }

    if (PyDataType_ISDATETIME(self) && (metadata != NULL)) {
        PyObject *old_metadata, *errmsg;
        PyArray_DatetimeMetaData temp_dt_data;

        if ((! PyTuple_Check(metadata)) || (PyTuple_Size(metadata) != 2)) {
            errmsg = PyUString_FromString("Invalid datetime dtype (metadata, c_metadata): ");
            PyUString_ConcatAndDel(&errmsg, PyObject_Repr(metadata));
            PyErr_SetObject(PyExc_ValueError, errmsg);
            Py_DECREF(errmsg);
            return NULL;
        }

        if (convert_datetime_metadata_tuple_to_datetime_metadata(
                                    PyTuple_GET_ITEM(metadata, 1),
                                    &temp_dt_data,
                                    NPY_TRUE) < 0) {
            return NULL;
        }

        old_metadata = self->metadata;
        self->metadata = PyTuple_GET_ITEM(metadata, 0);
        memcpy((char *) &((PyArray_DatetimeDTypeMetaData *)self->c_metadata)->meta,
               (char *) &temp_dt_data,
               sizeof(PyArray_DatetimeMetaData));
        Py_XINCREF(self->metadata);
        Py_XDECREF(old_metadata);
    }
    else {
        PyObject *old_metadata = self->metadata;
        self->metadata = metadata;
        Py_XINCREF(self->metadata);
        Py_XDECREF(old_metadata);
    }

    Py_RETURN_NONE;
}

/*NUMPY_API
 *
 * Get type-descriptor from an object forcing alignment if possible
 * None goes to DEFAULT type.
 *
 * any object with the .fields attribute and/or .itemsize attribute (if the
 *.fields attribute does not give the total size -- i.e. a partial record
 * naming).  If itemsize is given it must be >= size computed from fields
 *
 * The .fields attribute must return a convertible dictionary if present.
 * Result inherits from NPY_VOID.
*/
NPY_NO_EXPORT int
PyArray_DescrAlignConverter(PyObject *obj, PyArray_Descr **at)
{
    if (PyDict_Check(obj) || PyDictProxy_Check(obj)) {
        *at =  _convert_from_dict(obj, 1);
    }
    else if (PyBytes_Check(obj)) {
        *at = _convert_from_commastring(obj, 1);
    }
    else if (PyUnicode_Check(obj)) {
        PyObject *tmp;
        tmp = PyUnicode_AsASCIIString(obj);
        *at = _convert_from_commastring(tmp, 1);
        Py_DECREF(tmp);
    }
    else if (PyTuple_Check(obj)) {
        *at = _convert_from_tuple(obj, 1);
    }
    else if (PyList_Check(obj)) {
        *at = _convert_from_array_descr(obj, 1);
    }
    else {
        return PyArray_DescrConverter(obj, at);
    }
    if (*at == NULL) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_ValueError,
                    "data-type-descriptor not understood");
        }
        return NPY_FAIL;
    }
    return NPY_SUCCEED;
}

/*NUMPY_API
 *
 * Get type-descriptor from an object forcing alignment if possible
 * None goes to NULL.
 */
NPY_NO_EXPORT int
PyArray_DescrAlignConverter2(PyObject *obj, PyArray_Descr **at)
{
    if (PyDict_Check(obj) || PyDictProxy_Check(obj)) {
        *at =  _convert_from_dict(obj, 1);
    }
    else if (PyBytes_Check(obj)) {
        *at = _convert_from_commastring(obj, 1);
    }
    else if (PyUnicode_Check(obj)) {
        PyObject *tmp;
        tmp = PyUnicode_AsASCIIString(obj);
        *at = _convert_from_commastring(tmp, 1);
        Py_DECREF(tmp);
    }
    else if (PyList_Check(obj)) {
        *at = _convert_from_array_descr(obj, 1);
    }
    else {
        return PyArray_DescrConverter2(obj, at);
    }
    if (*at == NULL) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_ValueError,
                    "data-type-descriptor not understood");
        }
        return NPY_FAIL;
    }
    return NPY_SUCCEED;
}



/*NUMPY_API
 *
 * returns a copy of the PyArray_Descr structure with the byteorder
 * altered:
 * no arguments:  The byteorder is swapped (in all subfields as well)
 * single argument:  The byteorder is forced to the given state
 * (in all subfields as well)
 *
 * Valid states:  ('big', '>') or ('little' or '<')
 * ('native', or '=')
 *
 * If a descr structure with | is encountered it's own
 * byte-order is not changed but any fields are:
 *
 *
 * Deep bytorder change of a data-type descriptor
 * *** Leaves reference count of self unchanged --- does not DECREF self ***
 */
NPY_NO_EXPORT PyArray_Descr *
PyArray_DescrNewByteorder(PyArray_Descr *self, char newendian)
{
    PyArray_Descr *new;
    char endian;

    new = PyArray_DescrNew(self);
    endian = new->byteorder;
    if (endian != NPY_IGNORE) {
        if (newendian == NPY_SWAP) {
            /* swap byteorder */
            if (PyArray_ISNBO(endian)) {
                endian = NPY_OPPBYTE;
            }
            else {
                endian = NPY_NATBYTE;
            }
            new->byteorder = endian;
        }
        else if (newendian != NPY_IGNORE) {
            new->byteorder = newendian;
        }
    }
    if (PyDataType_HASFIELDS(new)) {
        PyObject *newfields;
        PyObject *key, *value;
        PyObject *newvalue;
        PyObject *old;
        PyArray_Descr *newdescr;
        Py_ssize_t pos = 0;
        int len, i;

        newfields = PyDict_New();
        /* make new dictionary with replaced PyArray_Descr Objects */
        while (PyDict_Next(self->fields, &pos, &key, &value)) {
            if NPY_TITLE_KEY(key, value) {
                continue;
            }
            if (!PyUString_Check(key) || !PyTuple_Check(value) ||
                ((len=PyTuple_GET_SIZE(value)) < 2)) {
                continue;
            }
            old = PyTuple_GET_ITEM(value, 0);
            if (!PyArray_DescrCheck(old)) {
                continue;
            }
            newdescr = PyArray_DescrNewByteorder(
                    (PyArray_Descr *)old, newendian);
            if (newdescr == NULL) {
                Py_DECREF(newfields); Py_DECREF(new);
                return NULL;
            }
            newvalue = PyTuple_New(len);
            PyTuple_SET_ITEM(newvalue, 0, (PyObject *)newdescr);
            for (i = 1; i < len; i++) {
                old = PyTuple_GET_ITEM(value, i);
                Py_INCREF(old);
                PyTuple_SET_ITEM(newvalue, i, old);
            }
            PyDict_SetItem(newfields, key, newvalue);
            Py_DECREF(newvalue);
        }
        Py_DECREF(new->fields);
        new->fields = newfields;
    }
    if (PyDataType_HASSUBARRAY(new)) {
        Py_DECREF(new->subarray->base);
        new->subarray->base = PyArray_DescrNewByteorder(
                self->subarray->base, newendian);
    }
    return new;
}


static PyObject *
arraydescr_newbyteorder(PyArray_Descr *self, PyObject *args)
{
    char endian=NPY_SWAP;

    if (!PyArg_ParseTuple(args, "|O&:newbyteorder", PyArray_ByteorderConverter,
                &endian)) {
        return NULL;
    }
    return (PyObject *)PyArray_DescrNewByteorder(self, endian);
}

static PyMethodDef arraydescr_methods[] = {
    /* for pickling */
    {"__reduce__",
        (PyCFunction)arraydescr_reduce,
        METH_VARARGS, NULL},
    {"__setstate__",
        (PyCFunction)arraydescr_setstate,
        METH_VARARGS, NULL},
    {"newbyteorder",
        (PyCFunction)arraydescr_newbyteorder,
        METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}           /* sentinel */
};

/*
 * Checks whether the structured data type in 'dtype'
 * has a simple layout, where all the fields are in order,
 * and follow each other with no alignment padding.
 *
 * When this returns true, the dtype can be reconstructed
 * from a list of the field names and dtypes with no additional
 * dtype parameters.
 *
 * Returns 1 if it has a simple layout, 0 otherwise.
 */
NPY_NO_EXPORT int
is_dtype_struct_simple_unaligned_layout(PyArray_Descr *dtype)
{
    PyObject *names, *fields, *key, *tup, *title;
    Py_ssize_t i, names_size;
    PyArray_Descr *fld_dtype;
    int fld_offset;
    npy_intp total_offset;

    /* Get some properties from the dtype */
    names = dtype->names;
    names_size = PyTuple_GET_SIZE(names);
    fields = dtype->fields;

    /* Start at offset zero */
    total_offset = 0;

    for (i = 0; i < names_size; ++i) {
        key = PyTuple_GET_ITEM(names, i);
        if (key == NULL) {
            return 0;
        }
        tup = PyDict_GetItem(fields, key);
        if (tup == NULL) {
            return 0;
        }
        if (!PyArg_ParseTuple(tup, "Oi|O", &fld_dtype, &fld_offset, &title)) {
            PyErr_Clear();
            return 0;
        }
        /* If this field doesn't follow the pattern, not a simple layout */
        if (total_offset != fld_offset) {
            return 0;
        }
        /* Get the next offset */
        total_offset += fld_dtype->elsize;
    }

    /*
     * If the itemsize doesn't match the final offset, it's
     * not a simple layout.
     */
    if (total_offset != dtype->elsize) {
        return 0;
    }

    /* It's a simple layout, since all the above tests passed */
    return 1;
}

/*
 * The general dtype repr function.
 */
static PyObject *
arraydescr_repr(PyArray_Descr *dtype)
{
    PyObject *_numpy_dtype;
    PyObject *res;
    _numpy_dtype = PyImport_ImportModule("numpy.core._dtype");
    if (_numpy_dtype == NULL) {
        return NULL;
    }
    res = PyObject_CallMethod(_numpy_dtype, "__repr__", "O", dtype);
    Py_DECREF(_numpy_dtype);
    return res;
}
/*
 * The general dtype str function.
 */
static PyObject *
arraydescr_str(PyArray_Descr *dtype)
{
    PyObject *_numpy_dtype;
    PyObject *res;
    _numpy_dtype = PyImport_ImportModule("numpy.core._dtype");
    if (_numpy_dtype == NULL) {
        return NULL;
    }
    res = PyObject_CallMethod(_numpy_dtype, "__str__", "O", dtype);
    Py_DECREF(_numpy_dtype);
    return res;
}

static PyObject *
arraydescr_richcompare(PyArray_Descr *self, PyObject *other, int cmp_op)
{
    PyArray_Descr *new = NULL;
    PyObject *result = Py_NotImplemented;
    if (!PyArray_DescrCheck(other)) {
        if (PyArray_DescrConverter(other, &new) == NPY_FAIL) {
            return NULL;
        }
    }
    else {
        new = (PyArray_Descr *)other;
        Py_INCREF(new);
    }
    switch (cmp_op) {
    case Py_LT:
        if (!PyArray_EquivTypes(self, new) && PyArray_CanCastTo(self, new)) {
            result = Py_True;
        }
        else {
            result = Py_False;
        }
        break;
    case Py_LE:
        if (PyArray_CanCastTo(self, new)) {
            result = Py_True;
        }
        else {
            result = Py_False;
        }
        break;
    case Py_EQ:
        if (PyArray_EquivTypes(self, new)) {
            result = Py_True;
        }
        else {
            result = Py_False;
        }
        break;
    case Py_NE:
        if (PyArray_EquivTypes(self, new))
            result = Py_False;
        else
            result = Py_True;
        break;
    case Py_GT:
        if (!PyArray_EquivTypes(self, new) && PyArray_CanCastTo(new, self)) {
            result = Py_True;
        }
        else {
            result = Py_False;
        }
        break;
    case Py_GE:
        if (PyArray_CanCastTo(new, self)) {
            result = Py_True;
        }
        else {
            result = Py_False;
        }
        break;
    default:
        result = Py_NotImplemented;
    }

    Py_XDECREF(new);
    Py_INCREF(result);
    return result;
}

static int
descr_nonzero(PyObject *self)
{
    /* `bool(np.dtype(...)) == True` for all dtypes. Needed to override default
     * nonzero implementation, which checks if `len(object) > 0`. */
    return 1;
}

static PyNumberMethods descr_as_number = {
    (binaryfunc)0,                          /* nb_add */
    (binaryfunc)0,                          /* nb_subtract */
    (binaryfunc)0,                          /* nb_multiply */
    #if defined(NPY_PY3K)
    #else
    (binaryfunc)0,                          /* nb_divide */
    #endif
    (binaryfunc)0,                          /* nb_remainder */
    (binaryfunc)0,                          /* nb_divmod */
    (ternaryfunc)0,                         /* nb_power */
    (unaryfunc)0,                           /* nb_negative */
    (unaryfunc)0,                           /* nb_positive */
    (unaryfunc)0,                           /* nb_absolute */
    (inquiry)descr_nonzero,                 /* nb_nonzero */
};

/*************************************************************************
 ****************   Implement Mapping Protocol ***************************
 *************************************************************************/

static Py_ssize_t
descr_length(PyObject *self0)
{
    PyArray_Descr *self = (PyArray_Descr *)self0;

    if (PyDataType_HASFIELDS(self)) {
        return PyTuple_GET_SIZE(self->names);
    }
    else {
        return 0;
    }
}

static PyObject *
descr_repeat(PyObject *self, Py_ssize_t length)
{
    PyObject *tup;
    PyArray_Descr *new;
    if (length < 0) {
        return PyErr_Format(PyExc_ValueError,
                "Array length must be >= 0, not %"NPY_INTP_FMT, (npy_intp)length);
    }
    tup = Py_BuildValue("O" NPY_SSIZE_T_PYFMT, self, length);
    if (tup == NULL) {
        return NULL;
    }
    PyArray_DescrConverter(tup, &new);
    Py_DECREF(tup);
    return (PyObject *)new;
}

static int
_check_has_fields(PyArray_Descr *self)
{
    if (!PyDataType_HASFIELDS(self)) {
        PyObject *astr = arraydescr_str(self);
        if (astr == NULL) {
            return -1;
        }
#if defined(NPY_PY3K)
        {
            PyObject *bstr = PyUnicode_AsUnicodeEscapeString(astr);
            Py_DECREF(astr);
            astr = bstr;
        }
#endif
        PyErr_Format(PyExc_KeyError,
                "There are no fields in dtype %s.", PyBytes_AsString(astr));
        Py_DECREF(astr);
        return -1;
    }
    else {
        return 0;
    }
}

static PyObject *
_subscript_by_name(PyArray_Descr *self, PyObject *op)
{
    PyObject *obj = PyDict_GetItem(self->fields, op);
    PyObject *descr;
    PyObject *s;

    if (obj == NULL) {
        if (PyUnicode_Check(op)) {
            s = PyUnicode_AsUnicodeEscapeString(op);
        }
        else {
            s = op;
        }

        PyErr_Format(PyExc_KeyError,
                "Field named \'%s\' not found.", PyBytes_AsString(s));
        if (s != op) {
            Py_DECREF(s);
        }
        return NULL;
    }
    descr = PyTuple_GET_ITEM(obj, 0);
    Py_INCREF(descr);
    return descr;
}

static PyObject *
_subscript_by_index(PyArray_Descr *self, Py_ssize_t i)
{
    PyObject *name = PySequence_GetItem(self->names, i);
    PyObject *ret;
    if (name == NULL) {
        PyErr_Format(PyExc_IndexError,
                     "Field index %zd out of range.", i);
        return NULL;
    }
    ret = _subscript_by_name(self, name);
    Py_DECREF(name);
    return ret;
}

static PyObject *
descr_subscript(PyArray_Descr *self, PyObject *op)
{
    if (_check_has_fields(self) < 0) {
        return NULL;
    }

    if (PyBaseString_Check(op)) {
        return _subscript_by_name(self, op);
    }
    else {
        Py_ssize_t i = PyArray_PyIntAsIntp(op);
        if (error_converting(i)) {
            /* if converting to an int gives a type error, adjust the message */
            PyObject *err = PyErr_Occurred();
            if (PyErr_GivenExceptionMatches(err, PyExc_TypeError)) {
                PyErr_SetString(PyExc_TypeError,
                        "Field key must be an integer, string, or unicode.");
            }
            return NULL;
        }
        return _subscript_by_index(self, i);
    }
}

static PySequenceMethods descr_as_sequence = {
    (lenfunc) descr_length,                  /* sq_length */
    (binaryfunc) NULL,                       /* sq_concat */
    (ssizeargfunc) descr_repeat,             /* sq_repeat */
    (ssizeargfunc) NULL,                     /* sq_item */
    (ssizessizeargfunc) NULL,                /* sq_slice */
    (ssizeobjargproc) NULL,                  /* sq_ass_item */
    (ssizessizeobjargproc) NULL,             /* sq_ass_slice */
    (objobjproc) NULL,                       /* sq_contains */
    (binaryfunc) NULL,                       /* sq_inplace_concat */
    (ssizeargfunc) NULL,                     /* sq_inplace_repeat */
};

static PyMappingMethods descr_as_mapping = {
    descr_length,                                /* mp_length*/
    (binaryfunc)descr_subscript,                 /* mp_subscript*/
    (objobjargproc)NULL,                         /* mp_ass_subscript*/
};

/****************** End of Mapping Protocol ******************************/

NPY_NO_EXPORT PyTypeObject PyArrayDescr_Type = {
#if defined(NPY_PY3K)
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                                          /* ob_size */
#endif
    "numpy.dtype",                              /* tp_name */
    sizeof(PyArray_Descr),                      /* tp_basicsize */
    0,                                          /* tp_itemsize */
    /* methods */
    (destructor)arraydescr_dealloc,             /* tp_dealloc */
    0,                                          /* tp_print */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
#if defined(NPY_PY3K)
    (void *)0,                                  /* tp_reserved */
#else
    0,                                          /* tp_compare */
#endif
    (reprfunc)arraydescr_repr,                  /* tp_repr */
    &descr_as_number,                           /* tp_as_number */
    &descr_as_sequence,                         /* tp_as_sequence */
    &descr_as_mapping,                          /* tp_as_mapping */
    0,                                          /* tp_hash */
    0,                                          /* tp_call */
    (reprfunc)arraydescr_str,                   /* tp_str */
    0,                                          /* tp_getattro */
    0,                                          /* tp_setattro */
    0,                                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,                         /* tp_flags */
    0,                                          /* tp_doc */
    0,                                          /* tp_traverse */
    0,                                          /* tp_clear */
    (richcmpfunc)arraydescr_richcompare,        /* tp_richcompare */
    0,                                          /* tp_weaklistoffset */
    0,                                          /* tp_iter */
    0,                                          /* tp_iternext */
    arraydescr_methods,                         /* tp_methods */
    arraydescr_members,                         /* tp_members */
    arraydescr_getsets,                         /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    0,                                          /* tp_init */
    0,                                          /* tp_alloc */
    arraydescr_new,                             /* tp_new */
    0,                                          /* tp_free */
    0,                                          /* tp_is_gc */
    0,                                          /* tp_bases */
    0,                                          /* tp_mro */
    0,                                          /* tp_cache */
    0,                                          /* tp_subclasses */
    0,                                          /* tp_weaklist */
    0,                                          /* tp_del */
    0,                                          /* tp_version_tag */
};
