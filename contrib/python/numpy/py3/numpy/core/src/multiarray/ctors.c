#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"

#include "numpy/npy_math.h"

#include "npy_config.h"

#include "npy_ctypes.h"
#include "npy_pycompat.h"
#include "multiarraymodule.h"

#include "common.h"
#include "ctors.h"
#include "convert_datatype.h"
#include "shape.h"
#include "npy_buffer.h"
#include "lowlevel_strided_loops.h"
#include "_datetime.h"
#include "datetime_strings.h"
#include "array_assign.h"
#include "mapping.h" /* for array_item_asarray */
#include "templ_common.h" /* for npy_mul_with_overflow_intp */
#include "alloc.h"
#include <assert.h>

#include "get_attr_string.h"
#include "array_coercion.h"

/*
 * Reading from a file or a string.
 *
 * As much as possible, we try to use the same code for both files and strings,
 * so the semantics for fromstring and fromfile are the same, especially with
 * regards to the handling of text representations.
 */

/*
 * Scanning function for next element parsing and separator skipping.
 * These functions return:
 *   - 0 to indicate more data to read
 *   - -1 when reading stopped at the end of the string/file
 *   - -2 when reading stopped before the end was reached.
 *
 * The dtype specific parsing functions may set the python error state
 * (they have to get the GIL first) additionally.
 */
typedef int (*next_element)(void **, void *, PyArray_Descr *, void *);
typedef int (*skip_separator)(void **, const char *, void *);


static npy_bool
string_is_fully_read(char const* start, char const* end) {
    if (end == NULL) {
        return *start == '\0';  /* null terminated */
    }
    else {
        return start >= end;  /* fixed length */
    }
}


static int
fromstr_next_element(char **s, void *dptr, PyArray_Descr *dtype,
                     const char *end)
{
    char *e = *s;
    int r = dtype->f->fromstr(*s, dptr, &e, dtype);
    /*
     * fromstr always returns 0 for basic dtypes; s points to the end of the
     * parsed string. If s is not changed an error occurred or the end was
     * reached.
     */
    if (*s == e || r < 0) {
        /* Nothing read, could be end of string or an error (or both) */
        if (string_is_fully_read(*s, end)) {
            return -1;
        }
        return -2;
    }
    *s = e;
    if (end != NULL && *s > end) {
        /* Stop the iteration if we read far enough */
        return -1;
    }
    return 0;
}

static int
fromfile_next_element(FILE **fp, void *dptr, PyArray_Descr *dtype,
                      void *NPY_UNUSED(stream_data))
{
    /* the NULL argument is for backwards-compatibility */
    int r = dtype->f->scanfunc(*fp, dptr, NULL, dtype);
    /* r can be EOF or the number of items read (0 or 1) */
    if (r == 1) {
        return 0;
    }
    else if (r == EOF) {
        return -1;
    }
    else {
        /* unable to read more, but EOF not reached indicating an error. */
        return -2;
    }
}

/*
 * Remove multiple whitespace from the separator, and add a space to the
 * beginning and end. This simplifies the separator-skipping code below.
 */
static char *
swab_separator(const char *sep)
{
    int skip_space = 0;
    char *s, *start;

    s = start = malloc(strlen(sep)+3);
    if (s == NULL) {
        PyErr_NoMemory();
        return NULL;
    }
    /* add space to front if there isn't one */
    if (*sep != '\0' && !isspace(*sep)) {
        *s = ' '; s++;
    }
    while (*sep != '\0') {
        if (isspace(*sep)) {
            if (skip_space) {
                sep++;
            }
            else {
                *s = ' ';
                s++;
                sep++;
                skip_space = 1;
            }
        }
        else {
            *s = *sep;
            s++;
            sep++;
            skip_space = 0;
        }
    }
    /* add space to end if there isn't one */
    if (s != start && s[-1] == ' ') {
        *s = ' ';
        s++;
    }
    *s = '\0';
    return start;
}

/*
 * Assuming that the separator is the next bit in the string (file), skip it.
 *
 * Single spaces in the separator are matched to arbitrary-long sequences
 * of whitespace in the input. If the separator consists only of spaces,
 * it matches one or more whitespace characters.
 *
 * If we can't match the separator, return -2.
 * If we hit the end of the string (file), return -1.
 * Otherwise, return 0.
 */
static int
fromstr_skip_separator(char **s, const char *sep, const char *end)
{
    char *string = *s;
    int result = 0;

    while (1) {
        char c = *string;
        if (string_is_fully_read(string, end)) {
            result = -1;
            break;
        }
        else if (*sep == '\0') {
            if (string != *s) {
                /* matched separator */
                result = 0;
                break;
            }
            else {
                /* separator was whitespace wildcard that didn't match */
                result = -2;
                break;
            }
        }
        else if (*sep == ' ') {
            /* whitespace wildcard */
            if (!isspace(c)) {
                sep++;
                continue;
            }
        }
        else if (*sep != c) {
            result = -2;
            break;
        }
        else {
            sep++;
        }
        string++;
    }
    *s = string;
    return result;
}

static int
fromfile_skip_separator(FILE **fp, const char *sep, void *NPY_UNUSED(stream_data))
{
    int result = 0;
    const char *sep_start = sep;

    while (1) {
        int c = fgetc(*fp);

        if (c == EOF) {
            result = -1;
            break;
        }
        else if (*sep == '\0') {
            ungetc(c, *fp);
            if (sep != sep_start) {
                /* matched separator */
                result = 0;
                break;
            }
            else {
                /* separator was whitespace wildcard that didn't match */
                result = -2;
                break;
            }
        }
        else if (*sep == ' ') {
            /* whitespace wildcard */
            if (!isspace(c)) {
                sep++;
                sep_start++;
                ungetc(c, *fp);
            }
            else if (sep == sep_start) {
                sep_start--;
            }
        }
        else if (*sep != c) {
            ungetc(c, *fp);
            result = -2;
            break;
        }
        else {
            sep++;
        }
    }
    return result;
}

/*
 * Change a sub-array field to the base descriptor
 * and update the dimensions and strides
 * appropriately.  Dimensions and strides are added
 * to the end.
 *
 * Strides are only added if given (because data is given).
 */
static int
_update_descr_and_dimensions(PyArray_Descr **des, npy_intp *newdims,
                             npy_intp *newstrides, int oldnd)
{
    PyArray_Descr *old;
    int newnd;
    int numnew;
    npy_intp *mydim;
    int i;
    int tuple;

    old = *des;
    *des = old->subarray->base;


    mydim = newdims + oldnd;
    tuple = PyTuple_Check(old->subarray->shape);
    if (tuple) {
        numnew = PyTuple_GET_SIZE(old->subarray->shape);
    }
    else {
        numnew = 1;
    }


    newnd = oldnd + numnew;
    if (newnd > NPY_MAXDIMS) {
        goto finish;
    }
    if (tuple) {
        for (i = 0; i < numnew; i++) {
            mydim[i] = (npy_intp) PyLong_AsLong(
                    PyTuple_GET_ITEM(old->subarray->shape, i));
        }
    }
    else {
        mydim[0] = (npy_intp) PyLong_AsLong(old->subarray->shape);
    }

    if (newstrides) {
        npy_intp tempsize;
        npy_intp *mystrides;

        mystrides = newstrides + oldnd;
        /* Make new strides -- always C-contiguous */
        tempsize = (*des)->elsize;
        for (i = numnew - 1; i >= 0; i--) {
            mystrides[i] = tempsize;
            tempsize *= mydim[i] ? mydim[i] : 1;
        }
    }

 finish:
    Py_INCREF(*des);
    Py_DECREF(old);
    return newnd;
}

NPY_NO_EXPORT void
_unaligned_strided_byte_copy(char *dst, npy_intp outstrides, char *src,
                             npy_intp instrides, npy_intp N, int elsize)
{
    npy_intp i;
    char *tout = dst;
    char *tin = src;

#define _COPY_N_SIZE(size) \
    for(i=0; i<N; i++) { \
        memcpy(tout, tin, size); \
        tin += instrides; \
        tout += outstrides; \
    } \
    return

    switch(elsize) {
    case 8:
        _COPY_N_SIZE(8);
    case 4:
        _COPY_N_SIZE(4);
    case 1:
        _COPY_N_SIZE(1);
    case 2:
        _COPY_N_SIZE(2);
    case 16:
        _COPY_N_SIZE(16);
    default:
        _COPY_N_SIZE(elsize);
    }
#undef _COPY_N_SIZE

}

NPY_NO_EXPORT void
_strided_byte_swap(void *p, npy_intp stride, npy_intp n, int size)
{
    char *a, *b, c = 0;
    int j, m;

    switch(size) {
    case 1: /* no byteswap necessary */
        break;
    case 4:
        if (npy_is_aligned((void*)((npy_intp)p | stride), sizeof(npy_uint32))) {
            for (a = (char*)p; n > 0; n--, a += stride) {
                npy_uint32 * a_ = (npy_uint32 *)a;
                *a_ = npy_bswap4(*a_);
            }
        }
        else {
            for (a = (char*)p; n > 0; n--, a += stride) {
                npy_bswap4_unaligned(a);
            }
        }
        break;
    case 8:
        if (npy_is_aligned((void*)((npy_intp)p | stride), sizeof(npy_uint64))) {
            for (a = (char*)p; n > 0; n--, a += stride) {
                npy_uint64 * a_ = (npy_uint64 *)a;
                *a_ = npy_bswap8(*a_);
            }
        }
        else {
            for (a = (char*)p; n > 0; n--, a += stride) {
                npy_bswap8_unaligned(a);
            }
        }
        break;
    case 2:
        if (npy_is_aligned((void*)((npy_intp)p | stride), sizeof(npy_uint16))) {
            for (a = (char*)p; n > 0; n--, a += stride) {
                npy_uint16 * a_ = (npy_uint16 *)a;
                *a_ = npy_bswap2(*a_);
            }
        }
        else {
            for (a = (char*)p; n > 0; n--, a += stride) {
                npy_bswap2_unaligned(a);
            }
        }
        break;
    default:
        m = size/2;
        for (a = (char *)p; n > 0; n--, a += stride - m) {
            b = a + (size - 1);
            for (j = 0; j < m; j++) {
                c=*a; *a++ = *b; *b-- = c;
            }
        }
        break;
    }
}

NPY_NO_EXPORT void
byte_swap_vector(void *p, npy_intp n, int size)
{
    _strided_byte_swap(p, (npy_intp) size, n, size);
    return;
}

/* If numitems > 1, then dst must be contiguous */
NPY_NO_EXPORT void
copy_and_swap(void *dst, void *src, int itemsize, npy_intp numitems,
              npy_intp srcstrides, int swap)
{
    if ((numitems == 1) || (itemsize == srcstrides)) {
        memcpy(dst, src, itemsize*numitems);
    }
    else {
        npy_intp i;
        char *s1 = (char *)src;
        char *d1 = (char *)dst;

        for (i = 0; i < numitems; i++) {
            memcpy(d1, s1, itemsize);
            d1 += itemsize;
            s1 += srcstrides;
        }
    }

    if (swap) {
        byte_swap_vector(dst, numitems, itemsize);
    }
}


/*
 * Recursive helper to assign using a coercion cache. This function
 * must consume the cache depth first, just as the cache was originally
 * produced.
 */
NPY_NO_EXPORT int
PyArray_AssignFromCache_Recursive(
        PyArrayObject *self, const int ndim, coercion_cache_obj **cache)
{
    /* Consume first cache element by extracting information and freeing it */
    PyObject *original_obj = (*cache)->converted_obj;
    PyObject *obj = (*cache)->arr_or_sequence;
    Py_INCREF(obj);
    npy_bool sequence = (*cache)->sequence;
    int depth = (*cache)->depth;
    *cache = npy_unlink_coercion_cache(*cache);

    /*
     * The maximum depth is special (specifically for objects), but usually
     * unrolled in the sequence branch below.
     */
    if (NPY_UNLIKELY(depth == ndim)) {
        /*
         * We have reached the maximum depth. We should simply assign to the
         * element in principle. There is one exception. If this is a 0-D
         * array being stored into a 0-D array (but we do not reach here then).
         */
        if (PyArray_ISOBJECT(self)) {
            assert(ndim != 0);  /* guaranteed by PyArray_AssignFromCache */
            assert(PyArray_NDIM(self) == 0);
            Py_DECREF(obj);
            return PyArray_Pack(PyArray_DESCR(self), PyArray_BYTES(self),
                                original_obj);
        }
        if (sequence) {
            /*
             * Sanity check which may be removed, the error is raised already
             * in `PyArray_DiscoverDTypeAndShape`.
             */
            assert(0);
            PyErr_SetString(PyExc_RuntimeError,
                    "setting an array element with a sequence");
            goto fail;
        }
        else if (original_obj != obj || !PyArray_CheckExact(obj)) {
            /*
             * If the leave node is an array-like, but not a numpy array,
             * we pretend it is an arbitrary scalar.  This means that in
             * most cases (where the dtype is int or float), we will end
             * up using float(array-like), or int(array-like).  That does
             * not support general casting, but helps Quantity and masked
             * arrays, because it allows them to raise an error when
             * `__float__()` or `__int__()` is called.
             */
            Py_DECREF(obj);
            return PyArray_SETITEM(self, PyArray_BYTES(self), original_obj);
        }
    }

    /* The element is either a sequence, or an array */
    if (!sequence) {
        /* Straight forward array assignment */
        assert(PyArray_Check(obj));
        if (PyArray_CopyInto(self, (PyArrayObject *)obj) < 0) {
            goto fail;
        }
    }
    else {
        assert(depth != ndim);
        npy_intp length = PySequence_Length(obj);
        if (length != PyArray_DIMS(self)[0]) {
            PyErr_SetString(PyExc_RuntimeError,
                    "Inconsistent object during array creation? "
                    "Content of sequences changed (length inconsistent).");
            goto fail;
        }

        for (npy_intp i = 0; i < length; i++) {
            PyObject *value = PySequence_Fast_GET_ITEM(obj, i);

            if (*cache == NULL || (*cache)->converted_obj != value ||
                        (*cache)->depth != depth + 1) {
                if (ndim != depth + 1) {
                    PyErr_SetString(PyExc_RuntimeError,
                            "Inconsistent object during array creation? "
                            "Content of sequences changed (now too shallow).");
                    goto fail;
                }
                /* Straight forward assignment of elements */
                char *item;
                item = (PyArray_BYTES(self) + i * PyArray_STRIDES(self)[0]);
                if (PyArray_Pack(PyArray_DESCR(self), item, value) < 0) {
                    goto fail;
                }
            }
            else {
                PyArrayObject *view;
                view = (PyArrayObject *)array_item_asarray(self, i);
                if (view == NULL) {
                    goto fail;
                }
                if (PyArray_AssignFromCache_Recursive(view, ndim, cache) < 0) {
                    Py_DECREF(view);
                    goto fail;
                }
                Py_DECREF(view);
            }
        }
    }
    Py_DECREF(obj);
    return 0;

  fail:
    Py_DECREF(obj);
    return -1;
}


/**
 * Fills an item based on a coercion cache object. It consumes the cache
 * object while doing so.
 *
 * @param self Array to fill.
 * @param cache coercion_cache_object, will be consumed. The cache must not
 *        contain a single array (must start with a sequence). The array case
 *        should be handled by `PyArray_FromArray()` before.
 * @return 0 on success -1 on failure.
 */
NPY_NO_EXPORT int
PyArray_AssignFromCache(PyArrayObject *self, coercion_cache_obj *cache) {
    int ndim = PyArray_NDIM(self);
    /*
     * Do not support ndim == 0 now with an array in the cache.
     * The ndim == 0 is special because np.array(np.array(0), dtype=object)
     * should unpack the inner array.
     * Since the single-array case is special, it is handled previously
     * in either case.
     */
    assert(cache->sequence);
    assert(ndim != 0);  /* guaranteed if cache contains a sequence */

    if (PyArray_AssignFromCache_Recursive(self, ndim, &cache) < 0) {
        /* free the remaining cache. */
        npy_free_coercion_cache(cache);
        return -1;
    }

    /*
     * Sanity check, this is the initial call, and when it returns, the
     * cache has to be fully consumed, otherwise something is wrong.
     * NOTE: May be nicer to put into a recursion helper.
     */
    if (cache != NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                "Inconsistent object during array creation? "
                "Content of sequences changed (cache not consumed).");
        npy_free_coercion_cache(cache);
        return -1;
    }
    return 0;
}


static void
raise_memory_error(int nd, npy_intp const *dims, PyArray_Descr *descr)
{
    static PyObject *exc_type = NULL;

    npy_cache_import(
        "numpy.core._exceptions", "_ArrayMemoryError",
        &exc_type);
    if (exc_type == NULL) {
        goto fail;
    }

    PyObject *shape = PyArray_IntTupleFromIntp(nd, dims);
    if (shape == NULL) {
        goto fail;
    }

    /* produce an error object */
    PyObject *exc_value = PyTuple_Pack(2, shape, (PyObject *)descr);
    Py_DECREF(shape);
    if (exc_value == NULL){
        goto fail;
    }
    PyErr_SetObject(exc_type, exc_value);
    Py_DECREF(exc_value);
    return;

fail:
    /* we couldn't raise the formatted exception for some reason */
    PyErr_WriteUnraisable(NULL);
    PyErr_NoMemory();
}

/*
 * Generic new array creation routine.
 * Internal variant with calloc argument for PyArray_Zeros.
 *
 * steals a reference to descr. On failure or descr->subarray, descr will
 * be decrefed.
 */
NPY_NO_EXPORT PyObject *
PyArray_NewFromDescr_int(
        PyTypeObject *subtype, PyArray_Descr *descr, int nd,
        npy_intp const *dims, npy_intp const *strides, void *data,
        int flags, PyObject *obj, PyObject *base, int zeroed,
        int allow_emptystring)
{
    PyArrayObject_fields *fa;
    npy_intp nbytes;

    if (descr == NULL) {
        return NULL;
    }
    if (nd > NPY_MAXDIMS || nd < 0) {
        PyErr_Format(PyExc_ValueError,
                "number of dimensions must be within [0, %d]", NPY_MAXDIMS);
        Py_DECREF(descr);
        return NULL;
    }

    if (descr->subarray) {
        PyObject *ret;
        npy_intp newdims[2*NPY_MAXDIMS];
        npy_intp *newstrides = NULL;
        memcpy(newdims, dims, nd*sizeof(npy_intp));
        if (strides) {
            newstrides = newdims + NPY_MAXDIMS;
            memcpy(newstrides, strides, nd*sizeof(npy_intp));
        }
        nd =_update_descr_and_dimensions(&descr, newdims,
                                         newstrides, nd);
        ret = PyArray_NewFromDescr_int(
                subtype, descr,
                nd, newdims, newstrides, data,
                flags, obj, base,
                zeroed, allow_emptystring);
        return ret;
    }

    /* Check datatype element size */
    nbytes = descr->elsize;
    if (PyDataType_ISUNSIZED(descr)) {
        if (!PyDataType_ISFLEXIBLE(descr)) {
            PyErr_SetString(PyExc_TypeError, "Empty data-type");
            Py_DECREF(descr);
            return NULL;
        }
        else if (PyDataType_ISSTRING(descr) && !allow_emptystring &&
                 data == NULL) {
            PyArray_DESCR_REPLACE(descr);
            if (descr == NULL) {
                return NULL;
            }
            if (descr->type_num == NPY_STRING) {
                nbytes = descr->elsize = 1;
            }
            else {
                nbytes = descr->elsize = sizeof(npy_ucs4);
            }
        }
    }

    fa = (PyArrayObject_fields *) subtype->tp_alloc(subtype, 0);
    if (fa == NULL) {
        Py_DECREF(descr);
        return NULL;
    }
    fa->_buffer_info = NULL;
    fa->nd = nd;
    fa->dimensions = NULL;
    fa->data = NULL;
    fa->mem_handler = NULL;

    if (data == NULL) {
        fa->flags = NPY_ARRAY_DEFAULT;
        if (flags) {
            fa->flags |= NPY_ARRAY_F_CONTIGUOUS;
            if (nd > 1) {
                fa->flags &= ~NPY_ARRAY_C_CONTIGUOUS;
            }
            flags = NPY_ARRAY_F_CONTIGUOUS;
        }
    }
    else {
        fa->flags = (flags & ~NPY_ARRAY_WRITEBACKIFCOPY);
        fa->flags &= ~NPY_ARRAY_UPDATEIFCOPY;
    }
    fa->descr = descr;
    fa->base = (PyObject *)NULL;
    fa->weakreflist = (PyObject *)NULL;

    if (nd > 0) {
        fa->dimensions = npy_alloc_cache_dim(2 * nd);
        if (fa->dimensions == NULL) {
            PyErr_NoMemory();
            goto fail;
        }
        fa->strides = fa->dimensions + nd;

        /*
         * Copy dimensions, check them, and find total array size `nbytes`
         *
         * Note that we ignore 0-length dimensions, to match this in the `free`
         * calls, `PyArray_NBYTES_ALLOCATED` is a private helper matching this
         * behaviour, but without overflow checking.
         */
        for (int i = 0; i < nd; i++) {
            fa->dimensions[i] = dims[i];

            if (fa->dimensions[i] == 0) {
                /*
                 * Compare to PyArray_OverflowMultiplyList that
                 * returns 0 in this case. See also `PyArray_NBYTES_ALLOCATED`.
                 */
                continue;
            }

            if (fa->dimensions[i] < 0) {
                PyErr_SetString(PyExc_ValueError,
                        "negative dimensions are not allowed");
                goto fail;
            }

            /*
             * Care needs to be taken to avoid integer overflow when multiplying
             * the dimensions together to get the total size of the array.
             */
            if (npy_mul_with_overflow_intp(&nbytes, nbytes, fa->dimensions[i])) {
                PyErr_SetString(PyExc_ValueError,
                        "array is too big; `arr.size * arr.dtype.itemsize` "
                        "is larger than the maximum possible size.");
                goto fail;
            }
        }

        /* Fill the strides (or copy them if they were passed in) */
        if (strides == NULL) {
            /* fill the strides and set the contiguity flags */
            _array_fill_strides(fa->strides, dims, nd, descr->elsize,
                                flags, &(fa->flags));
        }
        else {
            /* User to provided strides (user is responsible for correctness) */
            for (int i = 0; i < nd; i++) {
                fa->strides[i] = strides[i];
            }
            /* Since the strides were passed in must update contiguity */
            PyArray_UpdateFlags((PyArrayObject *)fa,
                    NPY_ARRAY_C_CONTIGUOUS|NPY_ARRAY_F_CONTIGUOUS);
        }
    }
    else {
        fa->dimensions = NULL;
        fa->strides = NULL;
        fa->flags |= NPY_ARRAY_C_CONTIGUOUS|NPY_ARRAY_F_CONTIGUOUS;
    }


    if (data == NULL) {
        /* Store the handler in case the default is modified */
        fa->mem_handler = PyDataMem_GetHandler();
        if (fa->mem_handler == NULL) {
            goto fail;
        }
        /*
         * Allocate something even for zero-space arrays
         * e.g. shape=(0,) -- otherwise buffer exposure
         * (a.data) doesn't work as it should.
         * Could probably just allocate a few bytes here. -- Chuck
         * Note: always sync this with calls to PyDataMem_UserFREE
         */
        if (nbytes == 0) {
            nbytes = descr->elsize ? descr->elsize : 1;
        }
        /*
         * It is bad to have uninitialized OBJECT pointers
         * which could also be sub-fields of a VOID array
         */
        if (zeroed || PyDataType_FLAGCHK(descr, NPY_NEEDS_INIT)) {
            data = PyDataMem_UserNEW_ZEROED(nbytes, 1, fa->mem_handler);
        }
        else {
            data = PyDataMem_UserNEW(nbytes, fa->mem_handler);
        }
        if (data == NULL) {
            raise_memory_error(fa->nd, fa->dimensions, descr);
            goto fail;
        }

        fa->flags |= NPY_ARRAY_OWNDATA;
    }
    else {
        /* The handlers should never be called in this case */
        fa->mem_handler = NULL;
        /*
         * If data is passed in, this object won't own it.
         */
        fa->flags &= ~NPY_ARRAY_OWNDATA;
    }
    fa->data = data;

    /*
     * Always update the aligned flag.  Not owned data or input strides may
     * not be aligned. Also on some platforms (debian sparc) malloc does not
     * provide enough alignment for long double types.
     */
    PyArray_UpdateFlags((PyArrayObject *)fa, NPY_ARRAY_ALIGNED);

    /* Set the base object. It's important to do it here so that
     * __array_finalize__ below receives it
     */
    if (base != NULL) {
        Py_INCREF(base);
        if (PyArray_SetBaseObject((PyArrayObject *)fa, base) < 0) {
            goto fail;
        }
    }

    /*
     * call the __array_finalize__ method if a subtype was requested.
     * If obj is NULL use Py_None for the Python callback.
     */
    if (subtype != &PyArray_Type) {
        PyObject *res, *func;

        func = PyObject_GetAttr((PyObject *)fa, npy_ma_str_array_finalize);
        if (func == NULL) {
            goto fail;
        }
        else if (func == Py_None) {
            Py_DECREF(func);
        }
        else {
            if (PyCapsule_CheckExact(func)) {
                /* A C-function is stored here */
                PyArray_FinalizeFunc *cfunc;
                cfunc = PyCapsule_GetPointer(func, NULL);
                Py_DECREF(func);
                if (cfunc == NULL) {
                    goto fail;
                }
                if (cfunc((PyArrayObject *)fa, obj) < 0) {
                    goto fail;
                }
            }
            else {
                if (obj == NULL) {
                    obj = Py_None;
                }
                res = PyObject_CallFunctionObjArgs(func, obj, NULL);
                Py_DECREF(func);
                if (res == NULL) {
                    goto fail;
                }
                else {
                    Py_DECREF(res);
                }
            }
        }
    }
    return (PyObject *)fa;

 fail:
    Py_XDECREF(fa->mem_handler);
    Py_DECREF(fa);
    return NULL;
}


/*NUMPY_API
 * Generic new array creation routine.
 *
 * steals a reference to descr. On failure or when dtype->subarray is
 * true, dtype will be decrefed.
 */
NPY_NO_EXPORT PyObject *
PyArray_NewFromDescr(
        PyTypeObject *subtype, PyArray_Descr *descr,
        int nd, npy_intp const *dims, npy_intp const *strides, void *data,
        int flags, PyObject *obj)
{
    return PyArray_NewFromDescrAndBase(
            subtype, descr,
            nd, dims, strides, data,
            flags, obj, NULL);
}

/*
 * Sets the base object using PyArray_SetBaseObject
 */
NPY_NO_EXPORT PyObject *
PyArray_NewFromDescrAndBase(
        PyTypeObject *subtype, PyArray_Descr *descr,
        int nd, npy_intp const *dims, npy_intp const *strides, void *data,
        int flags, PyObject *obj, PyObject *base)
{
    return PyArray_NewFromDescr_int(subtype, descr, nd,
                                    dims, strides, data,
                                    flags, obj, base, 0, 0);
}

/*
 * Creates a new array with the same shape as the provided one,
 * with possible memory layout order, data type and shape changes.
 *
 * prototype - The array the new one should be like.
 * order     - NPY_CORDER - C-contiguous result.
 *             NPY_FORTRANORDER - Fortran-contiguous result.
 *             NPY_ANYORDER - Fortran if prototype is Fortran, C otherwise.
 *             NPY_KEEPORDER - Keeps the axis ordering of prototype.
 * dtype     - If not NULL, overrides the data type of the result.
 * ndim      - If not -1, overrides the shape of the result.
 * dims      - If ndim is not -1, overrides the shape of the result.
 * subok     - If 1, use the prototype's array subtype, otherwise
 *             always create a base-class array.
 *
 * NOTE: If dtype is not NULL, steals the dtype reference.  On failure or when
 * dtype->subarray is true, dtype will be decrefed.
 */
NPY_NO_EXPORT PyObject *
PyArray_NewLikeArrayWithShape(PyArrayObject *prototype, NPY_ORDER order,
                              PyArray_Descr *dtype, int ndim, npy_intp const *dims, int subok)
{
    PyObject *ret = NULL;

    if (ndim == -1) {
        ndim = PyArray_NDIM(prototype);
        dims = PyArray_DIMS(prototype);
    }
    else if (order == NPY_KEEPORDER && (ndim != PyArray_NDIM(prototype))) {
        order = NPY_CORDER;
    }

    /* If no override data type, use the one from the prototype */
    if (dtype == NULL) {
        dtype = PyArray_DESCR(prototype);
        Py_INCREF(dtype);
    }

    /* Handle ANYORDER and simple KEEPORDER cases */
    switch (order) {
        case NPY_ANYORDER:
            order = PyArray_ISFORTRAN(prototype) ?
                                    NPY_FORTRANORDER : NPY_CORDER;
            break;
        case NPY_KEEPORDER:
            if (PyArray_IS_C_CONTIGUOUS(prototype) || ndim <= 1) {
                order = NPY_CORDER;
                break;
            }
            else if (PyArray_IS_F_CONTIGUOUS(prototype)) {
                order = NPY_FORTRANORDER;
                break;
            }
            break;
        default:
            break;
    }

    /* If it's not KEEPORDER, this is simple */
    if (order != NPY_KEEPORDER) {
        ret = PyArray_NewFromDescr(subok ? Py_TYPE(prototype) : &PyArray_Type,
                                        dtype,
                                        ndim,
                                        dims,
                                        NULL,
                                        NULL,
                                        order,
                                        subok ? (PyObject *)prototype : NULL);
    }
    /* KEEPORDER needs some analysis of the strides */
    else {
        npy_intp strides[NPY_MAXDIMS], stride;
        npy_stride_sort_item strideperm[NPY_MAXDIMS];
        int idim;

        PyArray_CreateSortedStridePerm(ndim,
                                        PyArray_STRIDES(prototype),
                                        strideperm);

        /* Build the new strides */
        stride = dtype->elsize;
        if (stride == 0 && PyDataType_ISSTRING(dtype)) {
            /* Special case for dtype=str or dtype=bytes. */
            if (dtype->type_num == NPY_STRING) {
                /* dtype is bytes */
                stride = 1;
            }
            else {
                /* dtype is str (type_num is NPY_UNICODE) */
                stride = 4;
            }
        }
        for (idim = ndim-1; idim >= 0; --idim) {
            npy_intp i_perm = strideperm[idim].perm;
            strides[i_perm] = stride;
            stride *= dims[i_perm];
        }

        /* Finally, allocate the array */
        ret = PyArray_NewFromDescr(subok ? Py_TYPE(prototype) : &PyArray_Type,
                                        dtype,
                                        ndim,
                                        dims,
                                        strides,
                                        NULL,
                                        0,
                                        subok ? (PyObject *)prototype : NULL);
    }

    return ret;
}

/*NUMPY_API
 * Creates a new array with the same shape as the provided one,
 * with possible memory layout order and data type changes.
 *
 * prototype - The array the new one should be like.
 * order     - NPY_CORDER - C-contiguous result.
 *             NPY_FORTRANORDER - Fortran-contiguous result.
 *             NPY_ANYORDER - Fortran if prototype is Fortran, C otherwise.
 *             NPY_KEEPORDER - Keeps the axis ordering of prototype.
 * dtype     - If not NULL, overrides the data type of the result.
 * subok     - If 1, use the prototype's array subtype, otherwise
 *             always create a base-class array.
 *
 * NOTE: If dtype is not NULL, steals the dtype reference.  On failure or when
 * dtype->subarray is true, dtype will be decrefed.
 */
NPY_NO_EXPORT PyObject *
PyArray_NewLikeArray(PyArrayObject *prototype, NPY_ORDER order,
                     PyArray_Descr *dtype, int subok)
{
    return PyArray_NewLikeArrayWithShape(prototype, order, dtype, -1, NULL, subok);
}

/*NUMPY_API
 * Generic new array creation routine.
 */
NPY_NO_EXPORT PyObject *
PyArray_New(
        PyTypeObject *subtype, int nd, npy_intp const *dims, int type_num,
        npy_intp const *strides, void *data, int itemsize, int flags,
        PyObject *obj)
{
    PyArray_Descr *descr;
    PyObject *new;

    descr = PyArray_DescrFromType(type_num);
    if (descr == NULL) {
        return NULL;
    }
    if (PyDataType_ISUNSIZED(descr)) {
        if (itemsize < 1) {
            PyErr_SetString(PyExc_ValueError,
                            "data type must provide an itemsize");
            Py_DECREF(descr);
            return NULL;
        }
        PyArray_DESCR_REPLACE(descr);
        if (descr == NULL) {
            return NULL;
        }
        descr->elsize = itemsize;
    }
    new = PyArray_NewFromDescr(subtype, descr, nd, dims, strides,
                               data, flags, obj);
    return new;
}


NPY_NO_EXPORT PyArray_Descr *
_dtype_from_buffer_3118(PyObject *memoryview)
{
    PyArray_Descr *descr;
    Py_buffer *view = PyMemoryView_GET_BUFFER(memoryview);
    if (view->format != NULL) {
        descr = _descriptor_from_pep3118_format(view->format);
        if (descr == NULL) {
            return NULL;
        }
    }
    else {
        /* If no format is specified, just assume a byte array
         * TODO: void would make more sense here, as it wouldn't null
         *       terminate.
         */
        descr = PyArray_DescrNewFromType(NPY_STRING);
        if (descr == NULL) {
            return NULL;
        }
        descr->elsize = view->itemsize;
    }
    return descr;
}


NPY_NO_EXPORT PyObject *
_array_from_buffer_3118(PyObject *memoryview)
{
    /* PEP 3118 */
    Py_buffer *view;
    PyArray_Descr *descr = NULL;
    PyObject *r = NULL;
    int nd, flags;
    Py_ssize_t d;
    npy_intp shape[NPY_MAXDIMS], strides[NPY_MAXDIMS];

    view = PyMemoryView_GET_BUFFER(memoryview);
    nd = view->ndim;
    descr = _dtype_from_buffer_3118(memoryview);

    if (descr == NULL) {
        return NULL;
    }

    /* Sanity check */
    if (descr->elsize != view->itemsize) {
        /* Ctypes has bugs in its PEP3118 implementation, which we need to
         * work around.
         *
         * bpo-10746
         * bpo-32780
         * bpo-32782
         *
         * Note that even if the above are fixed in main, we have to drop the
         * early patch versions of python to actually make use of the fixes.
         */
        if (!npy_ctypes_check(Py_TYPE(view->obj))) {
            /* This object has no excuse for a broken PEP3118 buffer */
            PyErr_Format(
                    PyExc_RuntimeError,
                   "Item size %zd for PEP 3118 buffer format "
                    "string %s does not match the dtype %c item size %d.",
                    view->itemsize, view->format, descr->type,
                    descr->elsize);
            Py_DECREF(descr);
            return NULL;
        }

        if (PyErr_Warn(
                    PyExc_RuntimeWarning,
                    "A builtin ctypes object gave a PEP3118 format "
                    "string that does not match its itemsize, so a "
                    "best-guess will be made of the data type. "
                    "Newer versions of python may behave correctly.") < 0) {
            Py_DECREF(descr);
            return NULL;
        }

        /* Thankfully, np.dtype(ctypes_type) works in most cases.
         * For an array input, this produces a dtype containing all the
         * dimensions, so the array is now 0d.
         */
        nd = 0;
        Py_DECREF(descr);
        descr = (PyArray_Descr *)PyObject_CallFunctionObjArgs(
                (PyObject *)&PyArrayDescr_Type, Py_TYPE(view->obj), NULL);
        if (descr == NULL) {
            return NULL;
        }
        if (descr->elsize != view->len) {
            PyErr_SetString(
                    PyExc_RuntimeError,
                    "For the given ctypes object, neither the item size "
                    "computed from the PEP 3118 buffer format nor from "
                    "converting the type to a np.dtype matched the actual "
                    "size. This is a bug both in python and numpy");
            Py_DECREF(descr);
            return NULL;
        }
    }

    if (view->shape != NULL) {
        int k;
        if (nd > NPY_MAXDIMS || nd < 0) {
            PyErr_Format(PyExc_RuntimeError,
                "PEP3118 dimensions do not satisfy 0 <= ndim <= NPY_MAXDIMS");
            goto fail;
        }
        for (k = 0; k < nd; ++k) {
            shape[k] = view->shape[k];
        }
        if (view->strides != NULL) {
            for (k = 0; k < nd; ++k) {
                strides[k] = view->strides[k];
            }
        }
        else {
            d = view->len;
            for (k = 0; k < nd; ++k) {
                if (view->shape[k] != 0) {
                    d /= view->shape[k];
                }
                strides[k] = d;
            }
        }
    }
    else {
        if (nd == 1) {
            shape[0] = view->len / view->itemsize;
            strides[0] = view->itemsize;
        }
        else if (nd > 1) {
            PyErr_SetString(PyExc_RuntimeError,
                           "ndim computed from the PEP 3118 buffer format "
                           "is greater than 1, but shape is NULL.");
            goto fail;
        }
    }

    flags = NPY_ARRAY_BEHAVED & (view->readonly ? ~NPY_ARRAY_WRITEABLE : ~0);
    r = PyArray_NewFromDescrAndBase(
            &PyArray_Type, descr,
            nd, shape, strides, view->buf,
            flags, NULL, memoryview);
    return r;


fail:
    Py_XDECREF(r);
    Py_XDECREF(descr);
    return NULL;

}


/**
 * Attempts to extract an array from an array-like object.
 *
 * array-like is defined as either
 *
 * * an object implementing the PEP 3118 buffer interface;
 * * an object with __array_struct__ or __array_interface__ attributes;
 * * an object with an __array__ function.
 *
 * @param op The object to convert to an array
 * @param requested_type a requested dtype instance, may be NULL; The result
 *                       DType may be used, but is not enforced.
 * @param writeable whether the result must be writeable.
 * @param context Unused parameter, must be NULL (should be removed later).
 * @param never_copy Specifies that a copy is not allowed.
 *
 * @returns The array object, Py_NotImplemented if op is not array-like,
 *          or NULL with an error set. (A new reference to Py_NotImplemented
 *          is returned.)
 */
NPY_NO_EXPORT PyObject *
_array_from_array_like(PyObject *op,
        PyArray_Descr *requested_dtype, npy_bool writeable, PyObject *context,
        int never_copy) {
    PyObject* tmp;

    /*
     * If op supports the PEP 3118 buffer interface.
     * We skip bytes and unicode since they are considered scalars. Unicode
     * would fail but bytes would be incorrectly converted to a uint8 array.
     */
    if (PyObject_CheckBuffer(op) && !PyBytes_Check(op) && !PyUnicode_Check(op)) {
        PyObject *memoryview = PyMemoryView_FromObject(op);
        if (memoryview == NULL) {
            /* TODO: Should probably not blanket ignore errors. */
            PyErr_Clear();
        }
        else {
            tmp = _array_from_buffer_3118(memoryview);
            Py_DECREF(memoryview);
            if (tmp == NULL) {
                return NULL;
            }

            if (writeable
                && PyArray_FailUnlessWriteable(
                        (PyArrayObject *)tmp, "PEP 3118 buffer") < 0) {
                Py_DECREF(tmp);
                return NULL;
            }

            return tmp;
        }
    }

    /*
     * If op supports the __array_struct__ or __array_interface__ interface.
     */
    tmp = PyArray_FromStructInterface(op);
    if (tmp == NULL) {
        return NULL;
    }
    if (tmp == Py_NotImplemented) {
        /* Until the return, NotImplemented is always a borrowed reference*/
        tmp = PyArray_FromInterface(op);
        if (tmp == NULL) {
            return NULL;
        }
    }

    /*
     * If op supplies the __array__ function.
     * The documentation says this should produce a copy, so
     * we skip this method if writeable is true, because the intent
     * of writeable is to modify the operand.
     * XXX: If the implementation is wrong, and/or if actual
     *      usage requires this behave differently,
     *      this should be changed!
     */
    if (!writeable && tmp == Py_NotImplemented) {
        tmp = PyArray_FromArrayAttr_int(op, requested_dtype, never_copy);
        if (tmp == NULL) {
            return NULL;
        }
    }

    if (tmp != Py_NotImplemented) {
        if (writeable &&
                PyArray_FailUnlessWriteable((PyArrayObject *)tmp,
                        "array interface object") < 0) {
            Py_DECREF(tmp);
            return NULL;
        }
        return tmp;
    }

    /* Until here Py_NotImplemented was borrowed */
    Py_INCREF(Py_NotImplemented);
    return Py_NotImplemented;
}


/*NUMPY_API*/
NPY_NO_EXPORT int
PyArray_GetArrayParamsFromObject(PyObject *NPY_UNUSED(op),
        PyArray_Descr *NPY_UNUSED(requested_dtype),
        npy_bool NPY_UNUSED(writeable),
        PyArray_Descr **NPY_UNUSED(out_dtype),
        int *NPY_UNUSED(out_ndim), npy_intp *NPY_UNUSED(out_dims),
        PyArrayObject **NPY_UNUSED(out_arr), PyObject *NPY_UNUSED(context))
{
    /* Deprecated in NumPy 1.19, removed in NumPy 1.20. */
    PyErr_SetString(PyExc_RuntimeError,
            "PyArray_GetArrayParamsFromObject() C-API function is removed "
            "`PyArray_FromAny()` should be used at this time.  New C-API "
            "may be exposed in the future (please do request this if it "
            "would help you).");
    return -1;
}


/*
 * This function is a legacy implementation to retain subarray dtype
 * behaviour in array coercion. The behaviour here makes sense if tuples
 * of matching dimensionality are being coerced. Due to the difficulty
 * that the result is ill-defined for lists of array-likes, this is deprecated.
 *
 * WARNING: Do not use this function, it exists purely to support a deprecated
 *          code path.
 */
static int
setArrayFromSequence(PyArrayObject *a, PyObject *s,
                        int dim, PyArrayObject * dst)
{
    Py_ssize_t i, slen;
    int res = -1;

    /* first recursion, view equal destination */
    if (dst == NULL)
        dst = a;

    /*
     * This code is to ensure that the sequence access below will
     * return a lower-dimensional sequence.
     */

    /* INCREF on entry DECREF on exit */
    Py_INCREF(s);

    PyObject *seq = NULL;

    if (PyArray_Check(s)) {
        if (!(PyArray_CheckExact(s))) {
            /*
             * make sure a base-class array is used so that the dimensionality
             * reduction assumption is correct.
             */
            /* This will DECREF(s) if replaced */
            s = PyArray_EnsureArray(s);
            if (s == NULL) {
                goto fail;
            }
        }

        /* dst points to correct array subsection */
        if (PyArray_CopyInto(dst, (PyArrayObject *)s) < 0) {
            goto fail;
        }

        Py_DECREF(s);
        return 0;
    }

    if (dim > PyArray_NDIM(a)) {
        PyErr_Format(PyExc_ValueError,
                 "setArrayFromSequence: sequence/array dimensions mismatch.");
        goto fail;
    }

    /* Try __array__ before using s as a sequence */
    PyObject *tmp = _array_from_array_like(s, NULL, 0, NULL, 0);
    if (tmp == NULL) {
        goto fail;
    }
    else if (tmp == Py_NotImplemented) {
        Py_DECREF(tmp);
    }
    else {
        int r = PyArray_CopyInto(dst, (PyArrayObject *)tmp);
        Py_DECREF(tmp);
        if (r < 0) {
            goto fail;
        }
        Py_DECREF(s);
        return 0;
    }

    seq = PySequence_Fast(s, "Could not convert object to sequence");
    if (seq == NULL) {
        goto fail;
    }
    slen = PySequence_Fast_GET_SIZE(seq);

    /*
     * Either the dimensions match, or the sequence has length 1 and can
     * be broadcast to the destination.
     */
    if (slen != PyArray_DIMS(a)[dim] && slen != 1) {
        PyErr_Format(PyExc_ValueError,
                 "cannot copy sequence with size %zd to array axis "
                 "with dimension %" NPY_INTP_FMT, slen, PyArray_DIMS(a)[dim]);
        goto fail;
    }

    /* Broadcast the one element from the sequence to all the outputs */
    if (slen == 1) {
        PyObject *o = PySequence_Fast_GET_ITEM(seq, 0);
        npy_intp alen = PyArray_DIM(a, dim);

        for (i = 0; i < alen; i++) {
            if ((PyArray_NDIM(a) - dim) > 1) {
                PyArrayObject * tmp =
                    (PyArrayObject *)array_item_asarray(dst, i);
                if (tmp == NULL) {
                    goto fail;
                }

                res = setArrayFromSequence(a, o, dim+1, tmp);
                Py_DECREF(tmp);
            }
            else {
                char * b = (PyArray_BYTES(dst) + i * PyArray_STRIDES(dst)[0]);
                res = PyArray_SETITEM(dst, b, o);
            }
            if (res < 0) {
                goto fail;
            }
        }
    }
    /* Copy element by element */
    else {
        for (i = 0; i < slen; i++) {
            PyObject * o = PySequence_Fast_GET_ITEM(seq, i);
            if ((PyArray_NDIM(a) - dim) > 1) {
                PyArrayObject * tmp =
                    (PyArrayObject *)array_item_asarray(dst, i);
                if (tmp == NULL) {
                    goto fail;
                }

                res = setArrayFromSequence(a, o, dim+1, tmp);
                Py_DECREF(tmp);
            }
            else {
                char * b = (PyArray_BYTES(dst) + i * PyArray_STRIDES(dst)[0]);
                res = PyArray_SETITEM(dst, b, o);
            }
            if (res < 0) {
                goto fail;
            }
        }
    }

    Py_DECREF(seq);
    Py_DECREF(s);
    return 0;

 fail:
    Py_XDECREF(seq);
    Py_DECREF(s);
    return res;
}



/*NUMPY_API
 * Does not check for NPY_ARRAY_ENSURECOPY and NPY_ARRAY_NOTSWAPPED in flags
 * Steals a reference to newtype --- which can be NULL
 */
NPY_NO_EXPORT PyObject *
PyArray_FromAny(PyObject *op, PyArray_Descr *newtype, int min_depth,
                int max_depth, int flags, PyObject *context)
{
    /*
     * This is the main code to make a NumPy array from a Python
     * Object.  It is called from many different places.
     */
    PyArrayObject *arr = NULL, *ret;
    PyArray_Descr *dtype = NULL;
    coercion_cache_obj *cache = NULL;
    int ndim = 0;
    npy_intp dims[NPY_MAXDIMS];

    if (context != NULL) {
        PyErr_SetString(PyExc_RuntimeError, "'context' must be NULL");
        return NULL;
    }

    PyArray_Descr *fixed_descriptor;
    PyArray_DTypeMeta *fixed_DType;
    if (PyArray_ExtractDTypeAndDescriptor((PyObject *)newtype,
            &fixed_descriptor, &fixed_DType) < 0) {
        Py_XDECREF(newtype);
        return NULL;
    }
    Py_XDECREF(newtype);

    ndim = PyArray_DiscoverDTypeAndShape(op,
            NPY_MAXDIMS, dims, &cache, fixed_DType, fixed_descriptor, &dtype,
            flags & NPY_ARRAY_ENSURENOCOPY);

    Py_XDECREF(fixed_descriptor);
    Py_XDECREF(fixed_DType);
    if (ndim < 0) {
        return NULL;
    }

    if (NPY_UNLIKELY(fixed_descriptor != NULL && PyDataType_HASSUBARRAY(dtype))) {
        /*
         * When a subarray dtype was passed in, its dimensions are appended
         * to the array dimension (causing a dimension mismatch).
         * There is a problem with that, because if we coerce from non-arrays
         * we do this correctly by element (as defined by tuples), but for
         * arrays we first append the dimensions and then assign to the base
         * dtype and then assign which causes the problem.
         *
         * Thus, we check if there is an array included, in that case we
         * give a FutureWarning.
         * When the warning is removed, PyArray_Pack will have to ensure
         * that that it does not append the dimensions when creating the
         * subarrays to assign `arr[0] = obj[0]`.
         */
        int includes_array = 0;
        if (cache != NULL) {
            /* This is not ideal, but it is a pretty special case */
            coercion_cache_obj *next = cache;
            while (next != NULL) {
                if (!next->sequence) {
                    includes_array = 1;
                    break;
                }
                next = next->next;
            }
        }
        if (includes_array) {
            npy_free_coercion_cache(cache);

            ret = (PyArrayObject *) PyArray_NewFromDescr(
                    &PyArray_Type, dtype, ndim, dims, NULL, NULL,
                    flags & NPY_ARRAY_F_CONTIGUOUS, NULL);
            if (ret == NULL) {
                return NULL;
            }
            assert(PyArray_NDIM(ret) != ndim);

            /* NumPy 1.20, 2020-10-01 */
            if (DEPRECATE_FUTUREWARNING(
                    "creating an array with a subarray dtype will behave "
                    "differently when the `np.array()` (or `asarray`, etc.) "
                    "call includes an array or array object.\n"
                    "If you are converting a single array or a list of arrays,"
                    "you can opt-in to the future behaviour using:\n"
                    "    np.array(arr, dtype=np.dtype(['f', dtype]))['f']\n"
                    "    np.array([arr1, arr2], dtype=np.dtype(['f', dtype]))['f']\n"
                    "\n"
                    "By including a new field and indexing it after the "
                    "conversion.\n"
                    "This may lead to a different result or to current failures "
                    "succeeding.  (FutureWarning since NumPy 1.20)") < 0) {
                Py_DECREF(ret);
                return NULL;
            }

            if (setArrayFromSequence(ret, op, 0, NULL) < 0) {
                Py_DECREF(ret);
                return NULL;
            }
            return (PyObject *)ret;
        }
    }

    if (dtype == NULL) {
        dtype = PyArray_DescrFromType(NPY_DEFAULT_TYPE);
    }

    if (min_depth != 0 && ndim < min_depth) {
        PyErr_SetString(PyExc_ValueError,
                "object of too small depth for desired array");
        Py_DECREF(dtype);
        npy_free_coercion_cache(cache);
        return NULL;
    }
    if (max_depth != 0 && ndim > max_depth) {
        PyErr_SetString(PyExc_ValueError,
                "object too deep for desired array");
        Py_DECREF(dtype);
        npy_free_coercion_cache(cache);
        return NULL;
    }

    /* Got the correct parameters, but the cache may already hold the result */
    if (cache != NULL && !(cache->sequence)) {
        /*
         * There is only a single array-like and it was converted, it
         * may still have the incorrect type, but that is handled below.
         */
        assert(cache->converted_obj == op);
        arr = (PyArrayObject *)(cache->arr_or_sequence);
        /* we may need to cast or assert flags (e.g. copy) */
        PyObject *res = PyArray_FromArray(arr, dtype, flags);
        npy_unlink_coercion_cache(cache);
        return res;
    }
    else if (cache == NULL && PyArray_IsScalar(op, Void) &&
            !(((PyVoidScalarObject *)op)->flags & NPY_ARRAY_OWNDATA) &&
            newtype == NULL) {
        /*
         * Special case, we return a *view* into void scalars, mainly to
         * allow things similar to the "reversed" assignment:
         *    arr[indx]["field"] = val  # instead of arr["field"][indx] = val
         *
         * It is unclear that this is necessary in this particular code path.
         * Note that this path is only activated when the user did _not_
         * provide a dtype (newtype is NULL).
         */
        assert(ndim == 0);

        return PyArray_NewFromDescrAndBase(
                &PyArray_Type, dtype,
                0, NULL, NULL,
                ((PyVoidScalarObject *)op)->obval,
                ((PyVoidScalarObject *)op)->flags,
                NULL, op);
    }
    /*
     * If we got this far, we definitely have to create a copy, since we are
     * converting either from a scalar (cache == NULL) or a (nested) sequence.
     */
    if (flags & NPY_ARRAY_ENSURENOCOPY ) {
        PyErr_SetString(PyExc_ValueError,
                "Unable to avoid copy while creating an array.");
        Py_DECREF(dtype);
        npy_free_coercion_cache(cache);
        return NULL;
    }

    if (cache == NULL && newtype != NULL &&
            PyDataType_ISSIGNED(newtype) && PyArray_IsScalar(op, Generic)) {
        assert(ndim == 0);
        /*
         * This is an (possible) inconsistency where:
         *
         *     np.array(np.float64(np.nan), dtype=np.int64)
         *
         * behaves differently from:
         *
         *     np.array([np.float64(np.nan)], dtype=np.int64)
         *     arr1d_int64[0] = np.float64(np.nan)
         *     np.array(np.array(np.nan), dtype=np.int64)
         *
         * by not raising an error instead of using typical casting.
         * The error is desirable, but to always error seems like a
         * larger change to be considered at some other time and it is
         * undesirable that 0-D arrays behave differently from scalars.
         * This retains the behaviour, largely due to issues in pandas
         * which relied on a try/except (although hopefully that will
         * have a better solution at some point):
         * https://github.com/pandas-dev/pandas/issues/35481
         */
        return PyArray_FromScalar(op, dtype);
    }

    /* There was no array (or array-like) passed in directly. */
    if ((flags & NPY_ARRAY_WRITEBACKIFCOPY) ||
            (flags & NPY_ARRAY_UPDATEIFCOPY)) {
        PyErr_SetString(PyExc_TypeError,
                        "WRITEBACKIFCOPY used for non-array input.");
        Py_DECREF(dtype);
        npy_free_coercion_cache(cache);
        return NULL;
    }

    /* Create a new array and copy the data */
    Py_INCREF(dtype);  /* hold on in case of a subarray that is replaced */
    ret = (PyArrayObject *)PyArray_NewFromDescr(
            &PyArray_Type, dtype, ndim, dims, NULL, NULL,
            flags&NPY_ARRAY_F_CONTIGUOUS, NULL);
    if (ret == NULL) {
        npy_free_coercion_cache(cache);
        Py_DECREF(dtype);
        return NULL;
    }
    if (ndim == PyArray_NDIM(ret)) {
        /*
         * Appending of dimensions did not occur, so use the actual dtype
         * below. This is relevant for S0 or U0 which can be replaced with
         * S1 or U1, although that should likely change.
         */
        Py_SETREF(dtype, PyArray_DESCR(ret));
        Py_INCREF(dtype);
    }

    if (cache == NULL) {
        /* This is a single item. Set it directly. */
        assert(ndim == 0);

        if (PyArray_Pack(dtype, PyArray_BYTES(ret), op) < 0) {
            Py_DECREF(dtype);
            Py_DECREF(ret);
            return NULL;
        }
        Py_DECREF(dtype);
        return (PyObject *)ret;
    }
    assert(ndim != 0);
    assert(op == cache->converted_obj);

    /* Decrease the number of dimensions to the detected ones */
    int out_ndim = PyArray_NDIM(ret);
    PyArray_Descr *out_descr = PyArray_DESCR(ret);
    ((PyArrayObject_fields *)ret)->nd = ndim;
    ((PyArrayObject_fields *)ret)->descr = dtype;

    int success = PyArray_AssignFromCache(ret, cache);

    ((PyArrayObject_fields *)ret)->nd = out_ndim;
    ((PyArrayObject_fields *)ret)->descr = out_descr;
    Py_DECREF(dtype);
    if (success < 0) {
        Py_DECREF(ret);
        return NULL;
    }
    return (PyObject *)ret;
}

/*
 * flags is any of
 * NPY_ARRAY_C_CONTIGUOUS (formerly CONTIGUOUS),
 * NPY_ARRAY_F_CONTIGUOUS (formerly FORTRAN),
 * NPY_ARRAY_ALIGNED,
 * NPY_ARRAY_WRITEABLE,
 * NPY_ARRAY_NOTSWAPPED,
 * NPY_ARRAY_ENSURECOPY,
 * NPY_ARRAY_UPDATEIFCOPY,
 * NPY_ARRAY_WRITEBACKIFCOPY,
 * NPY_ARRAY_FORCECAST,
 * NPY_ARRAY_ENSUREARRAY,
 * NPY_ARRAY_ELEMENTSTRIDES,
 * NPY_ARRAY_ENSURENOCOPY
 *
 * or'd (|) together
 *
 * Any of these flags present means that the returned array should
 * guarantee that aspect of the array.  Otherwise the returned array
 * won't guarantee it -- it will depend on the object as to whether or
 * not it has such features.
 *
 * Note that NPY_ARRAY_ENSURECOPY is enough
 * to guarantee NPY_ARRAY_C_CONTIGUOUS, NPY_ARRAY_ALIGNED and
 * NPY_ARRAY_WRITEABLE and therefore it is redundant to include
 * those as well.
 *
 * NPY_ARRAY_BEHAVED == NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE
 * NPY_ARRAY_CARRAY = NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_BEHAVED
 * NPY_ARRAY_FARRAY = NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_BEHAVED
 *
 * NPY_ARRAY_F_CONTIGUOUS can be set in the FLAGS to request a FORTRAN array.
 * Fortran arrays are always behaved (aligned,
 * notswapped, and writeable) and not (C) CONTIGUOUS (if > 1d).
 *
 * NPY_ARRAY_UPDATEIFCOPY is deprecated in favor of
 * NPY_ARRAY_WRITEBACKIFCOPY in 1.14

 * NPY_ARRAY_WRITEBACKIFCOPY flag sets this flag in the returned
 * array if a copy is made and the base argument points to the (possibly)
 * misbehaved array. Before returning to python, PyArray_ResolveWritebackIfCopy
 * must be called to update the contents of the original array from the copy.
 *
 * NPY_ARRAY_FORCECAST will cause a cast to occur regardless of whether or not
 * it is safe.
 *
 * context is passed through to PyArray_GetArrayParamsFromObject
 */

/*NUMPY_API
 * steals a reference to descr -- accepts NULL
 */
NPY_NO_EXPORT PyObject *
PyArray_CheckFromAny(PyObject *op, PyArray_Descr *descr, int min_depth,
                     int max_depth, int requires, PyObject *context)
{
    PyObject *obj;
    if (requires & NPY_ARRAY_NOTSWAPPED) {
        if (!descr && PyArray_Check(op) &&
                PyArray_ISBYTESWAPPED((PyArrayObject* )op)) {
            descr = PyArray_DescrNew(PyArray_DESCR((PyArrayObject *)op));
            if (descr == NULL) {
                return NULL;
            }
        }
        else if (descr && !PyArray_ISNBO(descr->byteorder)) {
            PyArray_DESCR_REPLACE(descr);
        }
        if (descr && descr->byteorder != NPY_IGNORE) {
            descr->byteorder = NPY_NATIVE;
        }
    }

    obj = PyArray_FromAny(op, descr, min_depth, max_depth, requires, context);
    if (obj == NULL) {
        return NULL;
    }

    if ((requires & NPY_ARRAY_ELEMENTSTRIDES)
            && !PyArray_ElementStrides(obj)) {
        PyObject *ret;
        if (requires & NPY_ARRAY_ENSURENOCOPY) {
            PyErr_SetString(PyExc_ValueError,
                    "Unable to avoid copy while creating a new array.");
            return NULL;
        }
        ret = PyArray_NewCopy((PyArrayObject *)obj, NPY_ANYORDER);
        Py_DECREF(obj);
        obj = ret;
    }
    return obj;
}


/*NUMPY_API
 * steals reference to newtype --- acc. NULL
 */
NPY_NO_EXPORT PyObject *
PyArray_FromArray(PyArrayObject *arr, PyArray_Descr *newtype, int flags)
{

    PyArrayObject *ret = NULL;
    int copy = 0;
    int arrflags;
    PyArray_Descr *oldtype;
    NPY_CASTING casting = NPY_SAFE_CASTING;

    oldtype = PyArray_DESCR(arr);
    if (newtype == NULL) {
        /*
         * Check if object is of array with Null newtype.
         * If so return it directly instead of checking for casting.
         */
        if (flags == 0) {
            Py_INCREF(arr);
            return (PyObject *)arr;
        }
        newtype = oldtype;
        Py_INCREF(oldtype);
    }
    else if (PyDataType_ISUNSIZED(newtype)) {
        PyArray_DESCR_REPLACE(newtype);
        if (newtype == NULL) {
            return NULL;
        }
        newtype->elsize = oldtype->elsize;
    }

    /* If the casting if forced, use the 'unsafe' casting rule */
    if (flags & NPY_ARRAY_FORCECAST) {
        casting = NPY_UNSAFE_CASTING;
    }

    /* Raise an error if the casting rule isn't followed */
    if (!PyArray_CanCastArrayTo(arr, newtype, casting)) {
        PyErr_Clear();
        npy_set_invalid_cast_error(
                PyArray_DESCR(arr), newtype, casting, PyArray_NDIM(arr) == 0);
        Py_DECREF(newtype);
        return NULL;
    }

    arrflags = PyArray_FLAGS(arr);
           /* If a guaranteed copy was requested */
    copy = (flags & NPY_ARRAY_ENSURECOPY) ||
           /* If C contiguous was requested, and arr is not */
           ((flags & NPY_ARRAY_C_CONTIGUOUS) &&
                   (!(arrflags & NPY_ARRAY_C_CONTIGUOUS))) ||
           /* If an aligned array was requested, and arr is not */
           ((flags & NPY_ARRAY_ALIGNED) &&
                   (!(arrflags & NPY_ARRAY_ALIGNED))) ||
           /* If a Fortran contiguous array was requested, and arr is not */
           ((flags & NPY_ARRAY_F_CONTIGUOUS) &&
                   (!(arrflags & NPY_ARRAY_F_CONTIGUOUS))) ||
           /* If a writeable array was requested, and arr is not */
           ((flags & NPY_ARRAY_WRITEABLE) &&
                   (!(arrflags & NPY_ARRAY_WRITEABLE))) ||
           !PyArray_EquivTypes(oldtype, newtype);

    if (copy) {
        if (flags & NPY_ARRAY_ENSURENOCOPY ) {
            PyErr_SetString(PyExc_ValueError,
                    "Unable to avoid copy while creating an array from given array.");
            Py_DECREF(newtype);
            return NULL;
        }

        NPY_ORDER order = NPY_KEEPORDER;
        int subok = 1;

        /* Set the order for the copy being made based on the flags */
        if (flags & NPY_ARRAY_F_CONTIGUOUS) {
            order = NPY_FORTRANORDER;
        }
        else if (flags & NPY_ARRAY_C_CONTIGUOUS) {
            order = NPY_CORDER;
        }

        if ((flags & NPY_ARRAY_ENSUREARRAY)) {
            subok = 0;
        }
        ret = (PyArrayObject *)PyArray_NewLikeArray(arr, order,
                                                    newtype, subok);
        if (ret == NULL) {
            return NULL;
        }

        if (PyArray_CopyInto(ret, arr) < 0) {
            Py_DECREF(ret);
            return NULL;
        }

        if (flags & NPY_ARRAY_UPDATEIFCOPY) {
            /* This is the ONLY place the NPY_ARRAY_UPDATEIFCOPY flag
             * is still used.
             * Can be deleted once the flag itself is removed
             */

            /* 2017-Nov-10 1.14 */
            if (DEPRECATE(
                    "NPY_ARRAY_UPDATEIFCOPY, NPY_ARRAY_INOUT_ARRAY, and "
                    "NPY_ARRAY_INOUT_FARRAY are deprecated, use NPY_WRITEBACKIFCOPY, "
                    "NPY_ARRAY_INOUT_ARRAY2, or NPY_ARRAY_INOUT_FARRAY2 respectively "
                    "instead, and call PyArray_ResolveWritebackIfCopy before the "
                    "array is deallocated, i.e. before the last call to Py_DECREF.") < 0) {
                Py_DECREF(ret);
                return NULL;
            }
            Py_INCREF(arr);
            if (PyArray_SetWritebackIfCopyBase(ret, arr) < 0) {
                Py_DECREF(ret);
                return NULL;
            }
            PyArray_ENABLEFLAGS(ret, NPY_ARRAY_UPDATEIFCOPY);
            PyArray_CLEARFLAGS(ret, NPY_ARRAY_WRITEBACKIFCOPY);
        }
        else if (flags & NPY_ARRAY_WRITEBACKIFCOPY) {
            Py_INCREF(arr);
            if (PyArray_SetWritebackIfCopyBase(ret, arr) < 0) {
                Py_DECREF(ret);
                return NULL;
            }
        }
    }
    /*
     * If no copy then take an appropriate view if necessary, or
     * just return a reference to ret itself.
     */
    else {
        int needview = ((flags & NPY_ARRAY_ENSUREARRAY) &&
                        !PyArray_CheckExact(arr));

        Py_DECREF(newtype);
        if (needview) {
            PyTypeObject *subtype = NULL;

            if (flags & NPY_ARRAY_ENSUREARRAY) {
                subtype = &PyArray_Type;
            }
            ret = (PyArrayObject *)PyArray_View(arr, NULL, subtype);
            if (ret == NULL) {
                return NULL;
            }
        }
        else {
            Py_INCREF(arr);
            ret = arr;
        }
    }

    return (PyObject *)ret;
}

/*NUMPY_API */
NPY_NO_EXPORT PyObject *
PyArray_FromStructInterface(PyObject *input)
{
    PyArray_Descr *thetype = NULL;
    PyArrayInterface *inter;
    PyObject *attr;
    char endian = NPY_NATBYTE;

    attr = PyArray_LookupSpecial_OnInstance(input, "__array_struct__");
    if (attr == NULL) {
        if (PyErr_Occurred()) {
            return NULL;
        } else {
            return Py_NotImplemented;
        }
    }
    if (!PyCapsule_CheckExact(attr)) {
        if (PyType_Check(input) && PyObject_HasAttrString(attr, "__get__")) {
            /*
             * If the input is a class `attr` should be a property-like object.
             * This cannot be interpreted as an array, but is a valid.
             * (Needed due to the lookup being on the instance rather than type)
             */
            Py_DECREF(attr);
            return Py_NotImplemented;
        }
        goto fail;
    }
    inter = PyCapsule_GetPointer(attr, NULL);
    if (inter == NULL) {
        goto fail;
    }
    if (inter->two != 2) {
        goto fail;
    }
    if ((inter->flags & NPY_ARRAY_NOTSWAPPED) != NPY_ARRAY_NOTSWAPPED) {
        endian = NPY_OPPBYTE;
        inter->flags &= ~NPY_ARRAY_NOTSWAPPED;
    }

    if (inter->flags & NPY_ARR_HAS_DESCR) {
        if (PyArray_DescrConverter(inter->descr, &thetype) == NPY_FAIL) {
            thetype = NULL;
            PyErr_Clear();
        }
    }

    if (thetype == NULL) {
        PyObject *type_str = PyUnicode_FromFormat(
            "%c%c%d", endian, inter->typekind, inter->itemsize);
        if (type_str == NULL) {
            Py_DECREF(attr);
            return NULL;
        }
        int ok = PyArray_DescrConverter(type_str, &thetype);
        Py_DECREF(type_str);
        if (ok != NPY_SUCCEED) {
            Py_DECREF(attr);
            return NULL;
        }
    }

    PyObject *ret = PyArray_NewFromDescrAndBase(
            &PyArray_Type, thetype,
            inter->nd, inter->shape, inter->strides, inter->data,
            inter->flags, NULL, input);
    Py_DECREF(attr);
    return ret;

 fail:
    PyErr_SetString(PyExc_ValueError, "invalid __array_struct__");
    Py_DECREF(attr);
    return NULL;
}

/*
 * Checks if the object in descr is the default 'descr' member for the
 * __array_interface__ dictionary with 'typestr' member typestr.
 */
NPY_NO_EXPORT int
_is_default_descr(PyObject *descr, PyObject *typestr) {
    if (!PyList_Check(descr) || PyList_GET_SIZE(descr) != 1) {
        return 0;
    }
    PyObject *tuple = PyList_GET_ITEM(descr, 0);
    if (!(PyTuple_Check(tuple) && PyTuple_GET_SIZE(tuple) == 2)) {
        return 0;
    }
    PyObject *name = PyTuple_GET_ITEM(tuple, 0);
    if (!(PyUnicode_Check(name) && PyUnicode_GetLength(name) == 0)) {
        return 0;
    }
    PyObject *typestr2 = PyTuple_GET_ITEM(tuple, 1);
    return PyObject_RichCompareBool(typestr, typestr2, Py_EQ);
}


/*
 * A helper function to transition away from ignoring errors during
 * special attribute lookups during array coercion.
 */
static NPY_INLINE int
deprecated_lookup_error_clearing(PyTypeObject *type, char *attribute)
{
    PyObject *exc_type, *exc_value, *traceback;
    PyErr_Fetch(&exc_type, &exc_value, &traceback);

    /* DEPRECATED 2021-05-12, NumPy 1.21. */
    int res = PyErr_WarnFormat(PyExc_DeprecationWarning, 1,
            "An exception was ignored while fetching the attribute `%s` from "
            "an object of type '%s'.  With the exception of `AttributeError` "
            "NumPy will always raise this exception in the future.  Raise this "
            "deprecation warning to see the original exception. "
            "(Warning added NumPy 1.21)", attribute, type->tp_name);

    if (res < 0) {
        npy_PyErr_ChainExceptionsCause(exc_type, exc_value, traceback);
        return -1;
    }
    else {
        /* `PyErr_Fetch` cleared the original error, delete the references */
        Py_DECREF(exc_type);
        Py_XDECREF(exc_value);
        Py_XDECREF(traceback);
        return 0;
    }
}


/*NUMPY_API*/
NPY_NO_EXPORT PyObject *
PyArray_FromInterface(PyObject *origin)
{
    PyObject *iface = NULL;
    PyObject *attr = NULL;
    PyObject *base = NULL;
    PyArrayObject *ret;
    PyArray_Descr *dtype = NULL;
    char *data = NULL;
    Py_buffer view;
    int i, n;
    npy_intp dims[NPY_MAXDIMS], strides[NPY_MAXDIMS];
    int dataflags = NPY_ARRAY_BEHAVED;

    iface = PyArray_LookupSpecial_OnInstance(origin, "__array_interface__");

    if (iface == NULL) {
        if (PyErr_Occurred()) {
            if (PyErr_ExceptionMatches(PyExc_RecursionError) ||
                    PyErr_ExceptionMatches(PyExc_MemoryError)) {
                /* RecursionError and MemoryError are considered fatal */
                return NULL;
            }
            if (deprecated_lookup_error_clearing(
                    Py_TYPE(origin), "__array_interface__") < 0) {
                return NULL;
            }
        }
        return Py_NotImplemented;
    }
    if (!PyDict_Check(iface)) {
        if (PyType_Check(origin) && PyObject_HasAttrString(iface, "__get__")) {
            /*
             * If the input is a class `iface` should be a property-like object.
             * This cannot be interpreted as an array, but is a valid.
             * (Needed due to the lookup being on the instance rather than type)
             */
            Py_DECREF(iface);
            return Py_NotImplemented;
        }

        Py_DECREF(iface);
        PyErr_SetString(PyExc_ValueError,
                "Invalid __array_interface__ value, must be a dict");
        return NULL;
    }

    /* Get type string from interface specification */
    attr = _PyDict_GetItemStringWithError(iface, "typestr");
    if (attr == NULL) {
        Py_DECREF(iface);
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_ValueError,
                    "Missing __array_interface__ typestr");
        }
        return NULL;
    }

    /* allow bytes for backwards compatibility */
    if (!PyBytes_Check(attr) && !PyUnicode_Check(attr)) {
        PyErr_SetString(PyExc_TypeError,
                    "__array_interface__ typestr must be a string");
        goto fail;
    }

    /* Get dtype from type string */
    if (PyArray_DescrConverter(attr, &dtype) != NPY_SUCCEED) {
        goto fail;
    }

    /*
     * If the dtype is NPY_VOID, see if there is extra information in
     * the 'descr' attribute.
     */
    if (dtype->type_num == NPY_VOID) {
        PyObject *descr = _PyDict_GetItemStringWithError(iface, "descr");
        if (descr == NULL && PyErr_Occurred()) {
            goto fail;
        }
        PyArray_Descr *new_dtype = NULL;
        if (descr != NULL) {
            int is_default = _is_default_descr(descr, attr);
            if (is_default < 0) {
                goto fail;
            }
            if (!is_default) {
                if (PyArray_DescrConverter2(descr, &new_dtype) != NPY_SUCCEED) {
                    goto fail;
                }
                if (new_dtype != NULL) {
                    Py_DECREF(dtype);
                    dtype = new_dtype;
                }
            }

        }

    }

    /* Get shape tuple from interface specification */
    attr = _PyDict_GetItemStringWithError(iface, "shape");
    if (attr == NULL) {
        if (PyErr_Occurred()) {
            return NULL;
        }
        /* Shape must be specified when 'data' is specified */
        PyObject *data = _PyDict_GetItemStringWithError(iface, "data");
        if (data == NULL && PyErr_Occurred()) {
            return NULL;
        }
        else if (data != NULL) {
            Py_DECREF(iface);
            PyErr_SetString(PyExc_ValueError,
                    "Missing __array_interface__ shape");
            return NULL;
        }
        /* Assume shape as scalar otherwise */
        else {
            /* NOTE: pointers to data and base should be NULL */
            n = dims[0] = 0;
        }
    }
    /* Make sure 'shape' is a tuple */
    else if (!PyTuple_Check(attr)) {
        PyErr_SetString(PyExc_TypeError,
                "shape must be a tuple");
        goto fail;
    }
    /* Get dimensions from shape tuple */
    else {
        n = PyTuple_GET_SIZE(attr);
        for (i = 0; i < n; i++) {
            PyObject *tmp = PyTuple_GET_ITEM(attr, i);
            dims[i] = PyArray_PyIntAsIntp(tmp);
            if (error_converting(dims[i])) {
                goto fail;
            }
        }
    }

    /* Get data buffer from interface specification */
    attr = _PyDict_GetItemStringWithError(iface, "data");
    if (attr == NULL && PyErr_Occurred()){
        return NULL;
    }

    /* Case for data access through pointer */
    if (attr && PyTuple_Check(attr)) {
        PyObject *dataptr;
        if (PyTuple_GET_SIZE(attr) != 2) {
            PyErr_SetString(PyExc_TypeError,
                    "__array_interface__ data must be a 2-tuple with "
                    "(data pointer integer, read-only flag)");
            goto fail;
        }
        dataptr = PyTuple_GET_ITEM(attr, 0);
        if (PyLong_Check(dataptr)) {
            data = PyLong_AsVoidPtr(dataptr);
            if (data == NULL && PyErr_Occurred()) {
                goto fail;
            }
        }
        else {
            PyErr_SetString(PyExc_TypeError,
                    "first element of __array_interface__ data tuple "
                    "must be an integer.");
            goto fail;
        }
        if (PyObject_IsTrue(PyTuple_GET_ITEM(attr,1))) {
            dataflags &= ~NPY_ARRAY_WRITEABLE;
        }
        base = origin;
    }

    /* Case for data access through buffer */
    else if (attr) {
        if (attr != Py_None) {
            base = attr;
        }
        else {
            base = origin;
        }
        if (PyObject_GetBuffer(base, &view,
                    PyBUF_WRITABLE|PyBUF_SIMPLE) < 0) {
            PyErr_Clear();
            if (PyObject_GetBuffer(base, &view,
                        PyBUF_SIMPLE) < 0) {
                goto fail;
            }
            dataflags &= ~NPY_ARRAY_WRITEABLE;
        }
        data = (char *)view.buf;
        /*
         * In Python 3 both of the deprecated functions PyObject_AsWriteBuffer and
         * PyObject_AsReadBuffer that this code replaces release the buffer. It is
         * up to the object that supplies the buffer to guarantee that the buffer
         * sticks around after the release.
         */
        PyBuffer_Release(&view);

        /* Get offset number from interface specification */
        attr = _PyDict_GetItemStringWithError(iface, "offset");
        if (attr == NULL && PyErr_Occurred()) {
            goto fail;
        }
        else if (attr) {
            npy_longlong num = PyLong_AsLongLong(attr);
            if (error_converting(num)) {
                PyErr_SetString(PyExc_TypeError,
                        "__array_interface__ offset must be an integer");
                goto fail;
            }
            data += num;
        }
    }

    ret = (PyArrayObject *)PyArray_NewFromDescrAndBase(
            &PyArray_Type, dtype,
            n, dims, NULL, data,
            dataflags, NULL, base);
    /*
     * Ref to dtype was stolen by PyArray_NewFromDescrAndBase
     * Prevent DECREFing dtype in fail codepath by setting to NULL
     */
    dtype = NULL;
    if (ret == NULL) {
        goto fail;
    }
    if (data == NULL) {
        if (PyArray_SIZE(ret) > 1) {
            PyErr_SetString(PyExc_ValueError,
                    "cannot coerce scalar to array with size > 1");
            Py_DECREF(ret);
            goto fail;
        }
        if (PyArray_SETITEM(ret, PyArray_DATA(ret), origin) < 0) {
            Py_DECREF(ret);
            goto fail;
        }
    }
    attr = _PyDict_GetItemStringWithError(iface, "strides");
    if (attr == NULL && PyErr_Occurred()){
        return NULL;
    }
    if (attr != NULL && attr != Py_None) {
        if (!PyTuple_Check(attr)) {
            PyErr_SetString(PyExc_TypeError,
                    "strides must be a tuple");
            Py_DECREF(ret);
            goto fail;
        }
        if (n != PyTuple_GET_SIZE(attr)) {
            PyErr_SetString(PyExc_ValueError,
                    "mismatch in length of strides and shape");
            Py_DECREF(ret);
            goto fail;
        }
        for (i = 0; i < n; i++) {
            PyObject *tmp = PyTuple_GET_ITEM(attr, i);
            strides[i] = PyArray_PyIntAsIntp(tmp);
            if (error_converting(strides[i])) {
                Py_DECREF(ret);
                goto fail;
            }
        }
        if (n) {
            memcpy(PyArray_STRIDES(ret), strides, n*sizeof(npy_intp));
        }
    }
    PyArray_UpdateFlags(ret, NPY_ARRAY_UPDATE_ALL);
    Py_DECREF(iface);
    return (PyObject *)ret;

 fail:
    Py_XDECREF(dtype);
    Py_XDECREF(iface);
    return NULL;
}


/**
 * Check for an __array__ attribute and call it when it exists.
 *
 *  .. warning:
 *      If returned, `NotImplemented` is borrowed and must not be Decref'd
 *
 * @param op The Python object to convert to an array.
 * @param descr The desired `arr.dtype`, passed into the `__array__` call,
 *        as information but is not checked/enforced!
 * @param never_copy Specifies that a copy is not allowed.
 *        NOTE: Currently, this means an error is raised instead of calling
 *        `op.__array__()`.  In the future we could call for example call
 *        `op.__array__(never_copy=True)` instead.
 * @returns NotImplemented if `__array__` is not defined or a NumPy array
 *          (or subclass).  On error, return NULL.
 */
NPY_NO_EXPORT PyObject *
PyArray_FromArrayAttr_int(
        PyObject *op, PyArray_Descr *descr, int never_copy)
{
    PyObject *new;
    PyObject *array_meth;

    array_meth = PyArray_LookupSpecial_OnInstance(op, "__array__");
    if (array_meth == NULL) {
        if (PyErr_Occurred()) {
            if (PyErr_ExceptionMatches(PyExc_RecursionError) ||
                PyErr_ExceptionMatches(PyExc_MemoryError)) {
                /* RecursionError and MemoryError are considered fatal */
                return NULL;
            }
            if (deprecated_lookup_error_clearing(
                    Py_TYPE(op), "__array__") < 0) {
                return NULL;
            }
        }
        return Py_NotImplemented;
    }
    if (never_copy) {
        /* Currently, we must always assume that `__array__` returns a copy */
        PyErr_SetString(PyExc_ValueError,
                "Unable to avoid copy while converting from an object "
                "implementing the `__array__` protocol.  NumPy cannot ensure "
                "that no copy will be made.");
        Py_DECREF(array_meth);
        return NULL;
    }

    if (PyType_Check(op) && PyObject_HasAttrString(array_meth, "__get__")) {
        /*
         * If the input is a class `array_meth` may be a property-like object.
         * This cannot be interpreted as an array (called), but is a valid.
         * Trying `array_meth.__call__()` on this should not be useful.
         * (Needed due to the lookup being on the instance rather than type)
         */
        Py_DECREF(array_meth);
        return Py_NotImplemented;
    }
    if (descr == NULL) {
        new = PyObject_CallFunction(array_meth, NULL);
    }
    else {
        new = PyObject_CallFunction(array_meth, "O", descr);
    }
    Py_DECREF(array_meth);
    if (new == NULL) {
        return NULL;
    }
    if (!PyArray_Check(new)) {
        PyErr_SetString(PyExc_ValueError,
                        "object __array__ method not "  \
                        "producing an array");
        Py_DECREF(new);
        return NULL;
    }
    return new;
}


/*NUMPY_API
 */
NPY_NO_EXPORT PyObject *
PyArray_FromArrayAttr(PyObject *op, PyArray_Descr *typecode, PyObject *context)
{
    if (context != NULL) {
        PyErr_SetString(PyExc_RuntimeError, "'context' must be NULL");
        return NULL;
    }

    return PyArray_FromArrayAttr_int(op, typecode, 0);
}


/*NUMPY_API
* new reference -- accepts NULL for mintype
*/
NPY_NO_EXPORT PyArray_Descr *
PyArray_DescrFromObject(PyObject *op, PyArray_Descr *mintype)
{
    PyArray_Descr *dtype;

    dtype = mintype;
    Py_XINCREF(dtype);

    if (PyArray_DTypeFromObject(op, NPY_MAXDIMS, &dtype) < 0) {
        return NULL;
    }

    if (dtype == NULL) {
        return PyArray_DescrFromType(NPY_DEFAULT_TYPE);
    }
    else {
        return dtype;
    }
}

/* These are also old calls (should use PyArray_NewFromDescr) */

/* They all zero-out the memory as previously done */

/* steals reference to descr -- and enforces native byteorder on it.*/

/*NUMPY_API
  Deprecated, use PyArray_NewFromDescr instead.
*/
NPY_NO_EXPORT PyObject *
PyArray_FromDimsAndDataAndDescr(int NPY_UNUSED(nd), int *NPY_UNUSED(d),
                                PyArray_Descr *descr,
                                char *NPY_UNUSED(data))
{
    PyErr_SetString(PyExc_NotImplementedError,
                "PyArray_FromDimsAndDataAndDescr: use PyArray_NewFromDescr.");
    Py_DECREF(descr);
    return NULL;
}

/*NUMPY_API
  Deprecated, use PyArray_SimpleNew instead.
*/
NPY_NO_EXPORT PyObject *
PyArray_FromDims(int NPY_UNUSED(nd), int *NPY_UNUSED(d), int NPY_UNUSED(type))
{
    PyErr_SetString(PyExc_NotImplementedError,
                "PyArray_FromDims: use PyArray_SimpleNew.");
    return NULL;
}

/* end old calls */

/*NUMPY_API
 * This is a quick wrapper around
 * PyArray_FromAny(op, NULL, 0, 0, NPY_ARRAY_ENSUREARRAY, NULL)
 * that special cases Arrays and PyArray_Scalars up front
 * It *steals a reference* to the object
 * It also guarantees that the result is PyArray_Type
 * Because it decrefs op if any conversion needs to take place
 * so it can be used like PyArray_EnsureArray(some_function(...))
 */
NPY_NO_EXPORT PyObject *
PyArray_EnsureArray(PyObject *op)
{
    PyObject *new;

    if ((op == NULL) || (PyArray_CheckExact(op))) {
        new = op;
        Py_XINCREF(new);
    }
    else if (PyArray_Check(op)) {
        new = PyArray_View((PyArrayObject *)op, NULL, &PyArray_Type);
    }
    else if (PyArray_IsScalar(op, Generic)) {
        new = PyArray_FromScalar(op, NULL);
    }
    else {
        new = PyArray_FROM_OF(op, NPY_ARRAY_ENSUREARRAY);
    }
    Py_XDECREF(op);
    return new;
}

/*NUMPY_API*/
NPY_NO_EXPORT PyObject *
PyArray_EnsureAnyArray(PyObject *op)
{
    if (op && PyArray_Check(op)) {
        return op;
    }
    return PyArray_EnsureArray(op);
}

/*
 * Private implementation of PyArray_CopyAnyInto with an additional order
 * parameter.
 */
NPY_NO_EXPORT int
PyArray_CopyAsFlat(PyArrayObject *dst, PyArrayObject *src, NPY_ORDER order)
{
    NpyIter *dst_iter, *src_iter;

    NpyIter_IterNextFunc *dst_iternext, *src_iternext;
    char **dst_dataptr, **src_dataptr;
    npy_intp dst_stride, src_stride;
    npy_intp *dst_countptr, *src_countptr;
    npy_uint32 baseflags;

    npy_intp dst_count, src_count, count;
    npy_intp dst_size, src_size;
    int needs_api;

    NPY_BEGIN_THREADS_DEF;

    if (PyArray_FailUnlessWriteable(dst, "destination array") < 0) {
        return -1;
    }

    /*
     * If the shapes match and a particular order is forced
     * for both, use the more efficient CopyInto
     */
    if (order != NPY_ANYORDER && order != NPY_KEEPORDER &&
            PyArray_NDIM(dst) == PyArray_NDIM(src) &&
            PyArray_CompareLists(PyArray_DIMS(dst), PyArray_DIMS(src),
                                PyArray_NDIM(dst))) {
        return PyArray_CopyInto(dst, src);
    }

    dst_size = PyArray_SIZE(dst);
    src_size = PyArray_SIZE(src);
    if (dst_size != src_size) {
        PyErr_Format(PyExc_ValueError,
                "cannot copy from array of size %" NPY_INTP_FMT " into an array "
                "of size %" NPY_INTP_FMT, src_size, dst_size);
        return -1;
    }

    /* Zero-sized arrays require nothing be done */
    if (dst_size == 0) {
        return 0;
    }

    baseflags = NPY_ITER_EXTERNAL_LOOP |
                NPY_ITER_DONT_NEGATE_STRIDES |
                NPY_ITER_REFS_OK;

    /*
     * This copy is based on matching C-order traversals of src and dst.
     * By using two iterators, we can find maximal sub-chunks that
     * can be processed at once.
     */
    dst_iter = NpyIter_New(dst, NPY_ITER_WRITEONLY | baseflags,
                                order,
                                NPY_NO_CASTING,
                                NULL);
    if (dst_iter == NULL) {
        return -1;
    }
    src_iter = NpyIter_New(src, NPY_ITER_READONLY | baseflags,
                                order,
                                NPY_NO_CASTING,
                                NULL);
    if (src_iter == NULL) {
        NpyIter_Deallocate(dst_iter);
        return -1;
    }

    /* Get all the values needed for the inner loop */
    dst_iternext = NpyIter_GetIterNext(dst_iter, NULL);
    dst_dataptr = NpyIter_GetDataPtrArray(dst_iter);
    /* Since buffering is disabled, we can cache the stride */
    dst_stride = NpyIter_GetInnerStrideArray(dst_iter)[0];
    dst_countptr = NpyIter_GetInnerLoopSizePtr(dst_iter);

    src_iternext = NpyIter_GetIterNext(src_iter, NULL);
    src_dataptr = NpyIter_GetDataPtrArray(src_iter);
    /* Since buffering is disabled, we can cache the stride */
    src_stride = NpyIter_GetInnerStrideArray(src_iter)[0];
    src_countptr = NpyIter_GetInnerLoopSizePtr(src_iter);

    if (dst_iternext == NULL || src_iternext == NULL) {
        NpyIter_Deallocate(dst_iter);
        NpyIter_Deallocate(src_iter);
        return -1;
    }

    needs_api = NpyIter_IterationNeedsAPI(dst_iter) ||
                NpyIter_IterationNeedsAPI(src_iter);

    /*
     * Because buffering is disabled in the iterator, the inner loop
     * strides will be the same throughout the iteration loop.  Thus,
     * we can pass them to this function to take advantage of
     * contiguous strides, etc.
     */
    NPY_cast_info cast_info;
    if (PyArray_GetDTypeTransferFunction(
                    IsUintAligned(src) && IsAligned(src) &&
                    IsUintAligned(dst) && IsAligned(dst),
                    src_stride, dst_stride,
                    PyArray_DESCR(src), PyArray_DESCR(dst),
                    0,
                    &cast_info, &needs_api) != NPY_SUCCEED) {
        NpyIter_Deallocate(dst_iter);
        NpyIter_Deallocate(src_iter);
        return -1;
    }

    if (!needs_api) {
        NPY_BEGIN_THREADS;
    }

    dst_count = *dst_countptr;
    src_count = *src_countptr;
    char *args[2] = {src_dataptr[0], dst_dataptr[0]};
    npy_intp strides[2] = {src_stride, dst_stride};

    int res = 0;
    for(;;) {
        /* Transfer the biggest amount that fits both */
        count = (src_count < dst_count) ? src_count : dst_count;
        if (cast_info.func(&cast_info.context,
                args, &count, strides, cast_info.auxdata) < 0) {
            res = -1;
            break;
        }

        /* If we exhausted the dst block, refresh it */
        if (dst_count == count) {
            res = dst_iternext(dst_iter);
            if (res == 0) {
                break;
            }
            dst_count = *dst_countptr;
            args[1] = dst_dataptr[0];
        }
        else {
            dst_count -= count;
            args[1] += count*dst_stride;
        }

        /* If we exhausted the src block, refresh it */
        if (src_count == count) {
            res = src_iternext(src_iter);
            if (res == 0) {
                break;
            }
            src_count = *src_countptr;
            args[0] = src_dataptr[0];
        }
        else {
            src_count -= count;
            args[0] += count*src_stride;
        }
    }

    NPY_END_THREADS;

    NPY_cast_info_xfree(&cast_info);
    NpyIter_Deallocate(dst_iter);
    NpyIter_Deallocate(src_iter);
    return res;
}

/*NUMPY_API
 * Copy an Array into another array -- memory must not overlap
 * Does not require src and dest to have "broadcastable" shapes
 * (only the same number of elements).
 *
 * TODO: For NumPy 2.0, this could accept an order parameter which
 *       only allows NPY_CORDER and NPY_FORDER.  Could also rename
 *       this to CopyAsFlat to make the name more intuitive.
 *
 * Returns 0 on success, -1 on error.
 */
NPY_NO_EXPORT int
PyArray_CopyAnyInto(PyArrayObject *dst, PyArrayObject *src)
{
    return PyArray_CopyAsFlat(dst, src, NPY_CORDER);
}

/*NUMPY_API
 * Copy an Array into another array.
 * Broadcast to the destination shape if necessary.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
PyArray_CopyInto(PyArrayObject *dst, PyArrayObject *src)
{
    return PyArray_AssignArray(dst, src, NULL, NPY_UNSAFE_CASTING);
}

/*NUMPY_API
 * Move the memory of one array into another, allowing for overlapping data.
 *
 * Returns 0 on success, negative on failure.
 */
NPY_NO_EXPORT int
PyArray_MoveInto(PyArrayObject *dst, PyArrayObject *src)
{
    return PyArray_AssignArray(dst, src, NULL, NPY_UNSAFE_CASTING);
}

/*NUMPY_API
 * PyArray_CheckAxis
 *
 * check that axis is valid
 * convert 0-d arrays to 1-d arrays
 */
NPY_NO_EXPORT PyObject *
PyArray_CheckAxis(PyArrayObject *arr, int *axis, int flags)
{
    PyObject *temp1, *temp2;
    int n = PyArray_NDIM(arr);

    if (*axis == NPY_MAXDIMS || n == 0) {
        if (n != 1) {
            temp1 = PyArray_Ravel(arr,0);
            if (temp1 == NULL) {
                *axis = 0;
                return NULL;
            }
            if (*axis == NPY_MAXDIMS) {
                *axis = PyArray_NDIM((PyArrayObject *)temp1)-1;
            }
        }
        else {
            temp1 = (PyObject *)arr;
            Py_INCREF(temp1);
            *axis = 0;
        }
        if (!flags && *axis == 0) {
            return temp1;
        }
    }
    else {
        temp1 = (PyObject *)arr;
        Py_INCREF(temp1);
    }
    if (flags) {
        temp2 = PyArray_CheckFromAny((PyObject *)temp1, NULL,
                                     0, 0, flags, NULL);
        Py_DECREF(temp1);
        if (temp2 == NULL) {
            return NULL;
        }
    }
    else {
        temp2 = (PyObject *)temp1;
    }
    n = PyArray_NDIM((PyArrayObject *)temp2);
    if (check_and_adjust_axis(axis, n) < 0) {
        Py_DECREF(temp2);
        return NULL;
    }
    return temp2;
}

/*NUMPY_API
 * Zeros
 *
 * steals a reference to type. On failure or when dtype->subarray is
 * true, dtype will be decrefed.
 * accepts NULL type
 */
NPY_NO_EXPORT PyObject *
PyArray_Zeros(int nd, npy_intp const *dims, PyArray_Descr *type, int is_f_order)
{
    PyArrayObject *ret;

    if (!type) {
        type = PyArray_DescrFromType(NPY_DEFAULT_TYPE);
    }

    ret = (PyArrayObject *)PyArray_NewFromDescr_int(
            &PyArray_Type, type,
            nd, dims, NULL, NULL,
            is_f_order, NULL, NULL,
            1, 0);

    if (ret == NULL) {
        return NULL;
    }

    /* handle objects */
    if (PyDataType_REFCHK(PyArray_DESCR(ret))) {
        if (_zerofill(ret) < 0) {
            Py_DECREF(ret);
            return NULL;
        }
    }


    return (PyObject *)ret;

}

/*NUMPY_API
 * Empty
 *
 * accepts NULL type
 * steals a reference to type
 */
NPY_NO_EXPORT PyObject *
PyArray_Empty(int nd, npy_intp const *dims, PyArray_Descr *type, int is_f_order)
{
    PyArrayObject *ret;

    if (!type) type = PyArray_DescrFromType(NPY_DEFAULT_TYPE);

    /*
     * PyArray_NewFromDescr steals a ref,
     * but we need to look at type later.
     * */
    Py_INCREF(type);

    ret = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type,
                                                type, nd, dims,
                                                NULL, NULL,
                                                is_f_order, NULL);
    if (ret != NULL && PyDataType_REFCHK(type)) {
        PyArray_FillObjectArray(ret, Py_None);
        if (PyErr_Occurred()) {
            Py_DECREF(ret);
            Py_DECREF(type);
            return NULL;
        }
    }

    Py_DECREF(type);
    return (PyObject *)ret;
}

/*
 * Like ceil(value), but check for overflow.
 *
 * Return 0 on success, -1 on failure. In case of failure, set a PyExc_Overflow
 * exception
 */
static npy_intp
_arange_safe_ceil_to_intp(double value)
{
    double ivalue;

    ivalue = npy_ceil(value);
    /* condition inverted to handle NaN */
    if (npy_isnan(ivalue)) {
        PyErr_SetString(PyExc_ValueError,
            "arange: cannot compute length");
        return -1;
    }
    if (!((double)NPY_MIN_INTP <= ivalue && ivalue <= (double)NPY_MAX_INTP)) {
        PyErr_SetString(PyExc_OverflowError,
                "arange: overflow while computing length");
        return -1;
    }

    return (npy_intp)ivalue;
}


/*NUMPY_API
  Arange,
*/
NPY_NO_EXPORT PyObject *
PyArray_Arange(double start, double stop, double step, int type_num)
{
    npy_intp length;
    PyArrayObject *range;
    PyArray_ArrFuncs *funcs;
    PyObject *obj;
    int ret;
    double delta, tmp_len;
    NPY_BEGIN_THREADS_DEF;

    delta = stop - start;
    tmp_len = delta/step;

    /* Underflow and divide-by-inf check */
    if (tmp_len == 0.0 && delta != 0.0) {
        if (npy_signbit(tmp_len)) {
            length = 0;
        }
        else {
            length = 1;
        }
    }
    else {
        length = _arange_safe_ceil_to_intp(tmp_len);
        if (error_converting(length)) {
            return NULL;
        }
    }

    if (length <= 0) {
        length = 0;
        return PyArray_New(&PyArray_Type, 1, &length, type_num,
                           NULL, NULL, 0, 0, NULL);
    }
    range = (PyArrayObject *)PyArray_New(&PyArray_Type, 1, &length, type_num,
                        NULL, NULL, 0, 0, NULL);
    if (range == NULL) {
        return NULL;
    }
    funcs = PyArray_DESCR(range)->f;

    /*
     * place start in the buffer and the next value in the second position
     * if length > 2, then call the inner loop, otherwise stop
     */
    obj = PyFloat_FromDouble(start);
    ret = funcs->setitem(obj, PyArray_DATA(range), range);
    Py_DECREF(obj);
    if (ret < 0) {
        goto fail;
    }
    if (length == 1) {
        return (PyObject *)range;
    }
    obj = PyFloat_FromDouble(start + step);
    ret = funcs->setitem(obj, PyArray_BYTES(range)+PyArray_ITEMSIZE(range),
                         range);
    Py_DECREF(obj);
    if (ret < 0) {
        goto fail;
    }
    if (length == 2) {
        return (PyObject *)range;
    }
    if (!funcs->fill) {
        PyErr_SetString(PyExc_ValueError,
                "no fill-function for data-type.");
        Py_DECREF(range);
        return NULL;
    }
    NPY_BEGIN_THREADS_DESCR(PyArray_DESCR(range));
    funcs->fill(PyArray_DATA(range), length, range);
    NPY_END_THREADS;
    if (PyErr_Occurred()) {
        goto fail;
    }
    return (PyObject *)range;

 fail:
    Py_DECREF(range);
    return NULL;
}

/*
 * the formula is len = (intp) ceil((stop - start) / step);
 */
static npy_intp
_calc_length(PyObject *start, PyObject *stop, PyObject *step, PyObject **next, int cmplx)
{
    npy_intp len, tmp;
    PyObject *zero, *val;
    int next_is_nonzero, val_is_zero;
    double value;

    *next = PyNumber_Subtract(stop, start);
    if (!(*next)) {
        if (PyTuple_Check(stop)) {
            PyErr_Clear();
            PyErr_SetString(PyExc_TypeError,
                            "arange: scalar arguments expected "\
                            "instead of a tuple.");
        }
        return -1;
    }

    zero = PyLong_FromLong(0);
    if (!zero) {
        Py_DECREF(*next);
        *next = NULL;
        return -1;
    }

    next_is_nonzero = PyObject_RichCompareBool(*next, zero, Py_NE);
    if (next_is_nonzero == -1) {
        Py_DECREF(zero);
        Py_DECREF(*next);
        *next = NULL;
        return -1;
    }
    val = PyNumber_TrueDivide(*next, step);
    Py_DECREF(*next);
    *next = NULL;

    if (!val) {
        Py_DECREF(zero);
        return -1;
    }

    val_is_zero = PyObject_RichCompareBool(val, zero, Py_EQ);
    Py_DECREF(zero);
    if (val_is_zero == -1) {
        Py_DECREF(val);
        return -1;
    }

    if (cmplx && PyComplex_Check(val)) {
        value = PyComplex_RealAsDouble(val);
        if (error_converting(value)) {
            Py_DECREF(val);
            return -1;
        }
        len = _arange_safe_ceil_to_intp(value);
        if (error_converting(len)) {
            Py_DECREF(val);
            return -1;
        }
        value = PyComplex_ImagAsDouble(val);
        Py_DECREF(val);
        if (error_converting(value)) {
            return -1;
        }
        tmp = _arange_safe_ceil_to_intp(value);
        if (error_converting(tmp)) {
            return -1;
        }
        len = PyArray_MIN(len, tmp);
    }
    else {
        value = PyFloat_AsDouble(val);
        Py_DECREF(val);
        if (error_converting(value)) {
            return -1;
        }

        /* Underflow and divide-by-inf check */
        if (val_is_zero && next_is_nonzero) {
            if (npy_signbit(value)) {
                len = 0;
            }
            else {
                len = 1;
            }
        }
        else {
            len = _arange_safe_ceil_to_intp(value);
            if (error_converting(len)) {
                return -1;
            }
        }
    }

    if (len > 0) {
        *next = PyNumber_Add(start, step);
        if (!*next) {
            return -1;
        }
    }
    return len;
}

/*NUMPY_API
 *
 * ArangeObj,
 *
 * this doesn't change the references
 */
NPY_NO_EXPORT PyObject *
PyArray_ArangeObj(PyObject *start, PyObject *stop, PyObject *step, PyArray_Descr *dtype)
{
    PyArrayObject *range;
    PyArray_ArrFuncs *funcs;
    PyObject *next, *err;
    npy_intp length;
    PyArray_Descr *native = NULL;
    int swap;
    NPY_BEGIN_THREADS_DEF;

    /* Datetime arange is handled specially */
    if ((dtype != NULL && (dtype->type_num == NPY_DATETIME ||
                           dtype->type_num == NPY_TIMEDELTA)) ||
            (dtype == NULL && (is_any_numpy_datetime_or_timedelta(start) ||
                              is_any_numpy_datetime_or_timedelta(stop) ||
                              is_any_numpy_datetime_or_timedelta(step)))) {
        return (PyObject *)datetime_arange(start, stop, step, dtype);
    }

    if (!dtype) {
        PyArray_Descr *deftype;
        PyArray_Descr *newtype;

        /* intentionally made to be at least NPY_LONG */
        deftype = PyArray_DescrFromType(NPY_LONG);
        newtype = PyArray_DescrFromObject(start, deftype);
        Py_DECREF(deftype);
        if (newtype == NULL) {
            return NULL;
        }
        deftype = newtype;
        if (stop && stop != Py_None) {
            newtype = PyArray_DescrFromObject(stop, deftype);
            Py_DECREF(deftype);
            if (newtype == NULL) {
                return NULL;
            }
            deftype = newtype;
        }
        if (step && step != Py_None) {
            newtype = PyArray_DescrFromObject(step, deftype);
            Py_DECREF(deftype);
            if (newtype == NULL) {
                return NULL;
            }
            deftype = newtype;
        }
        dtype = deftype;
    }
    else {
        Py_INCREF(dtype);
    }
    if (!step || step == Py_None) {
        step = PyLong_FromLong(1);
    }
    else {
        Py_XINCREF(step);
    }
    if (!stop || stop == Py_None) {
        stop = start;
        start = PyLong_FromLong(0);
    }
    else {
        Py_INCREF(start);
    }
    /* calculate the length and next = start + step*/
    length = _calc_length(start, stop, step, &next,
                          PyTypeNum_ISCOMPLEX(dtype->type_num));
    err = PyErr_Occurred();
    if (err) {
        Py_DECREF(dtype);
        if (err && PyErr_GivenExceptionMatches(err, PyExc_OverflowError)) {
            PyErr_SetString(PyExc_ValueError, "Maximum allowed size exceeded");
        }
        goto fail;
    }
    if (length <= 0) {
        length = 0;
        range = (PyArrayObject *)PyArray_SimpleNewFromDescr(1, &length, dtype);
        Py_DECREF(step);
        Py_DECREF(start);
        return (PyObject *)range;
    }

    /*
     * If dtype is not in native byte-order then get native-byte
     * order version.  And then swap on the way out.
     */
    if (!PyArray_ISNBO(dtype->byteorder)) {
        native = PyArray_DescrNewByteorder(dtype, NPY_NATBYTE);
        swap = 1;
    }
    else {
        native = dtype;
        swap = 0;
    }

    range = (PyArrayObject *)PyArray_SimpleNewFromDescr(1, &length, native);
    if (range == NULL) {
        goto fail;
    }

    /*
     * place start in the buffer and the next value in the second position
     * if length > 2, then call the inner loop, otherwise stop
     */
    funcs = PyArray_DESCR(range)->f;
    if (funcs->setitem(start, PyArray_DATA(range), range) < 0) {
        goto fail;
    }
    if (length == 1) {
        goto finish;
    }
    if (funcs->setitem(next, PyArray_BYTES(range)+PyArray_ITEMSIZE(range),
                       range) < 0) {
        goto fail;
    }
    if (length == 2) {
        goto finish;
    }
    if (!funcs->fill) {
        PyErr_SetString(PyExc_ValueError, "no fill-function for data-type.");
        Py_DECREF(range);
        goto fail;
    }
    NPY_BEGIN_THREADS_DESCR(PyArray_DESCR(range));
    funcs->fill(PyArray_DATA(range), length, range);
    NPY_END_THREADS;
    if (PyErr_Occurred()) {
        goto fail;
    }
 finish:
    /* TODO: This swapping could be handled on the fly by the nditer */
    if (swap) {
        PyObject *new;
        new = PyArray_Byteswap(range, 1);
        Py_DECREF(new);
        Py_DECREF(PyArray_DESCR(range));
        /* steals the reference */
        ((PyArrayObject_fields *)range)->descr = dtype;
    }
    Py_DECREF(start);
    Py_DECREF(step);
    Py_DECREF(next);
    return (PyObject *)range;

 fail:
    Py_DECREF(start);
    Py_DECREF(step);
    Py_XDECREF(next);
    return NULL;
}

/* This array creation function does not steal the reference to dtype. */
static PyArrayObject *
array_fromfile_binary(FILE *fp, PyArray_Descr *dtype, npy_intp num, size_t *nread)
{
    PyArrayObject *r;
    npy_off_t start, numbytes;
    int elsize;

    if (num < 0) {
        int fail = 0;
        start = npy_ftell(fp);
        if (start < 0) {
            fail = 1;
        }
        if (npy_fseek(fp, 0, SEEK_END) < 0) {
            fail = 1;
        }
        numbytes = npy_ftell(fp);
        if (numbytes < 0) {
            fail = 1;
        }
        numbytes -= start;
        if (npy_fseek(fp, start, SEEK_SET) < 0) {
            fail = 1;
        }
        if (fail) {
            PyErr_SetString(PyExc_OSError,
                            "could not seek in file");
            return NULL;
        }
        num = numbytes / dtype->elsize;
    }

    /*
     * Array creation may move sub-array dimensions from the dtype to array
     * dimensions, so we need to use the original element size when reading.
     */
    elsize = dtype->elsize;

    Py_INCREF(dtype);  /* do not steal the original dtype. */
    r = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type, dtype, 1, &num,
                                              NULL, NULL, 0, NULL);
    if (r == NULL) {
        return NULL;
    }

    NPY_BEGIN_ALLOW_THREADS;
    *nread = fread(PyArray_DATA(r), elsize, num, fp);
    NPY_END_ALLOW_THREADS;
    return r;
}

/*
 * Create an array by reading from the given stream, using the passed
 * next_element and skip_separator functions.
 * Does not steal the reference to dtype.
 */
#define FROM_BUFFER_SIZE 4096
static PyArrayObject *
array_from_text(PyArray_Descr *dtype, npy_intp num, char const *sep, size_t *nread,
                void *stream, next_element next, skip_separator skip_sep,
                void *stream_data)
{
    PyArrayObject *r;
    npy_intp i;
    char *dptr, *clean_sep, *tmp;
    int err = 0;
    int stop_reading_flag = 0;  /* -1 means end reached; -2 a parsing error */
    npy_intp thisbuf = 0;
    npy_intp size;
    npy_intp bytes, totalbytes;

    size = (num >= 0) ? num : FROM_BUFFER_SIZE;

    /*
     * Array creation may move sub-array dimensions from the dtype to array
     * dimensions, so we need to use the original dtype when reading.
     */
    Py_INCREF(dtype);

    r = (PyArrayObject *)
        PyArray_NewFromDescr(&PyArray_Type, dtype, 1, &size,
                             NULL, NULL, 0, NULL);
    if (r == NULL) {
        return NULL;
    }

    clean_sep = swab_separator(sep);
    if (clean_sep == NULL) {
        err = 1;
        goto fail;
    }

    NPY_BEGIN_ALLOW_THREADS;
    totalbytes = bytes = size * dtype->elsize;
    dptr = PyArray_DATA(r);
    for (i = 0; num < 0 || i < num; i++) {
        stop_reading_flag = next(&stream, dptr, dtype, stream_data);
        if (stop_reading_flag < 0) {
            break;
        }
        *nread += 1;
        thisbuf += 1;
        dptr += dtype->elsize;
        if (num < 0 && thisbuf == size) {
            totalbytes += bytes;
            /* The handler is always valid */
            tmp = PyDataMem_UserRENEW(PyArray_DATA(r), totalbytes,
                                  PyArray_HANDLER(r));
            if (tmp == NULL) {
                err = 1;
                break;
            }
            ((PyArrayObject_fields *)r)->data = tmp;
            dptr = tmp + (totalbytes - bytes);
            thisbuf = 0;
        }
        stop_reading_flag = skip_sep(&stream, clean_sep, stream_data);
        if (stop_reading_flag < 0) {
            if (num == i + 1) {
                /* if we read as much as requested sep is optional */
                stop_reading_flag = -1;
            }
            break;
        }
    }
    if (num < 0) {
        const size_t nsize = PyArray_MAX(*nread,1)*dtype->elsize;

        if (nsize != 0) {
            /* The handler is always valid */
            tmp = PyDataMem_UserRENEW(PyArray_DATA(r), nsize,
                                  PyArray_HANDLER(r));
            if (tmp == NULL) {
                err = 1;
            }
            else {
                PyArray_DIMS(r)[0] = *nread;
                ((PyArrayObject_fields *)r)->data = tmp;
            }
        }
    }
    NPY_END_ALLOW_THREADS;

    free(clean_sep);

    if (stop_reading_flag == -2) {
        if (PyErr_Occurred()) {
            /* If an error is already set (unlikely), do not create new one */
            Py_DECREF(r);
            return NULL;
        }
        /* 2019-09-12, NumPy 1.18 */
        if (DEPRECATE(
                "string or file could not be read to its end due to unmatched "
                "data; this will raise a ValueError in the future.") < 0) {
            goto fail;
        }
    }

fail:
    if (err == 1) {
        PyErr_NoMemory();
    }
    if (PyErr_Occurred()) {
        Py_DECREF(r);
        return NULL;
    }
    return r;
}
#undef FROM_BUFFER_SIZE

/*NUMPY_API
 *
 * Given a ``FILE *`` pointer ``fp``, and a ``PyArray_Descr``, return an
 * array corresponding to the data encoded in that file.
 *
 * The reference to `dtype` is stolen (it is possible that the passed in
 * dtype is not held on to).
 *
 * The number of elements to read is given as ``num``; if it is < 0, then
 * then as many as possible are read.
 *
 * If ``sep`` is NULL or empty, then binary data is assumed, else
 * text data, with ``sep`` as the separator between elements. Whitespace in
 * the separator matches any length of whitespace in the text, and a match
 * for whitespace around the separator is added.
 *
 * For memory-mapped files, use the buffer interface. No more data than
 * necessary is read by this routine.
 */
NPY_NO_EXPORT PyObject *
PyArray_FromFile(FILE *fp, PyArray_Descr *dtype, npy_intp num, char *sep)
{
    PyArrayObject *ret;
    size_t nread = 0;

    if (dtype == NULL) {
        return NULL;
    }

    if (PyDataType_REFCHK(dtype)) {
        PyErr_SetString(PyExc_ValueError,
                "Cannot read into object array");
        Py_DECREF(dtype);
        return NULL;
    }
    if (dtype->elsize == 0) {
        /* Nothing to read, just create an empty array of the requested type */
        return PyArray_NewFromDescr_int(
                &PyArray_Type, dtype,
                1, &num, NULL, NULL,
                0, NULL, NULL,
                0, 1);
    }
    if ((sep == NULL) || (strlen(sep) == 0)) {
        ret = array_fromfile_binary(fp, dtype, num, &nread);
    }
    else {
        if (dtype->f->scanfunc == NULL) {
            PyErr_SetString(PyExc_ValueError,
                    "Unable to read character files of that array type");
            Py_DECREF(dtype);
            return NULL;
        }
        ret = array_from_text(dtype, num, sep, &nread, fp,
                (next_element) fromfile_next_element,
                (skip_separator) fromfile_skip_separator, NULL);
    }
    if (ret == NULL) {
        Py_DECREF(dtype);
        return NULL;
    }
    if (((npy_intp) nread) < num) {
        /*
         * Realloc memory for smaller number of elements, use original dtype
         * which may have include a subarray (and is used for `nread`).
         */
        const size_t nsize = PyArray_MAX(nread,1) * dtype->elsize;
        char *tmp;

        /* The handler is always valid */
        if((tmp = PyDataMem_UserRENEW(PyArray_DATA(ret), nsize,
                                     PyArray_HANDLER(ret))) == NULL) {
            Py_DECREF(dtype);
            Py_DECREF(ret);
            return PyErr_NoMemory();
        }
        ((PyArrayObject_fields *)ret)->data = tmp;
        PyArray_DIMS(ret)[0] = nread;
    }
    Py_DECREF(dtype);
    return (PyObject *)ret;
}

/*NUMPY_API*/
NPY_NO_EXPORT PyObject *
PyArray_FromBuffer(PyObject *buf, PyArray_Descr *type,
                   npy_intp count, npy_intp offset)
{
    PyArrayObject *ret;
    char *data;
    Py_buffer view;
    Py_ssize_t ts;
    npy_intp s, n;
    int itemsize;
    int writeable = 1;

    if (type == NULL) {
        return NULL;
    }

    if (PyDataType_REFCHK(type)) {
        PyErr_SetString(PyExc_ValueError,
                        "cannot create an OBJECT array from memory"\
                        " buffer");
        Py_DECREF(type);
        return NULL;
    }
    if (PyDataType_ISUNSIZED(type)) {
        PyErr_SetString(PyExc_ValueError,
                        "itemsize cannot be zero in type");
        Py_DECREF(type);
        return NULL;
    }

    if (PyObject_GetBuffer(buf, &view, PyBUF_WRITABLE|PyBUF_SIMPLE) < 0) {
        writeable = 0;
        PyErr_Clear();
        if (PyObject_GetBuffer(buf, &view, PyBUF_SIMPLE) < 0) {
            Py_DECREF(type);
            return NULL;
        }
    }
    data = (char *)view.buf;
    ts = view.len;
    /*
     * In Python 3 both of the deprecated functions PyObject_AsWriteBuffer and
     * PyObject_AsReadBuffer that this code replaces release the buffer. It is
     * up to the object that supplies the buffer to guarantee that the buffer
     * sticks around after the release.
     */
    PyBuffer_Release(&view);

    if ((offset < 0) || (offset > ts)) {
        PyErr_Format(PyExc_ValueError,
                     "offset must be non-negative and no greater than buffer "\
                     "length (%" NPY_INTP_FMT ")", (npy_intp)ts);
        Py_DECREF(type);
        return NULL;
    }

    data += offset;
    s = (npy_intp)ts - offset;
    n = (npy_intp)count;
    itemsize = type->elsize;
    if (n < 0) {
        if (itemsize == 0) {
            PyErr_SetString(PyExc_ValueError,
                            "cannot determine count if itemsize is 0");
            Py_DECREF(type);
            return NULL;
        }
        if (s % itemsize != 0) {
            PyErr_SetString(PyExc_ValueError,
                            "buffer size must be a multiple"\
                            " of element size");
            Py_DECREF(type);
            return NULL;
        }
        n = s/itemsize;
    }
    else {
        if (s < n*itemsize) {
            PyErr_SetString(PyExc_ValueError,
                            "buffer is smaller than requested"\
                            " size");
            Py_DECREF(type);
            return NULL;
        }
    }

    ret = (PyArrayObject *)PyArray_NewFromDescrAndBase(
            &PyArray_Type, type,
            1, &n, NULL, data,
            NPY_ARRAY_DEFAULT, NULL, buf);
    if (ret == NULL) {
        return NULL;
    }

    if (!writeable) {
        PyArray_CLEARFLAGS(ret, NPY_ARRAY_WRITEABLE);
    }
    return (PyObject *)ret;
}

/*NUMPY_API
 *
 * Given a pointer to a string ``data``, a string length ``slen``, and
 * a ``PyArray_Descr``, return an array corresponding to the data
 * encoded in that string.
 *
 * If the dtype is NULL, the default array type is used (double).
 * If non-null, the reference is stolen.
 *
 * If ``slen`` is < 0, then the end of string is used for text data.
 * It is an error for ``slen`` to be < 0 for binary data (since embedded NULLs
 * would be the norm).
 *
 * The number of elements to read is given as ``num``; if it is < 0, then
 * then as many as possible are read.
 *
 * If ``sep`` is NULL or empty, then binary data is assumed, else
 * text data, with ``sep`` as the separator between elements. Whitespace in
 * the separator matches any length of whitespace in the text, and a match
 * for whitespace around the separator is added.
 */
NPY_NO_EXPORT PyObject *
PyArray_FromString(char *data, npy_intp slen, PyArray_Descr *dtype,
                   npy_intp num, char *sep)
{
    int itemsize;
    PyArrayObject *ret;
    npy_bool binary;

    if (dtype == NULL) {
        dtype=PyArray_DescrFromType(NPY_DEFAULT_TYPE);
        if (dtype == NULL) {
            return NULL;
        }
    }
    if (PyDataType_FLAGCHK(dtype, NPY_ITEM_IS_POINTER) ||
                    PyDataType_REFCHK(dtype)) {
        PyErr_SetString(PyExc_ValueError,
                        "Cannot create an object array from"    \
                        " a string");
        Py_DECREF(dtype);
        return NULL;
    }
    itemsize = dtype->elsize;
    if (itemsize == 0) {
        PyErr_SetString(PyExc_ValueError, "zero-valued itemsize");
        Py_DECREF(dtype);
        return NULL;
    }

    binary = ((sep == NULL) || (strlen(sep) == 0));
    if (binary) {
        if (num < 0 ) {
            if (slen % itemsize != 0) {
                PyErr_SetString(PyExc_ValueError,
                                "string size must be a "\
                                "multiple of element size");
                Py_DECREF(dtype);
                return NULL;
            }
            num = slen/itemsize;
        }
        else {
            if (slen < num*itemsize) {
                PyErr_SetString(PyExc_ValueError,
                                "string is smaller than " \
                                "requested size");
                Py_DECREF(dtype);
                return NULL;
            }
        }
        /*
         * NewFromDescr may replace dtype to absorb subarray shape
         * into the array, so get size beforehand.
         */
        npy_intp size_to_copy = num*dtype->elsize;
        ret = (PyArrayObject *)
            PyArray_NewFromDescr(&PyArray_Type, dtype,
                                 1, &num, NULL, NULL,
                                 0, NULL);
        if (ret == NULL) {
            return NULL;
        }
        memcpy(PyArray_DATA(ret), data, size_to_copy);
    }
    else {
        /* read from character-based string */
        size_t nread = 0;
        char *end;

        if (dtype->f->fromstr == NULL) {
            PyErr_SetString(PyExc_ValueError,
                            "don't know how to read "       \
                            "character strings with that "  \
                            "array type");
            Py_DECREF(dtype);
            return NULL;
        }
        if (slen < 0) {
            end = NULL;
        }
        else {
            end = data + slen;
        }
        ret = array_from_text(dtype, num, sep, &nread,
                              data,
                              (next_element) fromstr_next_element,
                              (skip_separator) fromstr_skip_separator,
                              end);
        Py_DECREF(dtype);
    }
    return (PyObject *)ret;
}

/*NUMPY_API
 *
 * steals a reference to dtype (which cannot be NULL)
 */
NPY_NO_EXPORT PyObject *
PyArray_FromIter(PyObject *obj, PyArray_Descr *dtype, npy_intp count)
{
    PyObject *value;
    PyObject *iter = NULL;
    PyArrayObject *ret = NULL;
    npy_intp i, elsize, elcount;
    char *item, *new_data;

    if (dtype == NULL) {
        return NULL;
    }

    iter = PyObject_GetIter(obj);
    if (iter == NULL) {
        goto done;
    }

    if (PyDataType_ISUNSIZED(dtype)) {
        PyErr_SetString(PyExc_ValueError,
                "Must specify length when using variable-size data-type.");
        goto done;
    }
    if (count < 0) {
        elcount = PyObject_LengthHint(obj, 0);
        if (elcount < 0) {
            goto done;
        }
    }
    else {
        elcount = count;
    }

    elsize = dtype->elsize;

    /*
     * We would need to alter the memory RENEW code to decrement any
     * reference counts before throwing away any memory.
     */
    if (PyDataType_REFCHK(dtype)) {
        PyErr_SetString(PyExc_ValueError,
                "cannot create object arrays from iterator");
        goto done;
    }

    ret = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type, dtype, 1,
                                                &elcount, NULL,NULL, 0, NULL);
    dtype = NULL;
    if (ret == NULL) {
        goto done;
    }
    for (i = 0; (i < count || count == -1) &&
             (value = PyIter_Next(iter)); i++) {
        if (i >= elcount && elsize != 0) {
            npy_intp nbytes;
            /*
              Grow PyArray_DATA(ret):
              this is similar for the strategy for PyListObject, but we use
              50% overallocation => 0, 4, 8, 14, 23, 36, 56, 86 ...
            */
            elcount = (i >> 1) + (i < 4 ? 4 : 2) + i;
            if (!npy_mul_with_overflow_intp(&nbytes, elcount, elsize)) {
                /* The handler is always valid */
                new_data = PyDataMem_UserRENEW(PyArray_DATA(ret), nbytes,
                                  PyArray_HANDLER(ret));
            }
            else {
                new_data = NULL;
            }
            if (new_data == NULL) {
                PyErr_SetString(PyExc_MemoryError,
                        "cannot allocate array memory");
                Py_DECREF(value);
                goto done;
            }
            ((PyArrayObject_fields *)ret)->data = new_data;
        }
        PyArray_DIMS(ret)[0] = i + 1;

        if (((item = index2ptr(ret, i)) == NULL) ||
                PyArray_SETITEM(ret, item, value) == -1) {
            Py_DECREF(value);
            goto done;
        }
        Py_DECREF(value);
    }


    if (PyErr_Occurred()) {
        goto done;
    }
    if (i < count) {
        PyErr_SetString(PyExc_ValueError,
                "iterator too short");
        goto done;
    }

    /*
     * Realloc the data so that don't keep extra memory tied up
     * (assuming realloc is reasonably good about reusing space...)
     */
    if (i == 0 || elsize == 0) {
        /* The size cannot be zero for realloc. */
        goto done;
    }
    /* The handler is always valid */
    new_data = PyDataMem_UserRENEW(PyArray_DATA(ret), i * elsize,
                                   PyArray_HANDLER(ret));
    if (new_data == NULL) {
        PyErr_SetString(PyExc_MemoryError,
                "cannot allocate array memory");
        goto done;
    }
    ((PyArrayObject_fields *)ret)->data = new_data;

 done:
    Py_XDECREF(iter);
    Py_XDECREF(dtype);
    if (PyErr_Occurred()) {
        Py_XDECREF(ret);
        return NULL;
    }
    return (PyObject *)ret;
}

/*
 * This is the main array creation routine.
 *
 * Flags argument has multiple related meanings
 * depending on data and strides:
 *
 * If data is given, then flags is flags associated with data.
 * If strides is not given, then a contiguous strides array will be created
 * and the NPY_ARRAY_C_CONTIGUOUS bit will be set.  If the flags argument
 * has the NPY_ARRAY_F_CONTIGUOUS bit set, then a FORTRAN-style strides array will be
 * created (and of course the NPY_ARRAY_F_CONTIGUOUS flag bit will be set).
 *
 * If data is not given but created here, then flags will be NPY_ARRAY_DEFAULT
 * and a non-zero flags argument can be used to indicate a FORTRAN style
 * array is desired.
 *
 * Dimensions and itemsize must have been checked for validity.
 */

NPY_NO_EXPORT void
_array_fill_strides(npy_intp *strides, npy_intp const *dims, int nd, size_t itemsize,
                    int inflag, int *objflags)
{
    int i;
#if NPY_RELAXED_STRIDES_CHECKING
    npy_bool not_cf_contig = 0;
    npy_bool nod = 0; /* A dim != 1 was found */

    /* Check if new array is both F- and C-contiguous */
    for (i = 0; i < nd; i++) {
        if (dims[i] != 1) {
            if (nod) {
                not_cf_contig = 1;
                break;
            }
            nod = 1;
        }
    }
#endif /* NPY_RELAXED_STRIDES_CHECKING */

    /* Only make Fortran strides if not contiguous as well */
    if ((inflag & (NPY_ARRAY_F_CONTIGUOUS|NPY_ARRAY_C_CONTIGUOUS)) ==
                                            NPY_ARRAY_F_CONTIGUOUS) {
        for (i = 0; i < nd; i++) {
            strides[i] = itemsize;
            if (dims[i]) {
                itemsize *= dims[i];
            }
#if NPY_RELAXED_STRIDES_CHECKING
            else {
                not_cf_contig = 0;
            }
#if NPY_RELAXED_STRIDES_DEBUG
            /* For testing purpose only */
            if (dims[i] == 1) {
                strides[i] = NPY_MAX_INTP;
            }
#endif /* NPY_RELAXED_STRIDES_DEBUG */
#endif /* NPY_RELAXED_STRIDES_CHECKING */
        }
#if NPY_RELAXED_STRIDES_CHECKING
        if (not_cf_contig) {
#else /* not NPY_RELAXED_STRIDES_CHECKING */
        if ((nd > 1) && ((strides[0] != strides[nd-1]) || (dims[nd-1] > 1))) {
#endif /* not NPY_RELAXED_STRIDES_CHECKING */
            *objflags = ((*objflags)|NPY_ARRAY_F_CONTIGUOUS) &
                                            ~NPY_ARRAY_C_CONTIGUOUS;
        }
        else {
            *objflags |= (NPY_ARRAY_F_CONTIGUOUS|NPY_ARRAY_C_CONTIGUOUS);
        }
    }
    else {
        for (i = nd - 1; i >= 0; i--) {
            strides[i] = itemsize;
            if (dims[i]) {
                itemsize *= dims[i];
            }
#if NPY_RELAXED_STRIDES_CHECKING
            else {
                not_cf_contig = 0;
            }
#if NPY_RELAXED_STRIDES_DEBUG
            /* For testing purpose only */
            if (dims[i] == 1) {
                strides[i] = NPY_MAX_INTP;
            }
#endif /* NPY_RELAXED_STRIDES_DEBUG */
#endif /* NPY_RELAXED_STRIDES_CHECKING */
        }
#if NPY_RELAXED_STRIDES_CHECKING
        if (not_cf_contig) {
#else /* not NPY_RELAXED_STRIDES_CHECKING */
        if ((nd > 1) && ((strides[0] != strides[nd-1]) || (dims[0] > 1))) {
#endif /* not NPY_RELAXED_STRIDES_CHECKING */
            *objflags = ((*objflags)|NPY_ARRAY_C_CONTIGUOUS) &
                                            ~NPY_ARRAY_F_CONTIGUOUS;
        }
        else {
            *objflags |= (NPY_ARRAY_C_CONTIGUOUS|NPY_ARRAY_F_CONTIGUOUS);
        }
    }
    return;
}

/*
 * Calls arr_of_subclass.__array_wrap__(towrap), in order to make 'towrap'
 * have the same ndarray subclass as 'arr_of_subclass'.
 */
NPY_NO_EXPORT PyArrayObject *
PyArray_SubclassWrap(PyArrayObject *arr_of_subclass, PyArrayObject *towrap)
{
    PyObject *wrapped = PyObject_CallMethodObjArgs((PyObject *)arr_of_subclass,
            npy_ma_str_array_wrap, (PyObject *)towrap, NULL);
    if (wrapped == NULL) {
        return NULL;
    }
    if (!PyArray_Check(wrapped)) {
        PyErr_SetString(PyExc_RuntimeError,
                "ndarray subclass __array_wrap__ method returned an "
                "object which was not an instance of an ndarray subclass");
        Py_DECREF(wrapped);
        return NULL;
    }

    return (PyArrayObject *)wrapped;
}
