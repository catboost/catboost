/*
  Provide multidimensional arrays as a basic object type in python.

  Based on Original Numeric implementation
  Copyright (c) 1995, 1996, 1997 Jim Hugunin, hugunin@mit.edu

  with contributions from many Numeric Python developers 1995-2004

  Heavily modified in 2005 with inspiration from Numarray

  by

  Travis Oliphant,  oliphant@ee.byu.edu
  Brigham Young University


maintainer email:  oliphant.travis@ieee.org

  Numarray design (which provided guidance) by
  Space Science Telescope Institute
  (J. Todd Miller, Perry Greenfield, Rick White)
*/
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"

#include "npy_config.h"

#include "common.h"

#include "npy_pycompat.h"

#include "usertypes.h"
#include "dtypemeta.h"
#include "scalartypes.h"
#include "array_method.h"
#include "convert_datatype.h"
#include "legacy_dtype_implementation.h"


NPY_NO_EXPORT PyArray_Descr **userdescrs=NULL;

static int
_append_new(int **p_types, int insert)
{
    int n = 0;
    int *newtypes;
    int *types = *p_types;

    while (types[n] != NPY_NOTYPE) {
        n++;
    }
    newtypes = (int *)realloc(types, (n + 2)*sizeof(int));
    if (newtypes == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    newtypes[n] = insert;
    newtypes[n + 1] = NPY_NOTYPE;

    /* Replace the passed-in pointer */
    *p_types = newtypes;
    return 0;
}

static npy_bool
_default_nonzero(void *ip, void *arr)
{
    int elsize = PyArray_ITEMSIZE(arr);
    char *ptr = ip;
    while (elsize--) {
        if (*ptr++ != 0) {
            return NPY_TRUE;
        }
    }
    return NPY_FALSE;
}

static void
_default_copyswapn(void *dst, npy_intp dstride, void *src,
                   npy_intp sstride, npy_intp n, int swap, void *arr)
{
    npy_intp i;
    PyArray_CopySwapFunc *copyswap;
    char *dstptr = dst;
    char *srcptr = src;

    copyswap = PyArray_DESCR(arr)->f->copyswap;

    for (i = 0; i < n; i++) {
        copyswap(dstptr, srcptr, swap, arr);
        dstptr += dstride;
        srcptr += sstride;
    }
}

/*NUMPY_API
  Initialize arrfuncs to NULL
*/
NPY_NO_EXPORT void
PyArray_InitArrFuncs(PyArray_ArrFuncs *f)
{
    int i;

    for(i = 0; i < NPY_NTYPES_ABI_COMPATIBLE; i++) {
        f->cast[i] = NULL;
    }
    f->getitem = NULL;
    f->setitem = NULL;
    f->copyswapn = NULL;
    f->copyswap = NULL;
    f->compare = NULL;
    f->argmax = NULL;
    f->argmin = NULL;
    f->dotfunc = NULL;
    f->scanfunc = NULL;
    f->fromstr = NULL;
    f->nonzero = NULL;
    f->fill = NULL;
    f->fillwithscalar = NULL;
    for(i = 0; i < NPY_NSORTS; i++) {
        f->sort[i] = NULL;
        f->argsort[i] = NULL;
    }
    f->castdict = NULL;
    f->scalarkind = NULL;
    f->cancastscalarkindto = NULL;
    f->cancastto = NULL;
    f->fastclip = NULL;
    f->fastputmask = NULL;
    f->fasttake = NULL;
}


static int
test_deprecated_arrfuncs_members(PyArray_ArrFuncs *f) {
    /* NumPy 1.19, 2020-01-15 */
    if (f->fastputmask != NULL) {
        if (DEPRECATE(
                "The ->f->fastputmask member of custom dtypes is ignored; "
                "setting it may be an error in the future.\n"
                "The custom dtype you are using must be revised, but "
                "results will not be affected.") < 0) {
            return -1;
        }
    }
    /* NumPy 1.19, 2020-01-15 */
    if (f->fasttake != NULL) {
        if (DEPRECATE(
                "The ->f->fastputmask member of custom dtypes is ignored; "
                "setting it may be an error in the future.\n"
                "The custom dtype you are using must be revised, but "
                "results will not be affected.") < 0) {
            return -1;
        }
    }
    /* NumPy 1.19, 2020-01-15 */
    if (f->fastclip != NULL) {
        /* fastclip was already deprecated at execution time in 1.17. */
        if (DEPRECATE(
                "The ->f->fastclip member of custom dtypes is deprecated; "
                "setting it will be an error in the future.\n"
                "The custom dtype you are using must be changed to use "
                "PyUFunc_RegisterLoopForDescr to attach a custom loop to "
                "np.core.umath.clip, np.minimum, and np.maximum") < 0) {
            return -1;
        }
    }
    return 0;
}

/*
  returns typenum to associate with this type >=NPY_USERDEF.
  needs the userdecrs table and PyArray_NUMUSER variables
  defined in arraytypes.inc
*/
/*NUMPY_API
  Register Data type
  Does not change the reference count of descr
*/
NPY_NO_EXPORT int
PyArray_RegisterDataType(PyArray_Descr *descr)
{
    PyArray_Descr *descr2;
    int typenum;
    int i;
    PyArray_ArrFuncs *f;

    /* See if this type is already registered */
    for (i = 0; i < NPY_NUMUSERTYPES; i++) {
        descr2 = userdescrs[i];
        if (descr2 == descr) {
            return descr->type_num;
        }
    }
    typenum = NPY_USERDEF + NPY_NUMUSERTYPES;
    descr->type_num = -1;
    if (PyDataType_ISUNSIZED(descr)) {
        PyErr_SetString(PyExc_ValueError, "cannot register a" \
                        "flexible data-type");
        return -1;
    }
    f = descr->f;
    if (f->nonzero == NULL) {
        f->nonzero = _default_nonzero;
    }
    if (f->copyswapn == NULL) {
        f->copyswapn = _default_copyswapn;
    }
    if (f->copyswap == NULL || f->getitem == NULL ||
        f->setitem == NULL) {
        PyErr_SetString(PyExc_ValueError, "a required array function"   \
                        " is missing.");
        return -1;
    }
    if (descr->typeobj == NULL) {
        PyErr_SetString(PyExc_ValueError, "missing typeobject");
        return -1;
    }
    if (descr->flags & (NPY_ITEM_IS_POINTER | NPY_ITEM_REFCOUNT)) {
        /*
         * User dtype can't actually do reference counting, however, there
         * are existing hacks (e.g. xpress), which use a structured one:
         *     dtype((xpress.var, [('variable', 'O')]))
         * so we have to support this. But such a structure must be constant
         * (i.e. fixed at registration time, this is the case for `xpress`).
         */
        if (descr->names == NULL || descr->fields == NULL ||
            !PyDict_CheckExact(descr->fields)) {
            PyErr_Format(PyExc_ValueError,
                    "Failed to register dtype for %S: Legacy user dtypes "
                    "using `NPY_ITEM_IS_POINTER` or `NPY_ITEM_REFCOUNT` are "
                    "unsupported.  It is possible to create such a dtype only "
                    "if it is a structured dtype with names and fields "
                    "hardcoded at registration time.\n"
                    "Please contact the NumPy developers if this used to work "
                    "but now fails.", descr->typeobj);
            return -1;
        }
    }

    if (test_deprecated_arrfuncs_members(f) < 0) {
        return -1;
    }

    userdescrs = realloc(userdescrs,
                         (NPY_NUMUSERTYPES+1)*sizeof(void *));
    if (userdescrs == NULL) {
        PyErr_SetString(PyExc_MemoryError, "RegisterDataType");
        return -1;
    }

    userdescrs[NPY_NUMUSERTYPES++] = descr;

    descr->type_num = typenum;
    if (dtypemeta_wrap_legacy_descriptor(descr) < 0) {
        descr->type_num = -1;
        NPY_NUMUSERTYPES--;
        return -1;
    }

    return typenum;
}


/*
 * Checks that there is no cast already cached using the new casting-impl
 * mechanism.
 * In that case, we do not clear out the cache (but otherwise silently
 * continue).  Users should not modify casts after they have been used,
 * but this may also happen accidentally during setup (and may never have
 * mattered).  See https://github.com/numpy/numpy/issues/20009
 */
static int _warn_if_cast_exists_already(
        PyArray_Descr *descr, int totype, char *funcname)
{
    PyArray_DTypeMeta *to_DType = PyArray_DTypeFromTypeNum(totype);
    if (to_DType == NULL) {
        return -1;
    }
    PyObject *cast_impl = PyDict_GetItemWithError(
            NPY_DT_SLOTS(NPY_DTYPE(descr))->castingimpls, (PyObject *)to_DType);
    Py_DECREF(to_DType);
    if (cast_impl == NULL) {
        if (PyErr_Occurred()) {
            return -1;
        }
    }
    else {
        char *extra_msg;
        if (cast_impl == Py_None) {
            extra_msg = "the cast will continue to be considered impossible.";
        }
        else {
            extra_msg = "the previous definition will continue to be used.";
        }
        Py_DECREF(cast_impl);
        PyArray_Descr *to_descr = PyArray_DescrFromType(totype);
        int ret = PyErr_WarnFormat(PyExc_RuntimeWarning, 1,
                "A cast from %R to %R was registered/modified using `%s` "
                "after the cast had been used.  "
                "This registration will have (mostly) no effect: %s\n"
                "The most likely fix is to ensure that casts are the first "
                "thing initialized after dtype registration.  "
                "Please contact the NumPy developers with any questions!",
                descr, to_descr, funcname, extra_msg);
        Py_DECREF(to_descr);
        if (ret < 0) {
            return -1;
        }
    }
    return 0;
}

/*NUMPY_API
  Register Casting Function
  Replaces any function currently stored.
*/
NPY_NO_EXPORT int
PyArray_RegisterCastFunc(PyArray_Descr *descr, int totype,
                         PyArray_VectorUnaryFunc *castfunc)
{
    PyObject *cobj, *key;
    int ret;

    if (totype >= NPY_NTYPES && !PyTypeNum_ISUSERDEF(totype)) {
        PyErr_SetString(PyExc_TypeError, "invalid type number.");
        return -1;
    }
    if (_warn_if_cast_exists_already(
            descr, totype, "PyArray_RegisterCastFunc") < 0) {
        return -1;
    }

    if (totype < NPY_NTYPES_ABI_COMPATIBLE) {
        descr->f->cast[totype] = castfunc;
        return 0;
    }
    if (descr->f->castdict == NULL) {
        descr->f->castdict = PyDict_New();
        if (descr->f->castdict == NULL) {
            return -1;
        }
    }
    key = PyLong_FromLong(totype);
    if (PyErr_Occurred()) {
        return -1;
    }
    cobj = PyCapsule_New((void *)castfunc, NULL, NULL);
    if (cobj == NULL) {
        Py_DECREF(key);
        return -1;
    }
    ret = PyDict_SetItem(descr->f->castdict, key, cobj);
    Py_DECREF(key);
    Py_DECREF(cobj);
    return ret;
}

/*NUMPY_API
 * Register a type number indicating that a descriptor can be cast
 * to it safely
 */
NPY_NO_EXPORT int
PyArray_RegisterCanCast(PyArray_Descr *descr, int totype,
                        NPY_SCALARKIND scalar)
{
    /*
     * If we were to allow this, the casting lookup table for
     * built-in types needs to be modified, as cancastto is
     * not checked for them.
     */
    if (!PyTypeNum_ISUSERDEF(descr->type_num) &&
                                        !PyTypeNum_ISUSERDEF(totype)) {
        PyErr_SetString(PyExc_ValueError,
                        "At least one of the types provided to "
                        "RegisterCanCast must be user-defined.");
        return -1;
    }
    if (_warn_if_cast_exists_already(
            descr, totype, "PyArray_RegisterCanCast") < 0) {
        return -1;
    }

    if (scalar == NPY_NOSCALAR) {
        /*
         * register with cancastto
         * These lists won't be freed once created
         * -- they become part of the data-type
         */
        if (descr->f->cancastto == NULL) {
            descr->f->cancastto = (int *)malloc(1*sizeof(int));
            if (descr->f->cancastto == NULL) {
                PyErr_NoMemory();
                return -1;
            }
            descr->f->cancastto[0] = NPY_NOTYPE;
        }
        return _append_new(&descr->f->cancastto, totype);
    }
    else {
        /* register with cancastscalarkindto */
        if (descr->f->cancastscalarkindto == NULL) {
            int i;
            descr->f->cancastscalarkindto =
                (int **)malloc(NPY_NSCALARKINDS* sizeof(int*));
            if (descr->f->cancastscalarkindto == NULL) {
                PyErr_NoMemory();
                return -1;
            }
            for (i = 0; i < NPY_NSCALARKINDS; i++) {
                descr->f->cancastscalarkindto[i] = NULL;
            }
        }
        if (descr->f->cancastscalarkindto[scalar] == NULL) {
            descr->f->cancastscalarkindto[scalar] =
                (int *)malloc(1*sizeof(int));
            if (descr->f->cancastscalarkindto[scalar] == NULL) {
                PyErr_NoMemory();
                return -1;
            }
            descr->f->cancastscalarkindto[scalar][0] =
                NPY_NOTYPE;
        }
        return _append_new(&descr->f->cancastscalarkindto[scalar], totype);
    }
}


/*
 * Legacy user DTypes implemented the common DType operation
 * (as used in type promotion/result_type, and e.g. the type for
 * concatenation), by using "safe cast" logic.
 *
 * New DTypes do have this behaviour generally, but we use can-cast
 * when legacy user dtypes are involved.
 */
NPY_NO_EXPORT PyArray_DTypeMeta *
legacy_userdtype_common_dtype_function(
        PyArray_DTypeMeta *cls, PyArray_DTypeMeta *other)
{
    int skind1 = NPY_NOSCALAR, skind2 = NPY_NOSCALAR, skind;

    if (!NPY_DT_is_legacy(other)) {
        /* legacy DTypes can always defer to new style ones */
        Py_INCREF(Py_NotImplemented);
        return (PyArray_DTypeMeta *)Py_NotImplemented;
    }
    /* Defer so that only one of the types handles the cast */
    if (cls->type_num < other->type_num) {
        Py_INCREF(Py_NotImplemented);
        return (PyArray_DTypeMeta *)Py_NotImplemented;
    }

    /* Check whether casting is possible from one type to the other */
    if (PyArray_CanCastSafely(cls->type_num, other->type_num)) {
        Py_INCREF(other);
        return other;
    }
    if (PyArray_CanCastSafely(other->type_num, cls->type_num)) {
        Py_INCREF(cls);
        return cls;
    }

    /*
     * The following code used to be part of PyArray_PromoteTypes().
     * We can expect that this code is never used.
     * In principle, it allows for promotion of two different user dtypes
     * to a single NumPy dtype of the same "kind". In practice
     * using the same `kind` as NumPy was never possible due to an
     * simplification where `PyArray_EquivTypes(descr1, descr2)` will
     * return True if both kind and element size match (e.g. bfloat16 and
     * float16 would be equivalent).
     * The option is also very obscure and not used in the examples.
     */

    /* Convert the 'kind' char into a scalar kind */
    switch (cls->singleton->kind) {
        case 'b':
            skind1 = NPY_BOOL_SCALAR;
            break;
        case 'u':
            skind1 = NPY_INTPOS_SCALAR;
            break;
        case 'i':
            skind1 = NPY_INTNEG_SCALAR;
            break;
        case 'f':
            skind1 = NPY_FLOAT_SCALAR;
            break;
        case 'c':
            skind1 = NPY_COMPLEX_SCALAR;
            break;
    }
    switch (other->singleton->kind) {
        case 'b':
            skind2 = NPY_BOOL_SCALAR;
            break;
        case 'u':
            skind2 = NPY_INTPOS_SCALAR;
            break;
        case 'i':
            skind2 = NPY_INTNEG_SCALAR;
            break;
        case 'f':
            skind2 = NPY_FLOAT_SCALAR;
            break;
        case 'c':
            skind2 = NPY_COMPLEX_SCALAR;
            break;
    }

    /* If both are scalars, there may be a promotion possible */
    if (skind1 != NPY_NOSCALAR && skind2 != NPY_NOSCALAR) {

        /* Start with the larger scalar kind */
        skind = (skind1 > skind2) ? skind1 : skind2;
        int ret_type_num = _npy_smallest_type_of_kind_table[skind];

        for (;;) {

            /* If there is no larger type of this kind, try a larger kind */
            if (ret_type_num < 0) {
                ++skind;
                /* Use -1 to signal no promoted type found */
                if (skind < NPY_NSCALARKINDS) {
                    ret_type_num = _npy_smallest_type_of_kind_table[skind];
                }
                else {
                    break;
                }
            }

            /* If we found a type to which we can promote both, done! */
            if (PyArray_CanCastSafely(cls->type_num, ret_type_num) &&
                PyArray_CanCastSafely(other->type_num, ret_type_num)) {
                return PyArray_DTypeFromTypeNum(ret_type_num);
            }

            /* Try the next larger type of this kind */
            ret_type_num = _npy_next_larger_type_table[ret_type_num];
        }
    }

    Py_INCREF(Py_NotImplemented);
    return (PyArray_DTypeMeta *)Py_NotImplemented;
}


/**
 * This function wraps a legacy cast into an array-method. This is mostly
 * used for legacy user-dtypes, but for example numeric to/from datetime
 * casts were only defined that way as well.
 *
 * @param from
 * @param to
 * @param casting If `NPY_NO_CASTING` will check the legacy registered cast,
 *        otherwise uses the provided cast.
 */
NPY_NO_EXPORT int
PyArray_AddLegacyWrapping_CastingImpl(
        PyArray_DTypeMeta *from, PyArray_DTypeMeta *to, NPY_CASTING casting)
{
    if (casting < 0) {
        if (from == to) {
            casting = NPY_NO_CASTING;
        }
        else if (PyArray_LegacyCanCastTypeTo(
                from->singleton, to->singleton, NPY_SAFE_CASTING)) {
            casting = NPY_SAFE_CASTING;
        }
        else if (PyArray_LegacyCanCastTypeTo(
                from->singleton, to->singleton, NPY_SAME_KIND_CASTING)) {
            casting = NPY_SAME_KIND_CASTING;
        }
        else {
            casting = NPY_UNSAFE_CASTING;
        }
    }

    PyArray_DTypeMeta *dtypes[2] = {from, to};
    PyArrayMethod_Spec spec = {
            /* Name is not actually used, but allows identifying these. */
            .name = "legacy_cast",
            .nin = 1,
            .nout = 1,
            .casting = casting,
            .dtypes = dtypes,
    };

    if (from == to) {
        spec.flags = NPY_METH_REQUIRES_PYAPI | NPY_METH_SUPPORTS_UNALIGNED;
        PyType_Slot slots[] = {
            {NPY_METH_get_loop, &legacy_cast_get_strided_loop},
            {NPY_METH_resolve_descriptors, &legacy_same_dtype_resolve_descriptors},
            {0, NULL}};
        spec.slots = slots;
        return PyArray_AddCastingImplementation_FromSpec(&spec, 1);
    }
    else {
        spec.flags = NPY_METH_REQUIRES_PYAPI;
        PyType_Slot slots[] = {
            {NPY_METH_get_loop, &legacy_cast_get_strided_loop},
            {NPY_METH_resolve_descriptors, &simple_cast_resolve_descriptors},
            {0, NULL}};
        spec.slots = slots;
        return PyArray_AddCastingImplementation_FromSpec(&spec, 1);
    }
}
