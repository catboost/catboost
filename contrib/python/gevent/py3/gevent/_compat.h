#ifndef _COMPAT_H
#define _COMPAT_H

/**
 * Compatibility helpers for things that are better handled at C
 * compilation time rather than Cython code generation time.
 */

#include <Python.h>

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if PY_VERSION_HEX >= 0x30B00A6
#  define GEVENT_PY311 1
#else
#  define GEVENT_PY311 0
#  define _PyCFrame CFrame
#endif

/* FrameType and CodeType changed a lot in 3.11. */
#if GREENLET_PY311
   /* _PyInterpreterFrame moved to the internal C API in Python 3.11 */
#  include <internal/pycore_frame.h>
#else
#include <frameobject.h>
#if PY_MAJOR_VERSION < 3 || (PY_MAJOR_VERSION >= 3 && PY_MINOR_VERSION < 9)
/* these were added in 3.9, though they officially became stable in 3.10 */
/* the official versions of these functions return strong references, so we
   need to increment the refcount before returning, not just to match the
   official functions, but to match what Cython expects an API like this to
   return. Otherwise we get crashes. */
static PyObject* PyFrame_GetBack(PyFrameObject* frame)
{
    PyObject* result = (PyObject*)((PyFrameObject*)frame)->f_back;
    Py_XINCREF(result);
    return result;
}

static PyObject* PyFrame_GetCode(PyFrameObject* frame)
{
    PyObject* result = (PyObject*)((PyFrameObject*)frame)->f_code;
    /* There is always code!  */
    Py_INCREF(result);
    return result;
}
#endif /* support 3.8 and below. */
#endif

/**
   Unlike PyFrame_GetBack, which can return NULL,
   this method is guaranteed to return a new reference to an object.

   The object is either a frame object or None.

   This is necessary to help Cython deal correctly with reference counting.
   (There are other ways of dealing with this having to do with exactly how
   variables/return types are declared IIRC, but this is the most
   straightforward. Still, it is critical that the cython declaration of
   this function use ``object`` as its return type.)
 */
static PyObject* Gevent_PyFrame_GetBack(PyObject* frame)
{
    PyObject* back = (PyObject*)PyFrame_GetBack((PyFrameObject*)frame);
    if (back) {
        return back;
    }
    Py_RETURN_NONE;
}

/* These are just for typing purposes to appease the compiler. */

static int Gevent_PyFrame_GetLineNumber(PyObject* o)
{
    return PyFrame_GetLineNumber((PyFrameObject*)o);
}

static PyObject* Gevent_PyFrame_GetCode(PyObject* o)
{
    return (PyObject*)PyFrame_GetCode((PyFrameObject*)o);
}

#ifdef __cplusplus
}
#ifdef __clang__
#pragma clang diagnostic pop
#endif
#endif
#endif /* _COMPAT_H */
