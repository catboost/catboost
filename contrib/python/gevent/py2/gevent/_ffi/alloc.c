#include <stddef.h>
#ifndef GEVENT_ALLOC_C
#define GEVENT_ALLOC_C

#ifdef PYPY_VERSION_NUM
#define GGIL_DECLARE
#define GGIL_ENSURE
#define GGIL_RELEASE
#define GPyObject_Free free
#define GPyObject_Realloc realloc
#else
#include "Python.h"
#define GGIL_DECLARE  PyGILState_STATE ___save
#define GGIL_ENSURE  ___save = PyGILState_Ensure();
#define GGIL_RELEASE  PyGILState_Release(___save);
#define GPyObject_Free PyObject_Free
#define GPyObject_Realloc PyObject_Realloc
#endif

void* gevent_realloc(void* ptr, size_t size)
{
    // libev is interesting and assumes that everything can be
    // done with realloc(), assuming that passing in a size of 0 means to
    // free the pointer. But the C/++ standard explicitly says that
    // this is undefined. So this wrapper function exists to do it all.
    GGIL_DECLARE;
    void* result;
    if(!size && !ptr) {
        // libev for some reason often tries to free(NULL); I won't specutale
        // why. No need to acquire the GIL or do anything in that case.
        return NULL;
    }

    // Using PyObject_* APIs to get access to pymalloc allocator on
    // all versions of CPython; in Python 3, PyMem_* and PyObject_* use
    // the same allocator, but in Python 2, only PyObject_* uses pymalloc.
    GGIL_ENSURE;

    if(!size) {
        GPyObject_Free(ptr);
        result = NULL;
    }
    else {
        result = GPyObject_Realloc(ptr, size);
    }
    GGIL_RELEASE;
    return result;
}

#undef GGIL_DECLARE
#undef GGIL_ENSURE
#undef GGIL_RELEASE
#undef GPyObject_Free
#undef GPyObject_Realloc

#endif /* GEVENT_ALLOC_C */
