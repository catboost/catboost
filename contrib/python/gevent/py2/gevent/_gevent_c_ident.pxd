cimport cython

cdef extern from "Python.h":

    ctypedef class weakref.ref [object PyWeakReference]:
        pass

cdef heappop
cdef heappush
cdef object WeakKeyDictionary
cdef type ref

@cython.internal
@cython.final
cdef class ValuedWeakRef(ref):
    cdef object value

@cython.final
cdef class IdentRegistry:
    cdef object _registry
    cdef list _available_idents

    @cython.final
    cpdef object get_ident(self, obj)
    @cython.final
    cpdef _return_ident(self, ValuedWeakRef ref)
