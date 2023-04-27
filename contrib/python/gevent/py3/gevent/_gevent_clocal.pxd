# cython: auto_pickle=False

cimport cython
from gevent._gevent_cgreenlet cimport Greenlet

cdef bint _PYPY
cdef ref
cdef copy

cdef object _marker
cdef str key_prefix
cdef bint _greenlet_imported


cdef extern from "greenlet/greenlet.h":

    ctypedef class greenlet.greenlet [object PyGreenlet]:
        pass

    # These are actually macros and so much be included
    # (defined) in each .pxd, as are the two functions
    # that call them.
    greenlet PyGreenlet_GetCurrent()
    void PyGreenlet_Import()

cdef inline greenlet getcurrent():
    return PyGreenlet_GetCurrent()

cdef inline void greenlet_init():
    global _greenlet_imported
    if not _greenlet_imported:
        PyGreenlet_Import()
        _greenlet_imported = True


cdef void _init()

@cython.final
@cython.internal
cdef class _wrefdict(dict):
   cdef object __weakref__

@cython.final
@cython.internal
cdef class _greenlet_deleted:
    cdef object idt
    cdef object wrdicts


@cython.final
@cython.internal
cdef class _local_deleted:
    cdef str key
    cdef object wrthread
    cdef _greenlet_deleted greenlet_deleted

@cython.final
@cython.internal
cdef class _localimpl:
    cdef str key
    cdef dict dicts
    cdef tuple localargs
    cdef dict localkwargs
    cdef tuple localtypeid
    cdef object __weakref__


@cython.final
@cython.internal
cdef class _localimpl_dict_entry:
    cdef object wrgreenlet
    cdef dict localdict

@cython.locals(localdict=dict, key=str,
               greenlet_deleted=_greenlet_deleted,
               local_deleted=_local_deleted)
cdef dict _localimpl_create_dict(_localimpl self,
                                 greenlet greenlet,
                                 object idt)

cdef set _local_attrs

cdef class local:
    cdef _localimpl _local__impl
    cdef set _local_type_get_descriptors
    cdef set _local_type_set_or_del_descriptors
    cdef set _local_type_del_descriptors
    cdef set _local_type_set_descriptors
    cdef set _local_type_vars
    cdef type _local_type

    @cython.locals(entry=_localimpl_dict_entry,
                   dct=dict, duplicate=dict,
                   instance=local)
    cpdef local __copy__(local self)


@cython.locals(impl=_localimpl,dct=dict,
               dct=dict, entry=_localimpl_dict_entry)
cdef inline dict _local_get_dict(local self)

@cython.locals(entry=_localimpl_dict_entry)
cdef _local__copy_dict_from(local self, _localimpl impl, dict duplicate)

@cython.locals(mro=list, gets=set, dels=set, set_or_del=set,
               type_self=type, type_attr=type,
               sets=set)
cdef tuple _local_find_descriptors(local self)

@cython.locals(result=list, local_impl=_localimpl,
               entry=_localimpl_dict_entry, k=str,
               greenlet_dict=dict)
cpdef all_local_dicts_for_greenlet(greenlet greenlet)
