#include <Python.h>

#include <library/python/symbols/registry/syms.h>

BEGIN_SYMS("python")
SYM(PyBuffer_Release)
SYM(PyCapsule_GetContext)
SYM(PyCapsule_GetDestructor)
SYM(PyCapsule_GetName)
SYM(PyCapsule_GetPointer)
SYM(PyCapsule_IsValid)
SYM(PyCapsule_New)
SYM(PyCapsule_SetContext)
SYM(PyCapsule_SetDestructor)
SYM(PyCapsule_SetName)
SYM(PyCapsule_SetPointer)
SYM(PyCell_New)
SYM(PyMem_Malloc)
SYM(PyObject_GetBuffer)
SYM(PyOS_Readline)
SYM(Py_DecRef)
SYM(Py_IncRef)

#if PY_VERSION_HEX < 0x3000000
SYM(PyFile_SetEncoding)
SYM(PyFile_AsFile)
#endif
END_SYMS()
