#include <Python.h>

#include <library/python/symbols/registry/syms.h>

// Internal API not declared by <Python.h>; declared opaquely to take &sym.
#if PY_VERSION_HEX >= 0x030b0000
extern "C" {
PyAPI_FUNC(int) _PyFrame_IsEntryFrame();
PyAPI_FUNC(const char *) _PyMem_GetCurrentAllocatorName();
}
#endif

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
SYM(PyDict_SetItem)
SYM(PyMem_Malloc)
SYM(PyObject_GetBuffer)
SYM(PyOS_Readline)
SYM(PyType_Modified)
SYM(Py_DecRef)
SYM(Py_IncRef)
SYM(_Py_NotImplementedStruct)

// Keep + export for attach-based profilers (vanilla memray attach).
#if PY_VERSION_HEX >= 0x3000000
SYM(Py_CompileString)
SYM(Py_Version)
SYM(PyObject_HasAttr)
SYM(PyImport_GetModuleDict)
SYM(PyEval_SetProfile)
SYM(PyDict_SetDefault)
SYM(_PyDict_NewPresized)
#endif
#if PY_VERSION_HEX >= 0x030b0000  // 3.11+
SYM(PyFrame_GetLasti)
SYM(_PyFrame_IsEntryFrame)
SYM(_PyMem_GetCurrentAllocatorName)
#endif
#if PY_VERSION_HEX >= 0x030c0000  // 3.12+
SYM(PyCode_AddWatcher)
SYM(PyEval_SetProfileAllThreads)
#endif
#if PY_VERSION_HEX >= 0x030d0000  // 3.13+
SYM(PyThreadState_GetUnchecked)
#endif

#if PY_VERSION_HEX < 0x3000000
SYM(PyFile_SetEncoding)
SYM(PyFile_AsFile)
#endif

END_SYMS()
