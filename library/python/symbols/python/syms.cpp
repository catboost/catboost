#define SYM(SYM_NAME) extern "C" void SYM_NAME();
SYM(PyObject_GetBuffer)
SYM(PyBuffer_Release)
SYM(PyCell_New)
SYM(Py_DecRef)
SYM(Py_IncRef)
#undef SYM

#include <library/python/symbols/registry/syms.h>

BEGIN_SYMS("python")
SYM(PyObject_GetBuffer)
SYM(PyBuffer_Release)
SYM(PyCell_New)
SYM(Py_DecRef)
SYM(Py_IncRef)
END_SYMS()
