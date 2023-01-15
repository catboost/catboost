#define SYM(SYM_NAME) extern "C" void SYM_NAME();
SYM(PyObject_GetBuffer)
SYM(PyBuffer_Release)
SYM(PyCell_New)
#undef SYM

#include <library/python/symbols/registry/syms.h>

BEGIN_SYMS("python")
SYM(PyObject_GetBuffer)
SYM(PyBuffer_Release)
SYM(PyCell_New)
END_SYMS()
