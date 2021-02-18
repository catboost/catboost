#define SYM(SYM_NAME) extern "C" void SYM_NAME();
SYM(PyFile_SetEncoding)
SYM(PyFile_AsFile)
SYM(PyMem_Malloc)
SYM(PyOS_Readline)
#undef SYM

#include <library/python/symbols/registry/syms.h>

BEGIN_SYMS("python")
SYM(PyFile_SetEncoding)
SYM(PyFile_AsFile)
SYM(PyMem_Malloc)
SYM(PyOS_Readline)
END_SYMS()
