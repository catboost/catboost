#include "vars.h"

#define RAWSTR(a) RAWXSTR(__weirdrawstr(a)__weirdrawstr)
#define RAWXSTR(a) R ## #a

extern "C" {
    const char* GetLibDir() THROWING {
        return RAWSTR(LIBDIR);
    }

    const char* GetPyLib() THROWING {
        return RAWSTR(PYLIB);
    }
}
