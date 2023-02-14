#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define _STR(a) #a
#define STR(a) _STR(a)

static const char* ArcadiaRoot() {
    const char* root = getenv("ARCADIA_ROOT_DISTBUILD");
    return root ? root : STR(ARCADIA_ROOT);
}

#ifdef _MSC_VER
static int setenv(const char* name, const char* value, int overwrite) {
    return (overwrite || !getenv(name)) ? _putenv_s(name, value) : 0;
}
#endif

static void InitSwigLib() {
    const char* root = ArcadiaRoot();
    const char* lib = STR(SWIG_LIB_ARCPATH);
    char* s = new char[strlen(root) + 1 + strlen(lib) + 1];
    sprintf(s, "%s/%s", root, lib);
    setenv("SWIG_LIB", s, false);
    delete[] s;
}

static int initSwigLib = (InitSwigLib(), 0);
