#include <stdlib.h>
#include <stdio.h>

#include "vars.h"

#include <util/system/platform.h>

#define main RealMain
#include "Modules/python.c"
#undef main

int main(int argc, char** argv) {
    char* distArcPath = getenv("ARCADIA_ROOT_DISTBUILD");
    char* pyPath = NULL;
    char* mx = 0;
    char* x = 0;
    int ret;
    putenv("PYTHONHOME=");
    putenv("PY_IGNORE_ENVIRONMENT=");
    putenv("PYTHONDONTWRITEBYTECODE=x");
    if (distArcPath) {
        pyPath = malloc(strlen("PYTHONPATH=") + strlen(distArcPath) + strlen(GetPyLib()) + 2);

        if (!pyPath)
            abort();

        mx = strdup(GetPyLib());
        x = mx;

        if (!x)
            abort();

        if (*x && *x == '"') {
            x += 1;
            x[strlen(x) - 1] = 0;
        }

        sprintf(pyPath, "PYTHONPATH=%s/%s", distArcPath, x);
    } else {
        pyPath = malloc(strlen("PYTHONPATH=") + strlen(GetLibDir()) + 1);
        sprintf(pyPath, "PYTHONPATH=%s", GetLibDir());
    }
    putenv(pyPath);
    ret =  RealMain(argc, argv);
    if (pyPath)
        free(pyPath);
    if (mx)
        free(mx);
    return ret;
}
