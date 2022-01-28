#include <Python.h>

#include <stdlib.h>
#include <string.h>

void Py_InitArgcArgv(int argc, char** argv);

static const char* env_entry_point = "Y_PYTHON_ENTRY_POINT";

#ifdef _MSC_VER
extern char** environ;

void unsetenv(const char* name) {
    const int n = strlen(name);
    char** dst = environ;
    for (char** src = environ; *src; src++)
        if (strncmp(*src, name, n) || (*src)[n] != '=')
            *dst++ = *src;
    *dst = NULL;
}
#endif

static int pymain(int argc, char** argv) {
    const char* entry_point = getenv(env_entry_point);
    if (entry_point && !strcmp(entry_point, ":main")) {
        unsetenv(env_entry_point);
        return Py_Main(argc, argv);
    }
    Py_InitArgcArgv(argc, argv);
    Py_SetProgramName(argv[0]);
    Py_Initialize();
    PySys_SetArgv(argc, argv);
    int rc = PyRun_SimpleString(
        "from library.python.runtime import entry_points\n"
        "entry_points.run_constructors()\n"
        "import __res\n"
        "__res.importer.run_main()\n");
    Py_Finalize();
    return rc == 0 ? 0 : 1;
}

int (*mainptr)(int argc, char** argv) = pymain;

int main(int argc, char** argv) {
    return mainptr(argc, argv);
}
