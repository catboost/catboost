#include <Python.h>

#include <stdlib.h>
#include <string.h>
#include <locale.h>

void Py_InitArgcArgv(int argc, wchar_t **argv);
char* GetPyMain();

static const char* env_entry_point = "Y_PYTHON_ENTRY_POINT";

#ifdef _MSC_VER
extern char** environ;

void unsetenv(const char* name) {
    const int n = strlen(name);
    const char** dst = environ;
    for (const char** src = environ; *src; src++)
        if (strncmp(*src, name, n) || (*src)[n] != '=')
            *dst++ = *src;
    *dst = NULL;
}
#endif

static int RunModule(const char *modname)
{
    PyObject *module, *runpy, *runmodule, *runargs, *result;
    runpy = PyImport_ImportModule("runpy");
    if (runpy == NULL) {
        fprintf(stderr, "Could not import runpy module\n");
        PyErr_Print();
        return -1;
    }
    runmodule = PyObject_GetAttrString(runpy, "_run_module_as_main");
    if (runmodule == NULL) {
        fprintf(stderr, "Could not access runpy._run_module_as_main\n");
        PyErr_Print();
        Py_DECREF(runpy);
        return -1;
    }
    module = PyUnicode_FromString(modname);
    if (module == NULL) {
        fprintf(stderr, "Could not convert module name to unicode\n");
        PyErr_Print();
        Py_DECREF(runpy);
        Py_DECREF(runmodule);
        return -1;
    }
    runargs = Py_BuildValue("(Oi)", module, 0);
    if (runargs == NULL) {
        fprintf(stderr,
            "Could not create arguments for runpy._run_module_as_main\n");
        PyErr_Print();
        Py_DECREF(runpy);
        Py_DECREF(runmodule);
        Py_DECREF(module);
        return -1;
    }
    result = PyObject_Call(runmodule, runargs, NULL);
    if (result == NULL) {
        PyErr_Print();
    }
    Py_DECREF(runpy);
    Py_DECREF(runmodule);
    Py_DECREF(module);
    Py_DECREF(runargs);
    if (result == NULL) {
        return -1;
    }
    Py_DECREF(result);
    return 0;
}

static int pymain(int argc, char** argv) {
    int i, sts = 1;
    char* oldloc = NULL;
    wchar_t** argv_copy = NULL;
    /* We need a second copies, as Python might modify the first one. */
    wchar_t** argv_copy2 = NULL;
    char* entry_point_copy = NULL;

    if (argc > 0) {
        argv_copy = PyMem_RawMalloc(sizeof(wchar_t*) * argc);
        argv_copy2 = PyMem_RawMalloc(sizeof(wchar_t*) * argc);
        if (!argv_copy || !argv_copy2) {
            fprintf(stderr, "out of memory\n");
            goto error;
        }
    }

    oldloc = _PyMem_RawStrdup(setlocale(LC_ALL, NULL));
    if (!oldloc) {
        fprintf(stderr, "out of memory\n");
        goto error;
    }

    setlocale(LC_ALL, "");
    for (i = 0; i < argc; i++) {
        argv_copy[i] = Py_DecodeLocale(argv[i], NULL);
        argv_copy2[i] = argv_copy[i];
        if (!argv_copy[i]) {
            fprintf(stderr, "Unable to decode the command line argument #%i\n",
                    i + 1);
            argc = i;
            goto error;
        }
    }
    setlocale(LC_ALL, oldloc);
    PyMem_RawFree(oldloc);
    oldloc = NULL;

    const char* entry_point = getenv(env_entry_point);
    if (entry_point) {
        entry_point_copy = strdup(entry_point);
        if (!entry_point_copy) {
            fprintf(stderr, "out of memory\n");
            goto error;
        }
    } else {
        entry_point_copy = GetPyMain();
    }

    if (entry_point_copy == NULL) {
        fprintf(stderr, "No entry point, did you forget PY_MAIN?\n");
        goto error;
    }

    if (entry_point_copy && !strcmp(entry_point_copy, ":main")) {
        unsetenv(env_entry_point);
        sts = Py_Main(argc, argv_copy);
        free(entry_point_copy);
        return sts;
    }

    Py_InitArgcArgv(argc, argv_copy);
    if (argc >= 1)
        Py_SetProgramName(argv_copy[0]);
    Py_Initialize();

    PySys_SetArgv(argc, argv_copy);

    {
        PyObject* module = PyImport_ImportModule("library.python.runtime_py3.entry_points");
        if (module == NULL) {
            PyErr_Print();
        } else {
            PyObject* res = PyObject_CallMethod(module, "run_constructors", NULL);
            if (res == NULL) {
                PyErr_Print();
            } else {
                Py_DECREF(res);
            }
            Py_DECREF(module);
        }
    }

    const char* module_name = entry_point_copy;
    const char* func_name = NULL;

    char *colon = strchr(entry_point_copy, ':');
    if (colon != NULL) {
        colon[0] = '\0';
        func_name = colon + 1;
    }
    if (module_name[0] == '\0') {
        module_name = "library.python.runtime_py3.entry_points";
    }

    if (!func_name) {
        sts = RunModule(module_name);
    } else {
        PyObject* module = PyImport_ImportModule(module_name);

        if (module == NULL) {
            PyErr_Print();
        } else {
            PyObject* value = PyObject_CallMethod(module, func_name, NULL);

            if (value == NULL) {
                PyErr_Print();
            } else {
                Py_DECREF(value);
                sts = 0;
            }

            Py_DECREF(module);
        }
    }

    if (Py_FinalizeEx() < 0) {
        sts = 120;
    }

error:
    free(entry_point_copy);
    PyMem_RawFree(argv_copy);
    if (argv_copy2) {
        for (i = 0; i < argc; i++)
            PyMem_RawFree(argv_copy2[i]);
        PyMem_RawFree(argv_copy2);
    }
    PyMem_RawFree(oldloc);
    return sts;
}

int (*mainptr)(int argc, char** argv) = pymain;

int main(int argc, char** argv) {
    return mainptr(argc, argv);
}
