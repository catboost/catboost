#include <Python.h>
#include <contrib/tools/python3/Include/internal/pycore_runtime.h> // _PyRuntime_Initialize()

#include <stdlib.h>
#include <string.h>
#include <locale.h>

char* GetPyMain();
int IsYaIdeVenv();

static const char* env_entry_point = "Y_PYTHON_ENTRY_POINT";
static const char* main_entry_point = ":main";
static const char* env_bytes_warning = "Y_PYTHON_BYTES_WARNING";

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

static int RunModule(const char* modname)
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

#ifdef MS_WINDOWS
static int pymain(int argc, wchar_t** argv)
#else
static int pymain(int argc, char** argv)
#endif
{
    PyStatus status;

    if (IsYaIdeVenv()) {
#ifdef MS_WINDOWS
        return Py_Main(argc, argv);
#else
        return Py_BytesMain(argc, argv);
#endif
    }

    status = _PyRuntime_Initialize();
    if (PyStatus_Exception(status)) {
        Py_ExitStatusException(status);
    }

    PyPreConfig preconfig;
    PyPreConfig_InitPythonConfig(&preconfig);
    // Enable UTF-8 mode for all (DEVTOOLSSUPPORT-46624)
    preconfig.utf8_mode = 1;
#ifdef MS_WINDOWS
    preconfig.legacy_windows_fs_encoding = 0;
#endif

    status = Py_PreInitialize(&preconfig);
    if (PyStatus_Exception(status)) {
        Py_ExitStatusException(status);
    }

    int sts = 1;
    char* entry_point_copy = NULL;

    PyConfig config;
    PyConfig_InitPythonConfig(&config);
    // Suppress errors from getpath.c
    config.pathconfig_warnings = 0;
    // Disable parsing command line arguments
    config.parse_argv = 0;

    const char* bytes_warning = getenv(env_bytes_warning);
    if (bytes_warning) {
        config.bytes_warning = atoi(bytes_warning);
    }

    if (argc > 0 && argv) {
#ifdef MS_WINDOWS
        status = PyConfig_SetString(&config, &config.program_name, argv[0]);
#else
        status = PyConfig_SetBytesString(&config, &config.program_name, argv[0]);
#endif
        if (PyStatus_Exception(status)) {
            goto error;
        }

#ifdef MS_WINDOWS
        status = PyConfig_SetArgv(&config, argc, argv);
#else
        status = PyConfig_SetBytesArgv(&config, argc, argv);
#endif
        if (PyStatus_Exception(status)) {
            goto error;
        }
    }

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

    if (entry_point_copy && !strcmp(entry_point_copy, main_entry_point)) {
        unsetenv(env_entry_point);
        // Py_InitializeFromConfig freeze environ, so we need to finish all manipulations with environ before
    }

    status = Py_InitializeFromConfig(&config);

    PyConfig_Clear(&config);
    if (PyStatus_Exception(status)) {
        Py_ExitStatusException(status);
    }

    if (entry_point_copy && !strcmp(entry_point_copy, main_entry_point)) {
#ifdef MS_WINDOWS
        sts = Py_Main(argc, argv);
#else
        sts = Py_BytesMain(argc, argv);
#endif
        free(entry_point_copy);
        return sts;
    }

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

    char* colon = strchr(entry_point_copy, ':');
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
    return sts;
}

#ifdef MS_WINDOWS
int (*mainptr)(int argc, wchar_t** argv) = pymain;

int wmain(int argc, wchar_t** argv) {
    return mainptr(argc, argv);
}
#else
int (*mainptr)(int argc, char** argv) = pymain;

int main(int argc, char** argv) {
    return mainptr(argc, argv);
}
#endif
