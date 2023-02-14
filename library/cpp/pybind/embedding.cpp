#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "embedding.h"

#include <util/generic/ptr.h>
#include <util/generic/yexception.h>

namespace NPyBind {
    TEmbedding::TEmbedding(char* argv0) {
#if PY_MAJOR_VERSION < 3
        Py_SetProgramName(argv0);
        Py_Initialize();
#else
        PyStatus status;

        PyConfig config;
        PyConfig_InitPythonConfig(&config);
        // Disable parsing command line arguments
        config.parse_argv = 0;

        status = PyConfig_SetBytesString(&config, &config.program_name, argv0);
        if (PyStatus_Exception(status)) {
            PyConfig_Clear(&config);
            Py_ExitStatusException(status);
        }

        status = Py_InitializeFromConfig(&config);
        if (PyStatus_Exception(status)) {
            PyConfig_Clear(&config);
            Py_ExitStatusException(status);
        }

        PyConfig_Clear(&config);
#endif
    }

    TEmbedding::~TEmbedding() {
        Py_Finalize();
    }
}
