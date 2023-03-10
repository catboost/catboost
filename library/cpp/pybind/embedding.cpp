#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "embedding.h"

#include <util/generic/ptr.h>
#include <util/generic/yexception.h>

namespace NPyBind {
#if PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION < 8
    class TDeleteRawMem {
    public:
        template <typename T>
        static inline void Destroy(T* t) noexcept {
            PyMem_RawFree(t);
        }
    };

    template <typename T>
    using TRawMemHolder = THolder<T, TDeleteRawMem>;

    static void SetProgramName(char* name) {
        TRawMemHolder<wchar_t> wideName(Py_DecodeLocale(name, nullptr));
        Y_ENSURE(wideName);
        Py_SetProgramName(wideName.Get());
    }
#endif

    TEmbedding::TEmbedding(char* argv0) {
#if PY_MAJOR_VERSION < 3
        Py_SetProgramName(argv0);
        Py_Initialize();
#elif PY_MAJOR_VERSION >= 3 && PY_MINOR_VERSION >= 8
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
#elif PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION < 8
        SetProgramName(argv0);
        Py_Initialize();
#endif
    }

    TEmbedding::~TEmbedding() {
        Py_Finalize();
    }
}
