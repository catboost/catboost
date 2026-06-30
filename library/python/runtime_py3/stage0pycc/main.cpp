#include <util/folder/path.h>
#include <util/generic/scope.h>
#include <util/generic/string.h>
#include <util/stream/file.h>
#include <util/stream/output.h>

#include <Python.h>
#include <marshal.h>

#include <cstdio>
#include <system_error>

struct TPyObjDeleter {
    static void Destroy(PyObject* o) noexcept {
        Py_XDECREF(o);
    }
};
using TPyObject = THolder<PyObject, TPyObjDeleter>;

constexpr TStringBuf modPrefix = "mod=";

int main(int argc, char** argv) {
    if ((argc - 1) % 3 != 0) {
        Cerr << "Usage:\n\t" << argv[0] << " (mod=SRC_PATH_X SRC OUT)+" << Endl;
        return 1;
    }

    PyConfig cfg{};
    PyConfig_InitIsolatedConfig(&cfg);
    cfg._install_importlib = 0;
    Y_SCOPE_EXIT(&cfg) {PyConfig_Clear(&cfg);};

    for (int i = 0; i < (argc - 1)/3; ++i) {
        const TString srcpath{TStringBuf{argv[3*i + 1]}.substr(modPrefix.size())};
        const TFsPath inPath{argv[3*i + 2]};
        const char* outPath = argv[3*i + 3];

        const auto status = Py_InitializeFromConfig(&cfg);
        if (PyStatus_Exception(status)) {
            Py_ExitStatusException(status);
        }
        Y_SCOPE_EXIT() {Py_Finalize();};

        TPyObject bytecode{Py_CompileString(
            TFileInput{inPath}.ReadAll().c_str(),
            srcpath.c_str(),
            Py_file_input
        )};
        if (!bytecode) {
            Cerr << "Failed to compile " << outPath << Endl;
            PyErr_Print();
            return 1;
        }

        if (FILE* out = fopen(outPath, "wb")) {
            PyMarshal_WriteObjectToFile(bytecode.Get(), out, Py_MARSHAL_VERSION);
            fclose(out);
            if (PyErr_Occurred()) {
                Cerr << "Failed to marshal " << outPath << Endl;
                PyErr_Print();
                return 1;
            }
        } else {
            Cerr << "Failed to write " << outPath << ": " << std::error_code{errno, std::system_category()}.message() << Endl;
            return 1;
        }
    }

    return 0;
}
