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

int main(int argc, char** argv) {
    if (argc < 4) {
        Cerr << "Usage: " << argv[0] << " SRC_PATH_X SRC OUT" << Endl;
        return 1;
    }

    TString srcpath{argv[1]};
    srcpath.pop_back();
    const TFsPath inPath{argv[2]};
    const char* outPath = argv[3];

    Py_Initialize();
    Y_SCOPE_EXIT() {
        Py_Finalize();
    };

    TPyObject bytecode{Py_CompileString(
        TFileInput{inPath}.ReadAll().c_str(),
        srcpath.c_str(),
        Py_file_input)};
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

    return 0;
}
