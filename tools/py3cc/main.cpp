#include <util/folder/path.h>
#include <util/generic/scope.h>
#include <util/generic/string.h>
#include <util/stream/file.h>
#include <util/stream/output.h>
#include <util/system/shellcommand.h>

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

int runSlowPy3cc(const char* slowPy3cc, const char* srcpath, const char* inPath, const char* outPath) {
    TShellCommandOptions opts;
    opts.SetUseShell(false);
    opts.SetOutputStream(&Cout);
    opts.SetErrorStream(&Cerr);

    TShellCommand cmd(slowPy3cc, {srcpath, inPath, outPath}, opts);
    cmd.Run().Wait();

    if (auto rc = cmd.GetExitCode(); rc.Defined()) {
        return *rc;
    }
    return 1;
}

int main(int argc, char** argv) {
    if (argc != 6) {
        Cerr << "Usage:\n\t" << argv[0] << "--slow-py3cc <slow-py3cc> SRC_PATH_X- SRC OUT" << Endl;
        return 1;
    }

    PyConfig cfg{};
    PyConfig_InitIsolatedConfig(&cfg);
    cfg._install_importlib = 0;
    Y_SCOPE_EXIT(&cfg) {PyConfig_Clear(&cfg);};

    const char* slowPy3cc{argv[2]};
    TString srcpath{argv[3]};
    srcpath.pop_back();
    const TFsPath inPath{argv[4]};
    const char* outPath = argv[5];

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
        int rc = runSlowPy3cc(slowPy3cc, argv[3], argv[4], argv[5]);
        return rc;
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
