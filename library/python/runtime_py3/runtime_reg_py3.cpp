#include <Python.h>

extern "C" PyObject* PyInit___res();
extern "C" PyObject* PyInit_sitecustomize();

namespace {
    struct TRegistrar {
        inline TRegistrar() {
            _inittab mods[] = {
                {"__res", PyInit___res},
                {"sitecustomize", PyInit_sitecustomize},
                {nullptr, nullptr}
            };
            PyImport_ExtendInittab(mods);
        }
    } REG;
}
