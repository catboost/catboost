#include <Python.h>
#include <marshal.h>

#include <iterator>

namespace {

const unsigned char sitecustomize_pyc[] = {
    #include "sitecustomize.pyc.inc"
};

int modsitecustomize_exec(PyObject *mod) noexcept {
    PyObject *bytecode = PyMarshal_ReadObjectFromString(
        reinterpret_cast<const char*>(sitecustomize_pyc),
        std::size(sitecustomize_pyc)
    );
    if (!bytecode) {
        return -1;
    }
    PyObject* modns = PyModule_GetDict(mod);
    if (!modns) {
        return -1;
    }
    if (PyObject* evalRes = PyEval_EvalCode(bytecode, modns, modns)) {
        Py_DECREF(evalRes);
    }
    if (PyErr_Occurred()) {
        return -1;
    }
    return 0;
}

PyDoc_STRVAR(modsitecustomize_doc,
"bridge between Arcadia resource system and python importlib resources interface.");

PyMethodDef modsitecustomize_methods[] = {
    {nullptr, nullptr, 0, nullptr}
};

PyModuleDef_Slot modsitecustomize_slots[] = {
    {Py_mod_exec, reinterpret_cast<void*>(&modsitecustomize_exec)},
    {Py_mod_multiple_interpreters, Py_MOD_PER_INTERPRETER_GIL_SUPPORTED},
    {0, nullptr},
};

PyModuleDef modsitecustomize = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "sitecustomize",
    .m_doc = modsitecustomize_doc,
    .m_size = 0,
    .m_methods = modsitecustomize_methods,
    .m_slots = modsitecustomize_slots,
    .m_traverse = nullptr,
    .m_clear = nullptr,
    .m_free = nullptr
};

}

PyMODINIT_FUNC
PyInit_sitecustomize() noexcept {
    return PyModuleDef_Init(&modsitecustomize);
}
