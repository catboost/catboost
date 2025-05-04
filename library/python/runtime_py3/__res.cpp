#include <library/cpp/resource/resource.h>

#include <util/generic/scope.h>
#include <util/generic/strbuf.h>

#include <Python.h>
#include <marshal.h>

#include <type_traits>
#include <concepts>

namespace {

namespace NWrap {

template<typename F>
    requires std::convertible_to<std::invoke_result_t<F>, PyObject*>
PyObject* CallWithErrorTranslation(F&& f) noexcept {
    try {
        return std::forward<F>(f)();
    } catch (const std::bad_alloc& err) {
        PyErr_SetString(PyExc_MemoryError, err.what());
    } catch(const std::out_of_range& err) {
        PyErr_SetString(PyExc_IndexError, err.what());
    } catch (const std::exception& err) {
        PyErr_SetString(PyExc_RuntimeError, err.what());
    } catch(...) {
        PyErr_SetString(PyExc_RuntimeError, "Unhandled C++ exception of unknown type");
    }
    return nullptr;
}

PyObject* Count(PyObject* self[[maybe_unused]], PyObject *args[[maybe_unused]]) noexcept {
    static_assert(
        noexcept(NResource::Count()),
        "Python3 Arcadia runtime binding assumes that NResource::Count do not throw exception. If this function start "
        "to throw someone must add code translating C++ exceptions into Python exceptions here."
    );
    return PyLong_FromLong(NResource::Count());
}

PyObject* KeyByIndex(PyObject* self[[maybe_unused]], PyObject *const *args, Py_ssize_t nargs) noexcept {
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "__res.key_by_index takes 1 positional arguments but %z were given", nargs);
        return nullptr;
    }
    if (PyFloat_Check(args[0])) {
        PyErr_SetString(PyExc_TypeError, "integer argument expected, got float");
        return nullptr;
    }
    PyObject* asNum = PyNumber_Index(args[0]);
    if (!asNum) {
        return nullptr;
    }
    const auto idx = PyLong_AsSize_t(asNum);
    Py_DECREF(asNum);
    if (idx == static_cast<size_t>(-1)) {
        return nullptr;
    }
    return CallWithErrorTranslation([&]{
        const auto res = NResource::KeyByIndex(idx);
        return PyBytes_FromStringAndSize(res.data(), res.size());
    });
}

PyObject* Find(PyObject* self[[maybe_unused]], PyObject *const* args, Py_ssize_t nargs) noexcept {
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "__res.find takes 1 positional arguments but %z were given", nargs);
        return nullptr;
    }

    TStringBuf key;
    if (PyUnicode_Check(args[0])) {
        Py_ssize_t sz;
        const char* data = PyUnicode_AsUTF8AndSize(args[0], &sz);
        if (sz < 0) {
            return nullptr;
        }
        key = {data, static_cast<size_t>(sz)};
    } else {
        char* data = nullptr;
        Py_ssize_t sz;
        if (PyBytes_AsStringAndSize(args[0], &data, &sz) != 0) {
            return nullptr;
        }
        key = {data, static_cast<size_t>(sz)};
    }

    return CallWithErrorTranslation([&]{
        TString res;
        if (!NResource::FindExact(key, &res)) {
            Py_RETURN_NONE;
        }
        return PyBytes_FromStringAndSize(res.data(), res.size());
    });
}

PyObject* Has(PyObject* self[[maybe_unused]], PyObject *const* args, Py_ssize_t nargs) noexcept {
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "__res.has takes 1 positional arguments but %z were given", nargs);
        return nullptr;
    }

    TStringBuf key;
    if (PyUnicode_Check(args[0])) {
        Py_ssize_t sz;
        const char* data = PyUnicode_AsUTF8AndSize(args[0], &sz);
        if (sz < 0) {
            return nullptr;
        }
        key = {data, static_cast<size_t>(sz)};
    } else {
        char* data = nullptr;
        Py_ssize_t sz;
        if (PyBytes_AsStringAndSize(args[0], &data, &sz) != 0) {
            return nullptr;
        }
        key = {data, static_cast<size_t>(sz)};
    }

    return CallWithErrorTranslation([&]{
        int res = NResource::Has(key);
        return PyBool_FromLong(res);
    });
}

}

const unsigned char res_importer_pyc[] = {
    #include "__res.pyc.inc"
};

int mod__res_exec(PyObject *mod) noexcept {
    PyObject* modules = PySys_GetObject("modules");
    Y_ASSERT(modules);
    Y_ASSERT(PyMapping_Check(modules));
    if (PyMapping_SetItemString(modules, "run_import_hook", mod) == -1) {
        return -1;
    }

    PyObject *bytecode = PyMarshal_ReadObjectFromString(
        reinterpret_cast<const char*>(res_importer_pyc),
        std::size(res_importer_pyc)
    );
    if (bytecode == NULL) {
        return -1;
    }

    // The code below which sets "__builtins__" is a workarownd for issue
    // reported here https://github.com/python/cpython/issues/130272 .
    // The problem can be seen for Y_PYTHON_SOURCE_ROOT mode when trying
    // compiling the code wich contains non-ascii identifiers. In this case
    // call to `compile` in get_code function raises the exception
    // KeyError: '__builtins__' inside `PyImport_Import` function.
    PyObject* builtinsKey = NULL;
    Y_DEFER {
        Py_DECREF(bytecode);
        Py_DECREF(builtinsKey);
    };
    PyObject* modns = PyModule_GetDict(mod);
    if (!modns) {
        return -1;
    }
    builtinsKey = PyUnicode_FromString("__builtins__");
    if (builtinsKey == NULL) {
        return -1;
    }
    int r = PyDict_Contains(modns, builtinsKey);
    if (r < 0) {
        return -1;
    } if (r == 0) {
        PyObject* builtins = PyEval_GetBuiltins();
        if (builtins == NULL) {
            return -1;
        }
        if (PyDict_SetItem(modns, builtinsKey, builtins) < 0) {
            return -1;
        }
    }

    if (PyObject* evalRes = PyEval_EvalCode(bytecode, modns, modns)) {
        Py_DECREF(evalRes);
    }
    if (PyErr_Occurred()) {
        return -1;
    }
    return 0;
}

PyDoc_STRVAR(mod__res_doc,
"resfs python bindings module with importer hook supporting hermetic python programs.");

PyMethodDef mod__res_methods[] = {
    {"count", _PyCFunction_CAST(NWrap::Count), METH_NOARGS, PyDoc_STR("Returns number of embedded resources.")},
    {"key_by_index", _PyCFunction_CAST(NWrap::KeyByIndex), METH_FASTCALL, PyDoc_STR("Returns resource key by resource index.")},
    {"find", _PyCFunction_CAST(NWrap::Find), METH_FASTCALL, PyDoc_STR("Finds resource content by key.")},
    {"has", _PyCFunction_CAST(NWrap::Has), METH_FASTCALL, PyDoc_STR("Checks if the resource with the given key exists.")},
    {nullptr, nullptr, 0, nullptr}
};

PyModuleDef_Slot mod__res_slots[] = {
    {Py_mod_exec, reinterpret_cast<void*>(&mod__res_exec)},
    {Py_mod_multiple_interpreters, Py_MOD_PER_INTERPRETER_GIL_SUPPORTED},
    {0, nullptr},
};

PyModuleDef mod__res = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "__res",
    .m_doc = mod__res_doc,
    .m_size = 0,
    .m_methods = mod__res_methods,
    .m_slots = mod__res_slots,
    .m_traverse = nullptr,
    .m_clear = nullptr,
    .m_free = nullptr
};

}

PyMODINIT_FUNC
PyInit___res() noexcept {
    return PyModuleDef_Init(&mod__res);
}
