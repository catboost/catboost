%module langs

%typemap(in) (const TStringBuf& name) (TStringBuf temp) {
    const char* chars;
    Py_ssize_t size;

%#if PY_MAJOR_VERSION >= 3
    if (PyUnicode_Check($input)) {
        chars = PyUnicode_AsUTF8($input);
        size = PyUnicode_GET_LENGTH($input);
    } else if (PyBytes_Check($input)) {
        chars = PyBytes_AsString($input);
        size = PyBytes_Size($input);
    } else {
        chars = NULL;
    }
%#else
    chars = PyString_AsString($input);
    size = PyString_Size($input);
%#endif

    if (chars == NULL) {
        PyErr_SetString(PyExc_ValueError, "Expected Python string.");
        return NULL;
    }
    temp = TStringBuf(chars, (size_t) size);
    $1 = &temp;
}

%typemap(in) wchar32 {
    $1 = (wchar32) PyLong_AsLong($input);
}


%begin %{
    #include <library/langs/langs.h>
%}

%include <library/langs/langs.h>
%include <library/langs/scripts.h>

