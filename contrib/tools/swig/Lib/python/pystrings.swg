/* ------------------------------------------------------------
 *  utility methods for char strings 
 * ------------------------------------------------------------ */
%fragment("SWIG_AsCharPtrAndSize","header",fragment="SWIG_pchar_descriptor") {
/* Return string from Python obj. NOTE: obj must remain in scope in order
   to use the returned cptr (but only when alloc is set to SWIG_OLDOBJ) */
SWIGINTERN int
SWIG_AsCharPtrAndSize(PyObject *obj, char **cptr, size_t *psize, int *alloc)
{
%#if PY_VERSION_HEX>=0x03000000
%#if defined(SWIG_PYTHON_STRICT_BYTE_CHAR)
  if (PyBytes_Check(obj))
%#else
  if (PyUnicode_Check(obj))
%#endif
%#else  
  if (PyString_Check(obj))
%#endif
  {
    char *cstr; Py_ssize_t len;
    PyObject *bytes = NULL;
    int ret = SWIG_OK;
    if (alloc)
      *alloc = SWIG_OLDOBJ;
%#if PY_VERSION_HEX>=0x03000000 && defined(SWIG_PYTHON_STRICT_BYTE_CHAR)
    if (PyBytes_AsStringAndSize(obj, &cstr, &len) == -1)
      return SWIG_TypeError;
%#else
    cstr = (char *)SWIG_PyUnicode_AsUTF8AndSize(obj, &len, &bytes);
    if (!cstr)
      return SWIG_TypeError;
    /* The returned string is only duplicated if the char * returned is not owned and memory managed by obj */
    if (bytes && cptr) {
      if (alloc) {
        cstr = %new_copy_array(cstr, len + 1, char);
        *alloc = SWIG_NEWOBJ;
      } else {
        /* alloc must be set in order to clean up allocated memory */
        return SWIG_RuntimeError;
      }
    }
%#endif
    if (cptr) *cptr = cstr;
    if (psize) *psize = len + 1;
    Py_XDECREF(bytes);
    return ret;
  } else {
%#if defined(SWIG_PYTHON_2_UNICODE)
%#if defined(SWIG_PYTHON_STRICT_BYTE_CHAR)
%#error "Cannot use both SWIG_PYTHON_2_UNICODE and SWIG_PYTHON_STRICT_BYTE_CHAR at once"
%#endif
%#if PY_VERSION_HEX<0x03000000
    if (PyUnicode_Check(obj)) {
      char *cstr; Py_ssize_t len;
      if (!alloc && cptr) {
        return SWIG_RuntimeError;
      }
      obj = PyUnicode_AsUTF8String(obj);
      if (!obj)
        return SWIG_TypeError;
      if (PyString_AsStringAndSize(obj, &cstr, &len) != -1) {
        if (cptr) {
          if (alloc) *alloc = SWIG_NEWOBJ;
          *cptr = %new_copy_array(cstr, len + 1, char);
        }
        if (psize) *psize = len + 1;

        Py_XDECREF(obj);
        return SWIG_OK;
      } else {
        Py_XDECREF(obj);
      }
    }
%#endif
%#endif

    swig_type_info* pchar_descriptor = SWIG_pchar_descriptor();
    if (pchar_descriptor) {
      void* vptr = 0;
      if (SWIG_ConvertPtr(obj, &vptr, pchar_descriptor, 0) == SWIG_OK) {
	if (cptr) *cptr = (char *) vptr;
	if (psize) *psize = vptr ? (strlen((char *)vptr) + 1) : 0;
	if (alloc) *alloc = SWIG_OLDOBJ;
	return SWIG_OK;
      }
    }
  }
  return SWIG_TypeError;
}
}

%fragment("SWIG_FromCharPtrAndSize","header",fragment="SWIG_pchar_descriptor") {
SWIGINTERNINLINE PyObject *
SWIG_FromCharPtrAndSize(const char* carray, size_t size)
{
  if (carray) {
    if (size > INT_MAX) {
      swig_type_info* pchar_descriptor = SWIG_pchar_descriptor();
      return pchar_descriptor ? 
	SWIG_InternalNewPointerObj(%const_cast(carray,char *), pchar_descriptor, 0) : SWIG_Py_Void();
    } else {
%#if PY_VERSION_HEX >= 0x03000000
%#if defined(SWIG_PYTHON_STRICT_BYTE_CHAR)
      return PyBytes_FromStringAndSize(carray, %numeric_cast(size, Py_ssize_t));
%#else
      return PyUnicode_DecodeUTF8(carray, %numeric_cast(size, Py_ssize_t), "surrogateescape");
%#endif
%#else
      return PyString_FromStringAndSize(carray, %numeric_cast(size, Py_ssize_t));
%#endif
    }
  } else {
    return SWIG_Py_Void();
  }
}
}

