/* Compatibility macros for Python 3 */
#if PY_VERSION_HEX >= 0x03000000

#define PyClass_Check(obj) PyObject_IsInstance(obj, (PyObject *)&PyType_Type)
#define PyInt_Check(x) PyLong_Check(x)
#define PyInt_AsLong(x) PyLong_AsLong(x)
#define PyInt_FromLong(x) PyLong_FromLong(x)
#define PyInt_FromSize_t(x) PyLong_FromSize_t(x)
#define PyString_Check(name) PyBytes_Check(name)
#define PyString_FromString(x) PyUnicode_FromString(x)
#define PyString_Format(fmt, args)  PyUnicode_Format(fmt, args)
#define PyString_AsString(str) PyBytes_AsString(str)
#define PyString_Size(str) PyBytes_Size(str)	
#define PyString_InternFromString(key) PyUnicode_InternFromString(key)
#define Py_TPFLAGS_HAVE_CLASS Py_TPFLAGS_BASETYPE
#define _PyLong_FromSsize_t(x) PyLong_FromSsize_t(x)

#endif

/* SWIG APIs for compatibility of both Python 2 & 3 */

#if PY_VERSION_HEX >= 0x03000000
#  define SWIG_Python_str_FromFormat PyUnicode_FromFormat
#else
#  define SWIG_Python_str_FromFormat PyString_FromFormat
#endif


/* Wrapper around PyUnicode_AsUTF8AndSize - call Py_XDECREF on the returned pbytes when finished with the returned string */
SWIGINTERN const char *
SWIG_PyUnicode_AsUTF8AndSize(PyObject *str, Py_ssize_t *psize, PyObject **pbytes)
{
#if PY_VERSION_HEX >= 0x03030000
# if !defined(Py_LIMITED_API) || Py_LIMITED_API+0 >= 0x030A0000
  *pbytes = NULL;
  return PyUnicode_AsUTF8AndSize(str, psize);
# else
  const char *chars;
  *pbytes = PyUnicode_AsUTF8String(str);
  chars = *pbytes ? PyBytes_AsString(*pbytes) : NULL;
  if (chars && psize)
    *psize = PyBytes_Size(*pbytes);
  return chars;
# endif
#else
  char *chars = NULL;
  *pbytes = NULL;
  PyString_AsStringAndSize(str, &chars, psize);
  return chars;
#endif
}

SWIGINTERN PyObject*
SWIG_Python_str_FromChar(const char *c)
{
#if PY_VERSION_HEX >= 0x03000000
  return PyUnicode_FromString(c); 
#else
  return PyString_FromString(c);
#endif
}

/* SWIGPY_USE_CAPSULE is no longer used within SWIG itself, but some user interface files check for it. */
# define SWIGPY_USE_CAPSULE
#ifdef SWIGPYTHON_BUILTIN
# define SWIGPY_CAPSULE_ATTR_NAME "type_pointer_capsule_builtin" SWIG_TYPE_TABLE_NAME
#else
# define SWIGPY_CAPSULE_ATTR_NAME "type_pointer_capsule" SWIG_TYPE_TABLE_NAME
#endif
# define SWIGPY_CAPSULE_NAME ("swig_runtime_data" SWIG_RUNTIME_VERSION "." SWIGPY_CAPSULE_ATTR_NAME)

#if PY_VERSION_HEX < 0x03020000
#define PyDescr_TYPE(x) (((PyDescrObject *)(x))->d_type)
#define PyDescr_NAME(x) (((PyDescrObject *)(x))->d_name)
#define Py_hash_t long
#endif

#ifdef Py_LIMITED_API
# define PyTuple_GET_ITEM PyTuple_GetItem
/* Note that PyTuple_SetItem() has different semantics from PyTuple_SET_ITEM as it decref's the original tuple item, so in general they cannot be used
  interchangeably. However in SWIG-generated code PyTuple_SET_ITEM is only used with newly initialized tuples without any items and for them this does work. */
# define PyTuple_SET_ITEM PyTuple_SetItem
# define PyTuple_GET_SIZE PyTuple_Size
# define PyCFunction_GET_FLAGS PyCFunction_GetFlags
# define PyCFunction_GET_FUNCTION PyCFunction_GetFunction
# define PyCFunction_GET_SELF PyCFunction_GetSelf
# define PyList_GET_ITEM PyList_GetItem
# define PyList_SET_ITEM PyList_SetItem
# define PySliceObject PyObject
#endif

/* Increment and Decrement wrappers - for portability when using the stable abi and for performance otherwise */
#ifdef Py_LIMITED_API
# define SWIG_Py_INCREF Py_IncRef
# define SWIG_Py_XINCREF Py_IncRef
# define SWIG_Py_DECREF Py_DecRef
# define SWIG_Py_XDECREF Py_DecRef
#else
# define SWIG_Py_INCREF Py_INCREF
# define SWIG_Py_XINCREF Py_XINCREF
# define SWIG_Py_DECREF Py_DECREF
# define SWIG_Py_XDECREF Py_XDECREF
#endif
