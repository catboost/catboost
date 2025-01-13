/* ------------------------------------------------------------
 * The start of the Python initialization function 
 * ------------------------------------------------------------ */

%insert(init) "swiginit.swg"

#if defined(SWIGPYTHON_BUILTIN)
%fragment("<stddef.h>"); // For offsetof
#endif

#if defined SWIGPYTHON_FASTPROXY && !defined SWIGPYTHON_BUILTIN

%insert(runtime) %{
#ifdef __cplusplus
extern "C" {
#endif

/* Method creation and docstring support functions */

SWIGINTERN PyMethodDef *SWIG_PythonGetProxyDoc(const char *name);
SWIGINTERN PyObject *SWIG_PyInstanceMethod_New(PyObject *SWIGUNUSEDPARM(self), PyObject *func);
SWIGINTERN PyObject *SWIG_PyStaticMethod_New(PyObject *SWIGUNUSEDPARM(self), PyObject *func);

#ifdef __cplusplus
}
#endif
%}

#endif

%init %{

#ifdef __cplusplus
extern "C" {
#endif

/* -----------------------------------------------------------------------------
 * constants/methods manipulation
 * ----------------------------------------------------------------------------- */

/* Install Constants */
SWIGINTERN void
SWIG_Python_InstallConstants(PyObject *d, swig_const_info constants[]) {
  PyObject *obj = 0;
  size_t i;
  for (i = 0; constants[i].type; ++i) {
    switch(constants[i].type) {
    case SWIG_PY_POINTER:
      obj = SWIG_InternalNewPointerObj(constants[i].pvalue, *(constants[i]).ptype,0);
      break;
    case SWIG_PY_BINARY:
      obj = SWIG_NewPackedObj(constants[i].pvalue, constants[i].lvalue, *(constants[i].ptype));
      break;
    default:
      obj = 0;
      break;
    }
    if (obj) {
      PyDict_SetItemString(d, constants[i].name, obj);
      SWIG_Py_DECREF(obj);
    }
  }
}

/* -----------------------------------------------------------------------------
 * Patch %callback methods' docstrings to hold the callback ptrs
 * -----------------------------------------------------------------------------*/

SWIGINTERN void
SWIG_Python_FixMethods(PyMethodDef *methods, const swig_const_info *const_table, swig_type_info **types, swig_type_info **types_initial) {
  size_t i;
  for (i = 0; methods[i].ml_name; ++i) {
    const char *c = methods[i].ml_doc;
    if (!c) continue;
    c = strstr(c, "swig_ptr: ");
    if (c) {
      int j;
      const swig_const_info *ci = 0;
      const char *name = c + 10;
      for (j = 0; const_table[j].type; ++j) {
	if (strncmp(const_table[j].name, name, 
		    strlen(const_table[j].name)) == 0) {
	  ci = &(const_table[j]);
	  break;
	}
      }
      if (ci) {
	void *ptr = (ci->type == SWIG_PY_POINTER) ? ci->pvalue : 0;
	if (ptr) {
	  size_t shift = (ci->ptype) - types;
	  swig_type_info *ty = types_initial[shift];
	  size_t ldoc = (c - methods[i].ml_doc);
	  size_t lptr = strlen(ty->name)+2*sizeof(void*)+2;
	  char *ndoc = (char*)malloc(ldoc + lptr + 10);
	  if (ndoc) {
	    char *buff = ndoc;
	    memcpy(buff, methods[i].ml_doc, ldoc);
	    buff += ldoc;
	    memcpy(buff, "swig_ptr: ", 10);
	    buff += 10;
	    SWIG_PackVoidPtr(buff, ptr, ty->name, lptr);
	    methods[i].ml_doc = ndoc;
	  }
	}
      }
    }
  }
} 

#ifdef __cplusplus
}
#endif

%}

#if defined SWIGPYTHON_FASTPROXY && !defined SWIGPYTHON_BUILTIN

%init %{

#ifdef __cplusplus
extern "C" {
#endif

/* -----------------------------------------------------------------------------
 * Method creation and docstring support functions
 * ----------------------------------------------------------------------------- */

/* -----------------------------------------------------------------------------
 * Function to find the method definition with the correct docstring for the
 * proxy module as opposed to the low-level API
 * ----------------------------------------------------------------------------- */

SWIGINTERN PyMethodDef *SWIG_PythonGetProxyDoc(const char *name) {
  /* Find the function in the modified method table */
  size_t offset = 0;
  int found = 0;
  while (SwigMethods_proxydocs[offset].ml_meth != NULL) {
    if (strcmp(SwigMethods_proxydocs[offset].ml_name, name) == 0) {
      found = 1;
      break;
    }
    offset++;
  }
  /* Use the copy with the modified docstring if available */
  return found ? &SwigMethods_proxydocs[offset] : NULL;
}

/* -----------------------------------------------------------------------------
 * Wrapper of PyInstanceMethod_New() used in Python 3
 * It is exported to the generated module, used for -fastproxy
 * ----------------------------------------------------------------------------- */

SWIGINTERN PyObject *SWIG_PyInstanceMethod_New(PyObject *SWIGUNUSEDPARM(self), PyObject *func) {
  if (PyCFunction_Check(func)) {
    PyCFunctionObject *funcobj = (PyCFunctionObject *)func;
    PyMethodDef *ml = SWIG_PythonGetProxyDoc(funcobj->m_ml->ml_name);
    if (ml)
      func = PyCFunction_NewEx(ml, funcobj->m_self, funcobj->m_module);
  }
#if PY_VERSION_HEX >= 0x03000000
  return PyInstanceMethod_New(func);
#else
  return PyMethod_New(func, NULL, NULL);
#endif
}

/* -----------------------------------------------------------------------------
 * Wrapper of PyStaticMethod_New()
 * It is exported to the generated module, used for -fastproxy
 * ----------------------------------------------------------------------------- */

SWIGINTERN PyObject *SWIG_PyStaticMethod_New(PyObject *SWIGUNUSEDPARM(self), PyObject *func) {
  if (PyCFunction_Check(func)) {
    PyCFunctionObject *funcobj = (PyCFunctionObject *)func;
    PyMethodDef *ml = SWIG_PythonGetProxyDoc(funcobj->m_ml->ml_name);
    if (ml)
      func = PyCFunction_NewEx(ml, funcobj->m_self, funcobj->m_module);
  }
  return PyStaticMethod_New(func);
}

#ifdef __cplusplus
}
#endif

%}

#endif

%init %{

/* -----------------------------------------------------------------------------*
 *  Partial Init method
 * -----------------------------------------------------------------------------*/

#ifdef __cplusplus
extern "C"
#endif

SWIGEXPORT 
#if PY_VERSION_HEX >= 0x03000000
  PyObject*
#else
  void
#endif
SWIG_init(void) {
  PyObject *m, *d, *md, *globals;

#if PY_VERSION_HEX >= 0x03000000
  static struct PyModuleDef SWIG_module = {
    PyModuleDef_HEAD_INIT,
    SWIG_name,
    NULL,
    -1,
    SwigMethods,
    NULL,
    NULL,
    NULL,
    NULL
  };
#endif

#if defined(SWIGPYTHON_BUILTIN)
  static SwigPyClientData SwigPyObject_clientdata = {0, 0, 0, 0, 0, 0, 0};
  static PyGetSetDef this_getset_def = {
    (char *)"this", &SwigPyBuiltin_ThisClosure, NULL, NULL, NULL
  };
  static SwigPyGetSet thisown_getset_closure = {
    SwigPyObject_own,
    SwigPyObject_own
  };
  static PyGetSetDef thisown_getset_def = {
    (char *)"thisown", SwigPyBuiltin_GetterClosure, SwigPyBuiltin_SetterClosure, NULL, &thisown_getset_closure
  };
  PyTypeObject *builtin_pytype;
  int builtin_base_count;
  swig_type_info *builtin_basetype;
  PyObject *tuple;
  PyGetSetDescrObject *static_getset;
  PyTypeObject *metatype;
  PyTypeObject *swigpyobject;
  SwigPyClientData *cd;
  PyObject *public_interface, *public_symbol;
  PyObject *this_descr;
  PyObject *thisown_descr;
  PyObject *self = 0;
  int i;

  (void)builtin_pytype;
  (void)builtin_base_count;
  (void)builtin_basetype;
  (void)tuple;
  (void)static_getset;
  (void)self;

  /* Metaclass is used to implement static member variables */
  metatype = SwigPyObjectType();
  assert(metatype);
#endif

  (void)globals;

  /* Create singletons now to avoid potential deadlocks with multi-threaded usage after module initialization */
  SWIG_This();
  SWIG_Python_TypeCache();
  SwigPyPacked_type();
#ifndef SWIGPYTHON_BUILTIN
  SwigPyObject_type();
#endif

  /* Fix SwigMethods to carry the callback ptrs when needed */
  SWIG_Python_FixMethods(SwigMethods, swig_const_table, swig_types, swig_type_initial);

#if PY_VERSION_HEX >= 0x03000000
  m = PyModule_Create(&SWIG_module);
#else
  m = Py_InitModule(SWIG_name, SwigMethods);
#endif

  md = d = PyModule_GetDict(m);
  (void)md;

  SWIG_InitializeModule(0);

#ifdef SWIGPYTHON_BUILTIN
  swigpyobject = SwigPyObject_TypeOnce();

  SwigPyObject_stype = SWIG_MangledTypeQuery("_p_SwigPyObject");
  assert(SwigPyObject_stype);
  cd = (SwigPyClientData*) SwigPyObject_stype->clientdata;
  if (!cd) {
    SwigPyObject_stype->clientdata = &SwigPyObject_clientdata;
    SwigPyObject_clientdata.pytype = swigpyobject;
  } else if (swigpyobject->tp_basicsize != cd->pytype->tp_basicsize) {
    PyErr_SetString(PyExc_RuntimeError, "Import error: attempted to load two incompatible swig-generated modules.");
# if PY_VERSION_HEX >= 0x03000000
    return NULL;
# else
    return;
# endif
  }

  /* All objects have a 'this' attribute */
  this_descr = PyDescr_NewGetSet(SwigPyObject_type(), &this_getset_def);
  (void)this_descr;

  /* All objects have a 'thisown' attribute */
  thisown_descr = PyDescr_NewGetSet(SwigPyObject_type(), &thisown_getset_def);
  (void)thisown_descr;

  public_interface = PyList_New(0);
  public_symbol = 0;
  (void)public_symbol;

  PyDict_SetItemString(md, "__all__", public_interface);
  SWIG_Py_DECREF(public_interface);
  for (i = 0; SwigMethods[i].ml_name != NULL; ++i)
    SwigPyBuiltin_AddPublicSymbol(public_interface, SwigMethods[i].ml_name);
  for (i = 0; swig_const_table[i].name != 0; ++i)
    SwigPyBuiltin_AddPublicSymbol(public_interface, swig_const_table[i].name);
#endif

  SWIG_InstallConstants(d,swig_const_table);
%}

