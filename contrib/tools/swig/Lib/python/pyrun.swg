/* -----------------------------------------------------------------------------
 * pyrun.swg
 *
 * This file contains the runtime support for Python modules
 * and includes code for managing global variables and pointer
 * type checking.
 *
 * ----------------------------------------------------------------------------- */

#if PY_VERSION_HEX < 0x02070000 /* 2.7.0 */
# error "This version of SWIG only supports Python >= 2.7"
#endif

#if PY_VERSION_HEX >= 0x03000000 && PY_VERSION_HEX < 0x03030000
# error "This version of SWIG only supports Python 3 >= 3.3"
#endif

/* Common SWIG API */

/* for raw pointers */
#define SWIG_Python_ConvertPtr(obj, pptr, type, flags)  SWIG_Python_ConvertPtrAndOwn(obj, pptr, type, flags, 0)
#define SWIG_ConvertPtr(obj, pptr, type, flags)         SWIG_Python_ConvertPtr(obj, pptr, type, flags)
#define SWIG_ConvertPtrAndOwn(obj,pptr,type,flags,own)  SWIG_Python_ConvertPtrAndOwn(obj, pptr, type, flags, own)

#ifdef SWIGPYTHON_BUILTIN
#define SWIG_NewPointerObj(ptr, type, flags)            SWIG_Python_NewPointerObj(self, ptr, type, flags)
#else
#define SWIG_NewPointerObj(ptr, type, flags)            SWIG_Python_NewPointerObj(NULL, ptr, type, flags)
#endif

#define SWIG_InternalNewPointerObj(ptr, type, flags)	SWIG_Python_NewPointerObj(NULL, ptr, type, flags)

#define SWIG_CheckImplicit(ty)                          SWIG_Python_CheckImplicit(ty) 
#define SWIG_AcquirePtr(ptr, src)                       SWIG_Python_AcquirePtr(ptr, src)
#define swig_owntype                                    int

/* for raw packed data */
#define SWIG_ConvertPacked(obj, ptr, sz, ty)            SWIG_Python_ConvertPacked(obj, ptr, sz, ty)
#define SWIG_NewPackedObj(ptr, sz, type)                SWIG_Python_NewPackedObj(ptr, sz, type)

/* for class or struct pointers */
#define SWIG_ConvertInstance(obj, pptr, type, flags)    SWIG_ConvertPtr(obj, pptr, type, flags)
#define SWIG_NewInstanceObj(ptr, type, flags)           SWIG_NewPointerObj(ptr, type, flags)

/* for C or C++ function pointers */
#define SWIG_ConvertFunctionPtr(obj, pptr, type)        SWIG_Python_ConvertFunctionPtr(obj, pptr, type)
#define SWIG_NewFunctionPtrObj(ptr, type)               SWIG_Python_NewPointerObj(NULL, ptr, type, 0)

/* for C++ member pointers, ie, member methods */
#define SWIG_ConvertMember(obj, ptr, sz, ty)            SWIG_Python_ConvertPacked(obj, ptr, sz, ty)
#define SWIG_NewMemberObj(ptr, sz, type)                SWIG_Python_NewPackedObj(ptr, sz, type)


/* Runtime API */

#define SWIG_GetModule(clientdata)                      SWIG_Python_GetModule(clientdata)
#define SWIG_SetModule(clientdata, pointer)             SWIG_Python_SetModule(pointer)
#define SWIG_NewClientData(obj)                         SwigPyClientData_New(obj)

#define SWIG_SetErrorObj                                SWIG_Python_SetErrorObj                            
#define SWIG_SetErrorMsg                        	SWIG_Python_SetErrorMsg				   
#define SWIG_ErrorType(code)                    	SWIG_Python_ErrorType(code)                        
#define SWIG_Error(code, msg)            		SWIG_Python_SetErrorMsg(SWIG_ErrorType(code), msg) 
#define SWIG_fail                        		goto fail					   


/* Runtime API implementation */

/* Error manipulation */

SWIGINTERN void 
SWIG_Python_SetErrorObj(PyObject *errtype, PyObject *obj) {
  SWIG_PYTHON_THREAD_BEGIN_BLOCK; 
  PyErr_SetObject(errtype, obj);
  Py_DECREF(obj);
  SWIG_PYTHON_THREAD_END_BLOCK;
}

SWIGINTERN void 
SWIG_Python_SetErrorMsg(PyObject *errtype, const char *msg) {
  SWIG_PYTHON_THREAD_BEGIN_BLOCK;
  PyErr_SetString(errtype, msg);
  SWIG_PYTHON_THREAD_END_BLOCK;
}

#define SWIG_Python_Raise(obj, type, desc)  SWIG_Python_SetErrorObj(SWIG_Python_ExceptionType(desc), obj)

/* Set a constant value */

#if defined(SWIGPYTHON_BUILTIN)

SWIGINTERN void
SwigPyBuiltin_AddPublicSymbol(PyObject *seq, const char *key) {
  PyObject *s = PyString_InternFromString(key);
  PyList_Append(seq, s);
  Py_DECREF(s);
}

SWIGINTERN void
SWIG_Python_SetConstant(PyObject *d, PyObject *public_interface, const char *name, PyObject *obj) {   
  PyDict_SetItemString(d, name, obj);
  Py_DECREF(obj);
  if (public_interface)
    SwigPyBuiltin_AddPublicSymbol(public_interface, name);
}

#else

SWIGINTERN void
SWIG_Python_SetConstant(PyObject *d, const char *name, PyObject *obj) {   
  PyDict_SetItemString(d, name, obj);
  Py_DECREF(obj);                            
}

#endif

/* Append a value to the result obj */

SWIGINTERN PyObject*
SWIG_Python_AppendOutput(PyObject* result, PyObject* obj) {
  if (!result) {
    result = obj;
  } else if (result == Py_None) {
    Py_DECREF(result);
    result = obj;
  } else {
    if (!PyList_Check(result)) {
      PyObject *o2 = result;
      result = PyList_New(1);
      if (result) {
        PyList_SET_ITEM(result, 0, o2);
      } else {
        Py_DECREF(obj);
        return o2;
      }
    }
    PyList_Append(result,obj);
    Py_DECREF(obj);
  }
  return result;
}

/* Unpack the argument tuple */

SWIGINTERN Py_ssize_t
SWIG_Python_UnpackTuple(PyObject *args, const char *name, Py_ssize_t min, Py_ssize_t max, PyObject **objs)
{
  if (!args) {
    if (!min && !max) {
      return 1;
    } else {
      PyErr_Format(PyExc_TypeError, "%s expected %s%d arguments, got none", 
		   name, (min == max ? "" : "at least "), (int)min);
      return 0;
    }
  }  
  if (!PyTuple_Check(args)) {
    if (min <= 1 && max >= 1) {
      Py_ssize_t i;
      objs[0] = args;
      for (i = 1; i < max; ++i) {
	objs[i] = 0;
      }
      return 2;
    }
    PyErr_SetString(PyExc_SystemError, "UnpackTuple() argument list is not a tuple");
    return 0;
  } else {
    Py_ssize_t l = PyTuple_GET_SIZE(args);
    if (l < min) {
      PyErr_Format(PyExc_TypeError, "%s expected %s%d arguments, got %d", 
		   name, (min == max ? "" : "at least "), (int)min, (int)l);
      return 0;
    } else if (l > max) {
      PyErr_Format(PyExc_TypeError, "%s expected %s%d arguments, got %d", 
		   name, (min == max ? "" : "at most "), (int)max, (int)l);
      return 0;
    } else {
      Py_ssize_t i;
      for (i = 0; i < l; ++i) {
	objs[i] = PyTuple_GET_ITEM(args, i);
      }
      for (; l < max; ++l) {
	objs[l] = 0;
      }
      return i + 1;
    }    
  }
}

SWIGINTERN int
SWIG_Python_CheckNoKeywords(PyObject *kwargs, const char *name) {
  int no_kwargs = 1;
  if (kwargs) {
    assert(PyDict_Check(kwargs));
    if (PyDict_Size(kwargs) > 0) {
      PyErr_Format(PyExc_TypeError, "%s() does not take keyword arguments", name);
      no_kwargs = 0;
    }
  }
  return no_kwargs;
}

/* A functor is a function object with one single object argument */
#define SWIG_Python_CallFunctor(functor, obj)	        PyObject_CallFunctionObjArgs(functor, obj, NULL);

/*
  Helper for static pointer initialization for both C and C++ code, for example
  static PyObject *SWIG_STATIC_POINTER(MyVar) = NewSomething(...);
*/
#ifdef __cplusplus
#define SWIG_STATIC_POINTER(var)  var
#else
#define SWIG_STATIC_POINTER(var)  var = 0; if (!var) var
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Python-specific SWIG API */
#define SWIG_newvarlink()                             SWIG_Python_newvarlink()
#define SWIG_addvarlink(p, name, get_attr, set_attr)  SWIG_Python_addvarlink(p, name, get_attr, set_attr)
#define SWIG_InstallConstants(d, constants)           SWIG_Python_InstallConstants(d, constants)
 
/* -----------------------------------------------------------------------------
 * global variable support code.
 * ----------------------------------------------------------------------------- */
 
typedef struct swig_globalvar {   
  char       *name;                  /* Name of global variable */
  PyObject *(*get_attr)(void);       /* Return the current value */
  int       (*set_attr)(PyObject *); /* Set the value */
  struct swig_globalvar *next;
} swig_globalvar;

typedef struct swig_varlinkobject {
  PyObject_HEAD
  swig_globalvar *vars;
} swig_varlinkobject;

SWIGINTERN PyObject *
swig_varlink_repr(PyObject *SWIGUNUSEDPARM(v)) {
#if PY_VERSION_HEX >= 0x03000000
  return PyUnicode_InternFromString("<Swig global variables>");
#else
  return PyString_FromString("<Swig global variables>");
#endif
}

SWIGINTERN PyObject *
swig_varlink_str(PyObject *o) {
  swig_varlinkobject *v = (swig_varlinkobject *) o;
#if PY_VERSION_HEX >= 0x03000000
  PyObject *str = PyUnicode_InternFromString("(");
  PyObject *tail;
  PyObject *joined;
  swig_globalvar *var;
  for (var = v->vars; var; var=var->next) {
    tail = PyUnicode_FromString(var->name);
    joined = PyUnicode_Concat(str, tail);
    Py_DecRef(str);
    Py_DecRef(tail);
    str = joined;
    if (var->next) {
        tail = PyUnicode_InternFromString(", ");
        joined = PyUnicode_Concat(str, tail);
        Py_DecRef(str);
        Py_DecRef(tail);
        str = joined;
    }
  }
  tail = PyUnicode_InternFromString(")");
  joined = PyUnicode_Concat(str, tail);
  Py_DecRef(str);
  Py_DecRef(tail);
  str = joined;
#else
  PyObject *str = PyString_FromString("(");
  swig_globalvar *var;
  for (var = v->vars; var; var=var->next) {
    PyString_ConcatAndDel(&str,PyString_FromString(var->name));
    if (var->next) PyString_ConcatAndDel(&str,PyString_FromString(", "));
  }
  PyString_ConcatAndDel(&str,PyString_FromString(")"));
#endif
  return str;
}

SWIGINTERN void
swig_varlink_dealloc(PyObject *o) {
  swig_varlinkobject *v = (swig_varlinkobject *) o;
  swig_globalvar *var = v->vars;
  while (var) {
    swig_globalvar *n = var->next;
    free(var->name);
    free(var);
    var = n;
  }
}

SWIGINTERN PyObject *
swig_varlink_getattr(PyObject *o, char *n) {
  swig_varlinkobject *v = (swig_varlinkobject *) o;
  PyObject *res = NULL;
  swig_globalvar *var = v->vars;
  while (var) {
    if (strcmp(var->name,n) == 0) {
      res = (*var->get_attr)();
      break;
    }
    var = var->next;
  }
  if (res == NULL && !PyErr_Occurred()) {
    PyErr_Format(PyExc_AttributeError, "Unknown C global variable '%s'", n);
  }
  return res;
}

SWIGINTERN int
swig_varlink_setattr(PyObject *o, char *n, PyObject *p) {
  swig_varlinkobject *v = (swig_varlinkobject *) o;
  int res = 1;
  swig_globalvar *var = v->vars;
  while (var) {
    if (strcmp(var->name,n) == 0) {
      res = (*var->set_attr)(p);
      break;
    }
    var = var->next;
  }
  if (res == 1 && !PyErr_Occurred()) {
    PyErr_Format(PyExc_AttributeError, "Unknown C global variable '%s'", n);
  }
  return res;
}

SWIGINTERN PyTypeObject*
swig_varlink_type(void) {
  static char varlink__doc__[] = "Swig var link object";
  static PyTypeObject varlink_type;
  static int type_init = 0;
  if (!type_init) {
    const PyTypeObject tmp = {
#if PY_VERSION_HEX >= 0x03000000
      PyVarObject_HEAD_INIT(NULL, 0)
#else
      PyObject_HEAD_INIT(NULL)
      0,                                  /* ob_size */
#endif
      "swigvarlink",                      /* tp_name */
      sizeof(swig_varlinkobject),         /* tp_basicsize */
      0,                                  /* tp_itemsize */
      (destructor) swig_varlink_dealloc,  /* tp_dealloc */
#if PY_VERSION_HEX < 0x030800b4
      (printfunc)0,                       /*tp_print*/
#else
      (Py_ssize_t)0,                      /*tp_vectorcall_offset*/
#endif
      (getattrfunc) swig_varlink_getattr, /* tp_getattr */
      (setattrfunc) swig_varlink_setattr, /* tp_setattr */
      0,                                  /* tp_compare */
      (reprfunc) swig_varlink_repr,       /* tp_repr */
      0,                                  /* tp_as_number */
      0,                                  /* tp_as_sequence */
      0,                                  /* tp_as_mapping */
      0,                                  /* tp_hash */
      0,                                  /* tp_call */
      (reprfunc) swig_varlink_str,        /* tp_str */
      0,                                  /* tp_getattro */
      0,                                  /* tp_setattro */
      0,                                  /* tp_as_buffer */
      0,                                  /* tp_flags */
      varlink__doc__,                     /* tp_doc */
      0,                                  /* tp_traverse */
      0,                                  /* tp_clear */
      0,                                  /* tp_richcompare */
      0,                                  /* tp_weaklistoffset */
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, /* tp_iter -> tp_weaklist */
      0,                                  /* tp_del */
      0,                                  /* tp_version_tag */
#if PY_VERSION_HEX >= 0x03040000
      0,                                  /* tp_finalize */
#endif
#if PY_VERSION_HEX >= 0x03080000
      0,                                  /* tp_vectorcall */
#endif
#if (PY_VERSION_HEX >= 0x03080000) && (PY_VERSION_HEX < 0x03090000)
      0,                                  /* tp_print */
#endif
#ifdef COUNT_ALLOCS
      0,                                  /* tp_allocs */
      0,                                  /* tp_frees */
      0,                                  /* tp_maxalloc */
      0,                                  /* tp_prev */
      0                                   /* tp_next */
#endif
    };
    varlink_type = tmp;
    type_init = 1;
    if (PyType_Ready(&varlink_type) < 0)
      return NULL;
  }
  return &varlink_type;
}

/* Create a variable linking object for use later */
SWIGINTERN PyObject *
SWIG_Python_newvarlink(void) {
  swig_varlinkobject *result = PyObject_NEW(swig_varlinkobject, swig_varlink_type());
  if (result) {
    result->vars = 0;
  }
  return ((PyObject*) result);
}

SWIGINTERN void 
SWIG_Python_addvarlink(PyObject *p, const char *name, PyObject *(*get_attr)(void), int (*set_attr)(PyObject *p)) {
  swig_varlinkobject *v = (swig_varlinkobject *) p;
  swig_globalvar *gv = (swig_globalvar *) malloc(sizeof(swig_globalvar));
  if (gv) {
    size_t size = strlen(name)+1;
    gv->name = (char *)malloc(size);
    if (gv->name) {
      memcpy(gv->name, name, size);
      gv->get_attr = get_attr;
      gv->set_attr = set_attr;
      gv->next = v->vars;
    }
  }
  v->vars = gv;
}


static PyObject *Swig_Globals_global = NULL;
  
SWIGINTERN PyObject *
SWIG_globals(void) {
  if (Swig_Globals_global == NULL) {
    Swig_Globals_global = SWIG_newvarlink();
  }
  return Swig_Globals_global;
}

#ifdef __cplusplus
}
#endif

/* -----------------------------------------------------------------------------
 * Pointer declarations
 * ----------------------------------------------------------------------------- */

/* Flags for new pointer objects */
#define SWIG_POINTER_NOSHADOW       (SWIG_POINTER_OWN      << 1)
#define SWIG_POINTER_NEW            (SWIG_POINTER_NOSHADOW | SWIG_POINTER_OWN)

#define SWIG_POINTER_IMPLICIT_CONV  (SWIG_POINTER_DISOWN   << 1)

#define SWIG_BUILTIN_TP_INIT	    (SWIG_POINTER_OWN << 2)
#define SWIG_BUILTIN_INIT	    (SWIG_BUILTIN_TP_INIT | SWIG_POINTER_OWN)

#ifdef __cplusplus
extern "C" {
#endif

/* The python void return value */

SWIGRUNTIMEINLINE PyObject * 
SWIG_Py_Void(void)
{
  PyObject *none = Py_None;
  Py_INCREF(none);
  return none;
}

/* SwigPyClientData */

typedef struct {
  PyObject *klass;
  PyObject *newraw;
  PyObject *newargs;
  PyObject *destroy;
  int delargs;
  int implicitconv;
  PyTypeObject *pytype;
} SwigPyClientData;

SWIGRUNTIMEINLINE int 
SWIG_Python_CheckImplicit(swig_type_info *ty)
{
  SwigPyClientData *data = (SwigPyClientData *)ty->clientdata;
  int fail = data ? data->implicitconv : 0;
  if (fail)
    PyErr_SetString(PyExc_TypeError, "Implicit conversion is prohibited for explicit constructors.");
  return fail;
}

SWIGRUNTIMEINLINE PyObject *
SWIG_Python_ExceptionType(swig_type_info *desc) {
  SwigPyClientData *data = desc ? (SwigPyClientData *) desc->clientdata : 0;
  PyObject *klass = data ? data->klass : 0;
  return (klass ? klass : PyExc_RuntimeError);
}


SWIGRUNTIME SwigPyClientData * 
SwigPyClientData_New(PyObject* obj)
{
  if (!obj) {
    return 0;
  } else {
    SwigPyClientData *data = (SwigPyClientData *)malloc(sizeof(SwigPyClientData));
    /* the klass element */
    data->klass = obj;
    Py_INCREF(data->klass);
    /* the newraw method and newargs arguments used to create a new raw instance */
    if (PyClass_Check(obj)) {
      data->newraw = 0;
      Py_INCREF(obj);
      data->newargs = obj;
    } else {
      data->newraw = PyObject_GetAttrString(data->klass, "__new__");
      if (data->newraw) {
        data->newargs = PyTuple_New(1);
        if (data->newargs) {
          Py_INCREF(obj);
          PyTuple_SET_ITEM(data->newargs, 0, obj);
        } else {
          Py_DECREF(data->newraw);
          Py_DECREF(data->klass);
          free(data);
          return 0;
        }
      } else {
        Py_INCREF(obj);
        data->newargs = obj;
      }
    }
    /* the destroy method, aka as the C++ delete method */
    data->destroy = PyObject_GetAttrString(data->klass, "__swig_destroy__");
    if (PyErr_Occurred()) {
      PyErr_Clear();
      data->destroy = 0;
    }
    if (data->destroy) {
      data->delargs = !(PyCFunction_GET_FLAGS(data->destroy) & METH_O);
    } else {
      data->delargs = 0;
    }
    data->implicitconv = 0;
    data->pytype = 0;
    return data;
  }
}

SWIGRUNTIME void 
SwigPyClientData_Del(SwigPyClientData *data)
{
  Py_XDECREF(data->klass);
  Py_XDECREF(data->newraw);
  Py_XDECREF(data->newargs);
  Py_XDECREF(data->destroy);
  free(data);
}

/* =============== SwigPyObject =====================*/

typedef struct {
  PyObject_HEAD
  void *ptr;
  swig_type_info *ty;
  int own;
  PyObject *next;
#ifdef SWIGPYTHON_BUILTIN
  PyObject *dict;
#endif
} SwigPyObject;


#ifdef SWIGPYTHON_BUILTIN

SWIGRUNTIME PyObject *
SwigPyObject_get___dict__(PyObject *v, PyObject *SWIGUNUSEDPARM(args))
{
  SwigPyObject *sobj = (SwigPyObject *)v;

  if (!sobj->dict)
    sobj->dict = PyDict_New();

  Py_XINCREF(sobj->dict);
  return sobj->dict;
}

#endif

SWIGRUNTIME PyObject *
SwigPyObject_long(SwigPyObject *v)
{
  return PyLong_FromVoidPtr(v->ptr);
}

SWIGRUNTIME PyObject *
SwigPyObject_format(const char* fmt, SwigPyObject *v)
{
  PyObject *res = NULL;
  PyObject *args = PyTuple_New(1);
  if (args) {
    PyObject *val = SwigPyObject_long(v);
    if (val) {
      PyObject *ofmt;
      PyTuple_SET_ITEM(args, 0, val);
      ofmt = SWIG_Python_str_FromChar(fmt);
      if (ofmt) {
#if PY_VERSION_HEX >= 0x03000000
        res = PyUnicode_Format(ofmt,args);
#else
        res = PyString_Format(ofmt,args);
#endif
        Py_DECREF(ofmt);
      }
    }
    Py_DECREF(args);
  }
  return res;
}

SWIGRUNTIME PyObject *
SwigPyObject_oct(SwigPyObject *v)
{
  return SwigPyObject_format("%o",v);
}

SWIGRUNTIME PyObject *
SwigPyObject_hex(SwigPyObject *v)
{
  return SwigPyObject_format("%x",v);
}

SWIGRUNTIME PyObject *
SwigPyObject_repr(SwigPyObject *v)
{
  const char *name = SWIG_TypePrettyName(v->ty);
  PyObject *repr = SWIG_Python_str_FromFormat("<Swig Object of type '%s' at %p>", (name ? name : "unknown"), (void *)v);
  if (repr && v->next) {
    PyObject *nrep = SwigPyObject_repr((SwigPyObject *)v->next);
    if (nrep) {
# if PY_VERSION_HEX >= 0x03000000
      PyObject *joined = PyUnicode_Concat(repr, nrep);
      Py_DecRef(repr);
      Py_DecRef(nrep);
      repr = joined;
# else
      PyString_ConcatAndDel(&repr,nrep);
# endif
    } else {
      Py_DecRef(repr);
      repr = NULL;
    }
  }
  return repr;
}

/* We need a version taking two PyObject* parameters so it's a valid
 * PyCFunction to use in swigobject_methods[]. */
SWIGRUNTIME PyObject *
SwigPyObject_repr2(PyObject *v, PyObject *SWIGUNUSEDPARM(args))
{
  return SwigPyObject_repr((SwigPyObject*)v);
}

SWIGRUNTIME int
SwigPyObject_compare(SwigPyObject *v, SwigPyObject *w)
{
  void *i = v->ptr;
  void *j = w->ptr;
  return (i < j) ? -1 : ((i > j) ? 1 : 0);
}

/* Added for Python 3.x, would it also be useful for Python 2.x? */
SWIGRUNTIME PyObject*
SwigPyObject_richcompare(SwigPyObject *v, SwigPyObject *w, int op)
{
  PyObject* res;
  if( op != Py_EQ && op != Py_NE ) {
    Py_INCREF(Py_NotImplemented);
    return Py_NotImplemented;
  }
  res = PyBool_FromLong( (SwigPyObject_compare(v, w)==0) == (op == Py_EQ) ? 1 : 0);
  return res;  
}


SWIGRUNTIME PyTypeObject* SwigPyObject_TypeOnce(void);

#ifdef SWIGPYTHON_BUILTIN
static swig_type_info *SwigPyObject_stype = 0;
SWIGRUNTIME PyTypeObject*
SwigPyObject_type(void) {
    SwigPyClientData *cd;
    assert(SwigPyObject_stype);
    cd = (SwigPyClientData*) SwigPyObject_stype->clientdata;
    assert(cd);
    assert(cd->pytype);
    return cd->pytype;
}
#else
SWIGRUNTIME PyTypeObject*
SwigPyObject_type(void) {
  static PyTypeObject *SWIG_STATIC_POINTER(type) = SwigPyObject_TypeOnce();
  return type;
}
#endif

SWIGRUNTIMEINLINE int
SwigPyObject_Check(PyObject *op) {
#ifdef SWIGPYTHON_BUILTIN
  PyTypeObject *target_tp = SwigPyObject_type();
  if (PyType_IsSubtype(op->ob_type, target_tp))
    return 1;
  return (strcmp(op->ob_type->tp_name, "SwigPyObject") == 0);
#else
  return (Py_TYPE(op) == SwigPyObject_type())
    || (strcmp(Py_TYPE(op)->tp_name,"SwigPyObject") == 0);
#endif
}

SWIGRUNTIME PyObject *
SwigPyObject_New(void *ptr, swig_type_info *ty, int own);

static PyObject* Swig_Capsule_global = NULL;

SWIGRUNTIME void
SwigPyObject_dealloc(PyObject *v)
{
  SwigPyObject *sobj = (SwigPyObject *) v;
  PyObject *next = sobj->next;
  if (sobj->own == SWIG_POINTER_OWN) {
    swig_type_info *ty = sobj->ty;
    SwigPyClientData *data = ty ? (SwigPyClientData *) ty->clientdata : 0;
    PyObject *destroy = data ? data->destroy : 0;
    if (destroy) {
      /* destroy is always a VARARGS method */
      PyObject *res;

      /* PyObject_CallFunction() has the potential to silently drop
         the active exception.  In cases of unnamed temporary
         variable or where we just finished iterating over a generator
         StopIteration will be active right now, and this needs to
         remain true upon return from SwigPyObject_dealloc.  So save
         and restore. */
      
      PyObject *type = NULL, *value = NULL, *traceback = NULL;
      PyErr_Fetch(&type, &value, &traceback);

      if (data->delargs) {
        /* we need to create a temporary object to carry the destroy operation */
        PyObject *tmp = SwigPyObject_New(sobj->ptr, ty, 0);
        if (tmp) {
          res = SWIG_Python_CallFunctor(destroy, tmp);
        } else {
          res = 0;
        }
        Py_XDECREF(tmp);
      } else {
        PyCFunction meth = PyCFunction_GET_FUNCTION(destroy);
        PyObject *mself = PyCFunction_GET_SELF(destroy);
        res = ((*meth)(mself, v));
      }
      if (!res)
        PyErr_WriteUnraisable(destroy);

      PyErr_Restore(type, value, traceback);

      Py_XDECREF(res);
    } 
#if !defined(SWIG_PYTHON_SILENT_MEMLEAK)
    else {
      const char *name = SWIG_TypePrettyName(ty);
      printf("swig/python detected a memory leak of type '%s', no destructor found.\n", (name ? name : "unknown"));
    }
#endif
    Py_XDECREF(Swig_Capsule_global);
  }
  Py_XDECREF(next);
#ifdef SWIGPYTHON_BUILTIN
  Py_XDECREF(sobj->dict);
#endif
  PyObject_DEL(v);
}

SWIGRUNTIME PyObject* 
SwigPyObject_append(PyObject* v, PyObject* next)
{
  SwigPyObject *sobj = (SwigPyObject *) v;
  if (!SwigPyObject_Check(next)) {
    PyErr_SetString(PyExc_TypeError, "Attempt to append a non SwigPyObject");
    return NULL;
  }
  ((SwigPyObject *)next)->next = sobj->next;
  sobj->next = next;
  Py_INCREF(next);
  return SWIG_Py_Void();
}

SWIGRUNTIME PyObject* 
SwigPyObject_next(PyObject* v, PyObject *SWIGUNUSEDPARM(args))
{
  SwigPyObject *sobj = (SwigPyObject *) v;
  if (sobj->next) {    
    Py_INCREF(sobj->next);
    return sobj->next;
  } else {
    return SWIG_Py_Void();
  }
}

SWIGINTERN PyObject*
SwigPyObject_disown(PyObject* v, PyObject *SWIGUNUSEDPARM(args))
{
  SwigPyObject *sobj = (SwigPyObject *)v;
  sobj->own = 0;
  return SWIG_Py_Void();
}

SWIGINTERN PyObject*
SwigPyObject_acquire(PyObject* v, PyObject *SWIGUNUSEDPARM(args))
{
  SwigPyObject *sobj = (SwigPyObject *)v;
  sobj->own = SWIG_POINTER_OWN;
  return SWIG_Py_Void();
}

SWIGINTERN PyObject*
SwigPyObject_own(PyObject *v, PyObject *args)
{
  PyObject *val = 0;
  if (!PyArg_UnpackTuple(args, "own", 0, 1, &val)) {
    return NULL;
  } else {
    SwigPyObject *sobj = (SwigPyObject *)v;
    PyObject *obj = PyBool_FromLong(sobj->own);
    if (val) {
      if (PyObject_IsTrue(val)) {
        Py_DECREF(SwigPyObject_acquire(v,args));
      } else {
        Py_DECREF(SwigPyObject_disown(v,args));
      }
    } 
    return obj;
  }
}

static PyMethodDef
swigobject_methods[] = {
  {"disown",  SwigPyObject_disown,  METH_NOARGS,  "releases ownership of the pointer"},
  {"acquire", SwigPyObject_acquire, METH_NOARGS,  "acquires ownership of the pointer"},
  {"own",     SwigPyObject_own,     METH_VARARGS, "returns/sets ownership of the pointer"},
  {"append",  SwigPyObject_append,  METH_O,       "appends another 'this' object"},
  {"next",    SwigPyObject_next,    METH_NOARGS,  "returns the next 'this' object"},
  {"__repr__",SwigPyObject_repr2,   METH_NOARGS,  "returns object representation"},
  {0, 0, 0, 0}  
};

SWIGRUNTIME PyTypeObject*
SwigPyObject_TypeOnce(void) {
  static char swigobject_doc[] = "Swig object carries a C/C++ instance pointer";

  static PyNumberMethods SwigPyObject_as_number = {
    (binaryfunc)0, /*nb_add*/
    (binaryfunc)0, /*nb_subtract*/
    (binaryfunc)0, /*nb_multiply*/
    /* nb_divide removed in Python 3 */
#if PY_VERSION_HEX < 0x03000000
    (binaryfunc)0, /*nb_divide*/
#endif
    (binaryfunc)0, /*nb_remainder*/
    (binaryfunc)0, /*nb_divmod*/
    (ternaryfunc)0,/*nb_power*/
    (unaryfunc)0,  /*nb_negative*/
    (unaryfunc)0,  /*nb_positive*/
    (unaryfunc)0,  /*nb_absolute*/
    (inquiry)0,    /*nb_nonzero*/
    0,		   /*nb_invert*/
    0,		   /*nb_lshift*/
    0,		   /*nb_rshift*/
    0,		   /*nb_and*/
    0,		   /*nb_xor*/
    0,		   /*nb_or*/
#if PY_VERSION_HEX < 0x03000000
    0,   /*nb_coerce*/
#endif
    (unaryfunc)SwigPyObject_long, /*nb_int*/
#if PY_VERSION_HEX < 0x03000000
    (unaryfunc)SwigPyObject_long, /*nb_long*/
#else
    0, /*nb_reserved*/
#endif
    (unaryfunc)0,                 /*nb_float*/
#if PY_VERSION_HEX < 0x03000000
    (unaryfunc)SwigPyObject_oct,  /*nb_oct*/
    (unaryfunc)SwigPyObject_hex,  /*nb_hex*/
#endif
#if PY_VERSION_HEX >= 0x03050000 /* 3.5 */
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 /* nb_inplace_add -> nb_inplace_matrix_multiply */
#elif PY_VERSION_HEX >= 0x03000000 /* 3.0 */
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 /* nb_inplace_add -> nb_index, nb_inplace_divide removed */
#else
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 /* nb_inplace_add -> nb_index */
#endif
  };

  static PyTypeObject swigpyobject_type;
  static int type_init = 0;
  if (!type_init) {
    const PyTypeObject tmp = {
#if PY_VERSION_HEX >= 0x03000000
      PyVarObject_HEAD_INIT(NULL, 0)
#else
      PyObject_HEAD_INIT(NULL)
      0,                                    /* ob_size */
#endif
      "SwigPyObject",                       /* tp_name */
      sizeof(SwigPyObject),                 /* tp_basicsize */
      0,                                    /* tp_itemsize */
      (destructor)SwigPyObject_dealloc,     /* tp_dealloc */
#if PY_VERSION_HEX < 0x030800b4
      (printfunc)0,                         /*tp_print*/
#else
      (Py_ssize_t)0,                        /*tp_vectorcall_offset*/
#endif
      (getattrfunc)0,                       /* tp_getattr */
      (setattrfunc)0,                       /* tp_setattr */
#if PY_VERSION_HEX >= 0x03000000
      0, /* tp_reserved in 3.0.1, tp_compare in 3.0.0 but not used */
#else
      (cmpfunc)SwigPyObject_compare,        /* tp_compare */
#endif
      (reprfunc)SwigPyObject_repr,          /* tp_repr */
      &SwigPyObject_as_number,              /* tp_as_number */
      0,                                    /* tp_as_sequence */
      0,                                    /* tp_as_mapping */
      (hashfunc)0,                          /* tp_hash */
      (ternaryfunc)0,                       /* tp_call */
      0,                                    /* tp_str */
      PyObject_GenericGetAttr,              /* tp_getattro */
      0,                                    /* tp_setattro */
      0,                                    /* tp_as_buffer */
      Py_TPFLAGS_DEFAULT,                   /* tp_flags */
      swigobject_doc,                       /* tp_doc */
      0,                                    /* tp_traverse */
      0,                                    /* tp_clear */
      (richcmpfunc)SwigPyObject_richcompare,/* tp_richcompare */
      0,                                    /* tp_weaklistoffset */
      0,                                    /* tp_iter */
      0,                                    /* tp_iternext */
      swigobject_methods,                   /* tp_methods */
      0,                                    /* tp_members */
      0,                                    /* tp_getset */
      0,                                    /* tp_base */
      0,                                    /* tp_dict */
      0,                                    /* tp_descr_get */
      0,                                    /* tp_descr_set */
      0,                                    /* tp_dictoffset */
      0,                                    /* tp_init */
      0,                                    /* tp_alloc */
      0,                                    /* tp_new */
      0,                                    /* tp_free */
      0,                                    /* tp_is_gc */
      0,                                    /* tp_bases */
      0,                                    /* tp_mro */
      0,                                    /* tp_cache */
      0,                                    /* tp_subclasses */
      0,                                    /* tp_weaklist */
      0,                                    /* tp_del */
      0,                                    /* tp_version_tag */
#if PY_VERSION_HEX >= 0x03040000
      0,                                    /* tp_finalize */
#endif
#if PY_VERSION_HEX >= 0x03080000
      0,                                    /* tp_vectorcall */
#endif
#if (PY_VERSION_HEX >= 0x03080000) && (PY_VERSION_HEX < 0x03090000)
      0,                                    /* tp_print */
#endif
#ifdef COUNT_ALLOCS
      0,                                    /* tp_allocs */
      0,                                    /* tp_frees */
      0,                                    /* tp_maxalloc */
      0,                                    /* tp_prev */
      0                                     /* tp_next */
#endif
    };
    swigpyobject_type = tmp;
    type_init = 1;
    if (PyType_Ready(&swigpyobject_type) != 0)
      return NULL;
  }
  return &swigpyobject_type;
}

SWIGRUNTIME PyObject *
SwigPyObject_New(void *ptr, swig_type_info *ty, int own)
{
  SwigPyObject *sobj = PyObject_NEW(SwigPyObject, SwigPyObject_type());
  if (sobj) {
    sobj->ptr  = ptr;
    sobj->ty   = ty;
    sobj->own  = own;
    sobj->next = 0;
#ifdef SWIGPYTHON_BUILTIN
    sobj->dict = 0;
#endif
    if (own == SWIG_POINTER_OWN) {
      /* Obtain a reference to the Python capsule wrapping the module information, so that the
       * module information is correctly destroyed after all SWIG python objects have been freed
       * by the GC (and corresponding destructors invoked) */
      Py_XINCREF(Swig_Capsule_global);
    }
  }
  return (PyObject *)sobj;
}

/* -----------------------------------------------------------------------------
 * Implements a simple Swig Packed type, and use it instead of string
 * ----------------------------------------------------------------------------- */

typedef struct {
  PyObject_HEAD
  void *pack;
  swig_type_info *ty;
  size_t size;
} SwigPyPacked;

SWIGRUNTIME PyObject *
SwigPyPacked_repr(SwigPyPacked *v)
{
  char result[SWIG_BUFFER_SIZE];
  if (SWIG_PackDataName(result, v->pack, v->size, 0, sizeof(result))) {
    return SWIG_Python_str_FromFormat("<Swig Packed at %s%s>", result, v->ty->name);
  } else {
    return SWIG_Python_str_FromFormat("<Swig Packed %s>", v->ty->name);
  }  
}

SWIGRUNTIME PyObject *
SwigPyPacked_str(SwigPyPacked *v)
{
  char result[SWIG_BUFFER_SIZE];
  if (SWIG_PackDataName(result, v->pack, v->size, 0, sizeof(result))){
    return SWIG_Python_str_FromFormat("%s%s", result, v->ty->name);
  } else {
    return SWIG_Python_str_FromChar(v->ty->name);
  }  
}

SWIGRUNTIME int
SwigPyPacked_compare(SwigPyPacked *v, SwigPyPacked *w)
{
  size_t i = v->size;
  size_t j = w->size;
  int s = (i < j) ? -1 : ((i > j) ? 1 : 0);
  return s ? s : strncmp((const char *)v->pack, (const char *)w->pack, 2*v->size);
}

SWIGRUNTIME PyTypeObject* SwigPyPacked_TypeOnce(void);

SWIGRUNTIME PyTypeObject*
SwigPyPacked_type(void) {
  static PyTypeObject *SWIG_STATIC_POINTER(type) = SwigPyPacked_TypeOnce();
  return type;
}

SWIGRUNTIMEINLINE int
SwigPyPacked_Check(PyObject *op) {
  return ((op)->ob_type == SwigPyPacked_TypeOnce()) 
    || (strcmp((op)->ob_type->tp_name,"SwigPyPacked") == 0);
}

SWIGRUNTIME void
SwigPyPacked_dealloc(PyObject *v)
{
  if (SwigPyPacked_Check(v)) {
    SwigPyPacked *sobj = (SwigPyPacked *) v;
    free(sobj->pack);
  }
  PyObject_DEL(v);
}

SWIGRUNTIME PyTypeObject*
SwigPyPacked_TypeOnce(void) {
  static char swigpacked_doc[] = "Swig object carries a C/C++ instance pointer";
  static PyTypeObject swigpypacked_type;
  static int type_init = 0;
  if (!type_init) {
    const PyTypeObject tmp = {
#if PY_VERSION_HEX>=0x03000000
      PyVarObject_HEAD_INIT(NULL, 0)
#else
      PyObject_HEAD_INIT(NULL)
      0,                                    /* ob_size */
#endif
      "SwigPyPacked",                       /* tp_name */
      sizeof(SwigPyPacked),                 /* tp_basicsize */
      0,                                    /* tp_itemsize */
      (destructor)SwigPyPacked_dealloc,     /* tp_dealloc */
#if PY_VERSION_HEX < 0x030800b4
      (printfunc)0,                         /*tp_print*/
#else
      (Py_ssize_t)0,                        /*tp_vectorcall_offset*/
#endif
      (getattrfunc)0,                       /* tp_getattr */
      (setattrfunc)0,                       /* tp_setattr */
#if PY_VERSION_HEX>=0x03000000
      0, /* tp_reserved in 3.0.1 */
#else
      (cmpfunc)SwigPyPacked_compare,        /* tp_compare */
#endif
      (reprfunc)SwigPyPacked_repr,          /* tp_repr */
      0,                                    /* tp_as_number */
      0,                                    /* tp_as_sequence */
      0,                                    /* tp_as_mapping */
      (hashfunc)0,                          /* tp_hash */
      (ternaryfunc)0,                       /* tp_call */
      (reprfunc)SwigPyPacked_str,           /* tp_str */
      PyObject_GenericGetAttr,              /* tp_getattro */
      0,                                    /* tp_setattro */
      0,                                    /* tp_as_buffer */
      Py_TPFLAGS_DEFAULT,                   /* tp_flags */
      swigpacked_doc,                       /* tp_doc */
      0,                                    /* tp_traverse */
      0,                                    /* tp_clear */
      0,                                    /* tp_richcompare */
      0,                                    /* tp_weaklistoffset */
      0,                                    /* tp_iter */
      0,                                    /* tp_iternext */
      0,                                    /* tp_methods */
      0,                                    /* tp_members */
      0,                                    /* tp_getset */
      0,                                    /* tp_base */
      0,                                    /* tp_dict */
      0,                                    /* tp_descr_get */
      0,                                    /* tp_descr_set */
      0,                                    /* tp_dictoffset */
      0,                                    /* tp_init */
      0,                                    /* tp_alloc */
      0,                                    /* tp_new */
      0,                                    /* tp_free */
      0,                                    /* tp_is_gc */
      0,                                    /* tp_bases */
      0,                                    /* tp_mro */
      0,                                    /* tp_cache */
      0,                                    /* tp_subclasses */
      0,                                    /* tp_weaklist */
      0,                                    /* tp_del */
      0,                                    /* tp_version_tag */
#if PY_VERSION_HEX >= 0x03040000
      0,                                    /* tp_finalize */
#endif
#if PY_VERSION_HEX >= 0x03080000
      0,                                    /* tp_vectorcall */
#endif
#if (PY_VERSION_HEX >= 0x03080000) && (PY_VERSION_HEX < 0x03090000)
      0,                                    /* tp_print */
#endif
#ifdef COUNT_ALLOCS
      0,                                    /* tp_allocs */
      0,                                    /* tp_frees */
      0,                                    /* tp_maxalloc */
      0,                                    /* tp_prev */
      0                                     /* tp_next */
#endif
    };
    swigpypacked_type = tmp;
    type_init = 1;
    if (PyType_Ready(&swigpypacked_type) != 0)
      return NULL;
  }
  return &swigpypacked_type;
}

SWIGRUNTIME PyObject *
SwigPyPacked_New(void *ptr, size_t size, swig_type_info *ty)
{
  SwigPyPacked *sobj = PyObject_NEW(SwigPyPacked, SwigPyPacked_type());
  if (sobj) {
    void *pack = malloc(size);
    if (pack) {
      memcpy(pack, ptr, size);
      sobj->pack = pack;
      sobj->ty   = ty;
      sobj->size = size;
    } else {
      PyObject_DEL((PyObject *) sobj);
      sobj = 0;
    }
  }
  return (PyObject *) sobj;
}

SWIGRUNTIME swig_type_info *
SwigPyPacked_UnpackData(PyObject *obj, void *ptr, size_t size)
{
  if (SwigPyPacked_Check(obj)) {
    SwigPyPacked *sobj = (SwigPyPacked *)obj;
    if (sobj->size != size) return 0;
    memcpy(ptr, sobj->pack, size);
    return sobj->ty;
  } else {
    return 0;
  }
}

/* -----------------------------------------------------------------------------
 * pointers/data manipulation
 * ----------------------------------------------------------------------------- */

static PyObject *Swig_This_global = NULL;

SWIGRUNTIME PyObject *
SWIG_This(void)
{
  if (Swig_This_global == NULL)
    Swig_This_global = SWIG_Python_str_FromChar("this");
  return Swig_This_global;
}

/* #define SWIG_PYTHON_SLOW_GETSET_THIS */

/* TODO: I don't know how to implement the fast getset in Python 3 right now */
#if PY_VERSION_HEX>=0x03000000
#define SWIG_PYTHON_SLOW_GETSET_THIS 
#endif

SWIGRUNTIME SwigPyObject *
SWIG_Python_GetSwigThis(PyObject *pyobj) 
{
  PyObject *obj;

  if (SwigPyObject_Check(pyobj))
    return (SwigPyObject *) pyobj;

#ifdef SWIGPYTHON_BUILTIN
  (void)obj;
# ifdef PyWeakref_CheckProxy
  if (PyWeakref_CheckProxy(pyobj)) {
    pyobj = PyWeakref_GET_OBJECT(pyobj);
    if (pyobj && SwigPyObject_Check(pyobj))
      return (SwigPyObject*) pyobj;
  }
# endif
  return NULL;
#else

  obj = 0;

#if !defined(SWIG_PYTHON_SLOW_GETSET_THIS)
  if (PyInstance_Check(pyobj)) {
    obj = _PyInstance_Lookup(pyobj, SWIG_This());      
  } else {
    PyObject **dictptr = _PyObject_GetDictPtr(pyobj);
    if (dictptr != NULL) {
      PyObject *dict = *dictptr;
      obj = dict ? PyDict_GetItem(dict, SWIG_This()) : 0;
    } else {
#ifdef PyWeakref_CheckProxy
      if (PyWeakref_CheckProxy(pyobj)) {
	PyObject *wobj = PyWeakref_GET_OBJECT(pyobj);
	return wobj ? SWIG_Python_GetSwigThis(wobj) : 0;
      }
#endif
      obj = PyObject_GetAttr(pyobj,SWIG_This());
      if (obj) {
	Py_DECREF(obj);
      } else {
	if (PyErr_Occurred()) PyErr_Clear();
	return 0;
      }
    }
  }
#else
  obj = PyObject_GetAttr(pyobj,SWIG_This());
  if (obj) {
    Py_DECREF(obj);
  } else {
    if (PyErr_Occurred()) PyErr_Clear();
    return 0;
  }
#endif
  if (obj && !SwigPyObject_Check(obj)) {
    /* a PyObject is called 'this', try to get the 'real this'
       SwigPyObject from it */ 
    return SWIG_Python_GetSwigThis(obj);
  }
  return (SwigPyObject *)obj;
#endif
}

/* Acquire a pointer value */

SWIGRUNTIME int
SWIG_Python_AcquirePtr(PyObject *obj, int own) {
  if (own == SWIG_POINTER_OWN) {
    SwigPyObject *sobj = SWIG_Python_GetSwigThis(obj);
    if (sobj) {
      int oldown = sobj->own;
      sobj->own = own;
      return oldown;
    }
  }
  return 0;
}

/* Convert a pointer value */

SWIGRUNTIME int
SWIG_Python_ConvertPtrAndOwn(PyObject *obj, void **ptr, swig_type_info *ty, int flags, int *own) {
  int res;
  SwigPyObject *sobj;
  int implicit_conv = (flags & SWIG_POINTER_IMPLICIT_CONV) != 0;

  if (!obj)
    return SWIG_ERROR;
  if (obj == Py_None && !implicit_conv) {
    if (ptr)
      *ptr = 0;
    return (flags & SWIG_POINTER_NO_NULL) ? SWIG_NullReferenceError : SWIG_OK;
  }

  res = SWIG_ERROR;

  sobj = SWIG_Python_GetSwigThis(obj);
  if (own)
    *own = 0;
  while (sobj) {
    void *vptr = sobj->ptr;
    if (ty) {
      swig_type_info *to = sobj->ty;
      if (to == ty) {
        /* no type cast needed */
        if (ptr) *ptr = vptr;
        break;
      } else {
        swig_cast_info *tc = SWIG_TypeCheck(to->name,ty);
        if (!tc) {
          sobj = (SwigPyObject *)sobj->next;
        } else {
          if (ptr) {
            int newmemory = 0;
            *ptr = SWIG_TypeCast(tc,vptr,&newmemory);
            if (newmemory == SWIG_CAST_NEW_MEMORY) {
              assert(own); /* badly formed typemap which will lead to a memory leak - it must set and use own to delete *ptr */
              if (own)
                *own = *own | SWIG_CAST_NEW_MEMORY;
            }
          }
          break;
        }
      }
    } else {
      if (ptr) *ptr = vptr;
      break;
    }
  }
  if (sobj) {
    if (((flags & SWIG_POINTER_RELEASE) == SWIG_POINTER_RELEASE) && !sobj->own) {
      res = SWIG_ERROR_RELEASE_NOT_OWNED;
    } else {
      if (own)
        *own = *own | sobj->own;
      if (flags & SWIG_POINTER_DISOWN) {
        sobj->own = 0;
      }
      if (flags & SWIG_POINTER_CLEAR) {
        sobj->ptr = 0;
      }
      res = SWIG_OK;
    }
  } else {
    if (implicit_conv) {
      SwigPyClientData *data = ty ? (SwigPyClientData *) ty->clientdata : 0;
      if (data && !data->implicitconv) {
        PyObject *klass = data->klass;
        if (klass) {
          PyObject *impconv;
          data->implicitconv = 1; /* avoid recursion and call 'explicit' constructors*/
          impconv = SWIG_Python_CallFunctor(klass, obj);
          data->implicitconv = 0;
          if (PyErr_Occurred()) {
            PyErr_Clear();
            impconv = 0;
          }
          if (impconv) {
            SwigPyObject *iobj = SWIG_Python_GetSwigThis(impconv);
            if (iobj) {
              void *vptr;
              res = SWIG_Python_ConvertPtrAndOwn((PyObject*)iobj, &vptr, ty, 0, 0);
              if (SWIG_IsOK(res)) {
                if (ptr) {
                  *ptr = vptr;
                  /* transfer the ownership to 'ptr' */
                  iobj->own = 0;
                  res = SWIG_AddCast(res);
                  res = SWIG_AddNewMask(res);
                } else {
                  res = SWIG_AddCast(res);		    
                }
              }
            }
            Py_DECREF(impconv);
          }
        }
      }
      if (!SWIG_IsOK(res) && obj == Py_None) {
        if (ptr)
          *ptr = 0;
        if (PyErr_Occurred())
          PyErr_Clear();
        res = SWIG_OK;
      }
    }
  }
  return res;
}

/* Convert a function ptr value */

SWIGRUNTIME int
SWIG_Python_ConvertFunctionPtr(PyObject *obj, void **ptr, swig_type_info *ty) {
  if (!PyCFunction_Check(obj)) {
    return SWIG_ConvertPtr(obj, ptr, ty, 0);
  } else {
    void *vptr = 0;
    swig_cast_info *tc;

    /* here we get the method pointer for callbacks */
    const char *doc = (((PyCFunctionObject *)obj) -> m_ml -> ml_doc);
    const char *desc = doc ? strstr(doc, "swig_ptr: ") : 0;
    if (desc)
      desc = ty ? SWIG_UnpackVoidPtr(desc + 10, &vptr, ty->name) : 0;
    if (!desc)
      return SWIG_ERROR;
    tc = SWIG_TypeCheck(desc,ty);
    if (tc) {
      int newmemory = 0;
      *ptr = SWIG_TypeCast(tc,vptr,&newmemory);
      assert(!newmemory); /* newmemory handling not yet implemented */
    } else {
      return SWIG_ERROR;
    }
    return SWIG_OK;
  }
}

/* Convert a packed pointer value */

SWIGRUNTIME int
SWIG_Python_ConvertPacked(PyObject *obj, void *ptr, size_t sz, swig_type_info *ty) {
  swig_type_info *to = SwigPyPacked_UnpackData(obj, ptr, sz);
  if (!to) return SWIG_ERROR;
  if (ty) {
    if (to != ty) {
      /* check type cast? */
      swig_cast_info *tc = SWIG_TypeCheck(to->name,ty);
      if (!tc) return SWIG_ERROR;
    }
  }
  return SWIG_OK;
}  

/* -----------------------------------------------------------------------------
 * Create a new pointer object
 * ----------------------------------------------------------------------------- */

/*
  Create a new instance object, without calling __init__, and set the
  'this' attribute.
*/

SWIGRUNTIME PyObject* 
SWIG_Python_NewShadowInstance(SwigPyClientData *data, PyObject *swig_this)
{
  PyObject *inst = 0;
  PyObject *newraw = data->newraw;
  if (newraw) {
    inst = PyObject_Call(newraw, data->newargs, NULL);
    if (inst) {
#if !defined(SWIG_PYTHON_SLOW_GETSET_THIS)
      PyObject **dictptr = _PyObject_GetDictPtr(inst);
      if (dictptr != NULL) {
        PyObject *dict = *dictptr;
        if (dict == NULL) {
          dict = PyDict_New();
          *dictptr = dict;
        }
        if (dict) {
          PyDict_SetItem(dict, SWIG_This(), swig_this);
        } else{
          Py_DECREF(inst);
          inst = 0;
        }
      }
#else
      if (PyObject_SetAttr(inst, SWIG_This(), swig_this) == -1) {
        Py_DECREF(inst);
        inst = 0;
      }
#endif
    }
  } else {
#if PY_VERSION_HEX >= 0x03000000
    PyObject *empty_args = PyTuple_New(0);
    if (empty_args) {
      PyObject *empty_kwargs = PyDict_New();
      if (empty_kwargs) {
        inst = ((PyTypeObject *)data->newargs)->tp_new((PyTypeObject *)data->newargs, empty_args, empty_kwargs);
        Py_DECREF(empty_kwargs);
        if (inst) {
          if (PyObject_SetAttr(inst, SWIG_This(), swig_this) == -1) {
            Py_DECREF(inst);
            inst = 0;
          } else {
            PyType_Modified(Py_TYPE(inst));
          }
        }
      }
      Py_DECREF(empty_args);
    }
#else
    PyObject *dict = PyDict_New();
    if (dict) {
      PyDict_SetItem(dict, SWIG_This(), swig_this);
      inst = PyInstance_NewRaw(data->newargs, dict);
      Py_DECREF(dict);
    }
#endif
  }
  return inst;
}

SWIGRUNTIME int
SWIG_Python_SetSwigThis(PyObject *inst, PyObject *swig_this)
{
#if !defined(SWIG_PYTHON_SLOW_GETSET_THIS)
  PyObject **dictptr = _PyObject_GetDictPtr(inst);
  if (dictptr != NULL) {
    PyObject *dict = *dictptr;
    if (dict == NULL) {
      dict = PyDict_New();
      *dictptr = dict;
    }
    if (dict) {
      return PyDict_SetItem(dict, SWIG_This(), swig_this);
    } else{
      return -1;
    }
  }
#endif
  return PyObject_SetAttr(inst, SWIG_This(), swig_this);
} 


SWIGINTERN PyObject *
SWIG_Python_InitShadowInstance(PyObject *args) {
  PyObject *obj[2];
  if (!SWIG_Python_UnpackTuple(args, "swiginit", 2, 2, obj)) {
    return NULL;
  } else {
    SwigPyObject *sthis = SWIG_Python_GetSwigThis(obj[0]);
    if (sthis) {
      Py_DECREF(SwigPyObject_append((PyObject*) sthis, obj[1]));
    } else {
      if (SWIG_Python_SetSwigThis(obj[0], obj[1]) != 0)
        return NULL;
    }
    return SWIG_Py_Void();
  }
}

/* Create a new pointer object */

SWIGRUNTIME PyObject *
SWIG_Python_NewPointerObj(PyObject *self, void *ptr, swig_type_info *type, int flags) {
  SwigPyClientData *clientdata;
  PyObject * robj;
  int own;

  if (!ptr)
    return SWIG_Py_Void();

  clientdata = type ? (SwigPyClientData *)(type->clientdata) : 0;
  own = (flags & SWIG_POINTER_OWN) ? SWIG_POINTER_OWN : 0;
  if (clientdata && clientdata->pytype) {
    SwigPyObject *newobj;
    if (flags & SWIG_BUILTIN_TP_INIT) {
      newobj = (SwigPyObject*) self;
      if (newobj->ptr) {
        PyObject *next_self = clientdata->pytype->tp_alloc(clientdata->pytype, 0);
        while (newobj->next)
	  newobj = (SwigPyObject *) newobj->next;
        newobj->next = next_self;
        newobj = (SwigPyObject *)next_self;
#ifdef SWIGPYTHON_BUILTIN
        newobj->dict = 0;
#endif
      }
    } else {
      newobj = PyObject_New(SwigPyObject, clientdata->pytype);
#ifdef SWIGPYTHON_BUILTIN
      if (newobj) {
        newobj->dict = 0;
      }
#endif
    }
    if (newobj) {
      newobj->ptr = ptr;
      newobj->ty = type;
      newobj->own = own;
      newobj->next = 0;
      return (PyObject*) newobj;
    }
    return SWIG_Py_Void();
  }

  assert(!(flags & SWIG_BUILTIN_TP_INIT));

  robj = SwigPyObject_New(ptr, type, own);
  if (robj && clientdata && !(flags & SWIG_POINTER_NOSHADOW)) {
    PyObject *inst = SWIG_Python_NewShadowInstance(clientdata, robj);
    Py_DECREF(robj);
    robj = inst;
  }
  return robj;
}

/* Create a new packed object */

SWIGRUNTIMEINLINE PyObject *
SWIG_Python_NewPackedObj(void *ptr, size_t sz, swig_type_info *type) {
  return ptr ? SwigPyPacked_New((void *) ptr, sz, type) : SWIG_Py_Void();
}

/* -----------------------------------------------------------------------------*
 *  Get type list 
 * -----------------------------------------------------------------------------*/

#ifdef SWIG_LINK_RUNTIME
void *SWIG_ReturnGlobalTypeList(void *);
#endif

static PyObject *Swig_TypeCache_global = NULL;

/* The python cached type query */
SWIGRUNTIME PyObject *
SWIG_Python_TypeCache(void) {
  if (Swig_TypeCache_global == NULL) {
    Swig_TypeCache_global = PyDict_New();
  }
  return Swig_TypeCache_global;
}

SWIGRUNTIME swig_module_info *
SWIG_Python_GetModule(void *SWIGUNUSEDPARM(clientdata)) {
#ifdef SWIG_LINK_RUNTIME
  static void *type_pointer = (void *)0;
  /* first check if module already created */
  if (!type_pointer) {
    type_pointer = SWIG_ReturnGlobalTypeList((void *)0);
  }
#else
  void *type_pointer = PyCapsule_Import(SWIGPY_CAPSULE_NAME, 0);
  if (PyErr_Occurred()) {
    PyErr_Clear();
    type_pointer = (void *)0;
  }
#endif
  return (swig_module_info *) type_pointer;
}


static int interpreter_counter = 0; // how many (sub-)interpreters are using swig_module's types

SWIGRUNTIME void
SWIG_Python_DestroyModule(PyObject *obj)
{
  swig_module_info *swig_module = (swig_module_info *) PyCapsule_GetPointer(obj, SWIGPY_CAPSULE_NAME);
  swig_type_info **types = swig_module->types;
  size_t i;
  if (--interpreter_counter != 0) // another sub-interpreter may still be using the swig_module's types
    return;
  for (i =0; i < swig_module->size; ++i) {
    swig_type_info *ty = types[i];
    if (ty->owndata) {
      SwigPyClientData *data = (SwigPyClientData *) ty->clientdata;
      ty->clientdata = 0;
      if (data) SwigPyClientData_Del(data);
    }
  }
  Py_DECREF(SWIG_This());
  Swig_This_global = NULL;
  Py_DECREF(SWIG_globals());
  Swig_Globals_global = NULL;
  Py_DECREF(SWIG_Python_TypeCache());
  Swig_TypeCache_global = NULL;
  Swig_Capsule_global = NULL;
}

SWIGRUNTIME void
SWIG_Python_SetModule(swig_module_info *swig_module) {
#if PY_VERSION_HEX >= 0x03000000
 /* Add a dummy module object into sys.modules */
  PyObject *module = PyImport_AddModule("swig_runtime_data" SWIG_RUNTIME_VERSION);
#else
  static PyMethodDef swig_empty_runtime_method_table[] = { {NULL, NULL, 0, NULL} }; /* Sentinel */
  PyObject *module = Py_InitModule("swig_runtime_data" SWIG_RUNTIME_VERSION, swig_empty_runtime_method_table);
#endif
  PyObject *pointer = PyCapsule_New((void *) swig_module, SWIGPY_CAPSULE_NAME, SWIG_Python_DestroyModule);
  if (pointer && module) {
    if (PyModule_AddObject(module, SWIGPY_CAPSULE_ATTR_NAME, pointer) == 0) {
      ++interpreter_counter;
      Swig_Capsule_global = pointer;
    } else {
      Py_DECREF(pointer);
    }
  } else {
    Py_XDECREF(pointer);
  }
}

SWIGRUNTIME swig_type_info *
SWIG_Python_TypeQuery(const char *type)
{
  PyObject *cache = SWIG_Python_TypeCache();
  PyObject *key = SWIG_Python_str_FromChar(type); 
  PyObject *obj = PyDict_GetItem(cache, key);
  swig_type_info *descriptor;
  if (obj) {
    descriptor = (swig_type_info *) PyCapsule_GetPointer(obj, NULL);
  } else {
    swig_module_info *swig_module = SWIG_GetModule(0);
    descriptor = SWIG_TypeQueryModule(swig_module, swig_module, type);
    if (descriptor) {
      obj = PyCapsule_New((void*) descriptor, NULL, NULL);
      if (obj) {
        PyDict_SetItem(cache, key, obj);
        Py_DECREF(obj);
      }
    }
  }
  Py_DECREF(key);
  return descriptor;
}

/* 
   For backward compatibility only
*/
#define SWIG_POINTER_EXCEPTION  0
#define SWIG_arg_fail(arg)      SWIG_Python_ArgFail(arg)
#define SWIG_MustGetPtr(p, type, argnum, flags)  SWIG_Python_MustGetPtr(p, type, argnum, flags)

SWIGRUNTIME int
SWIG_Python_AddErrMesg(const char* mesg, int infront)
{  
  if (PyErr_Occurred()) {
    PyObject *type = 0;
    PyObject *value = 0;
    PyObject *traceback = 0;
    PyErr_Fetch(&type, &value, &traceback);
    if (value) {
      PyObject *old_str = PyObject_Str(value);
      const char *tmp = SWIG_Python_str_AsChar(old_str);
      const char *errmesg = tmp ? tmp : "Invalid error message";
      Py_XINCREF(type);
      PyErr_Clear();
      if (infront) {
	PyErr_Format(type, "%s %s", mesg, errmesg);
      } else {
	PyErr_Format(type, "%s %s", errmesg, mesg);
      }
      Py_DECREF(old_str);
    }
    return 1;
  } else {
    return 0;
  }
}
  
SWIGRUNTIME int
SWIG_Python_ArgFail(int argnum)
{
  if (PyErr_Occurred()) {
    /* add information about failing argument */
    char mesg[256];
    PyOS_snprintf(mesg, sizeof(mesg), "argument number %d:", argnum);
    return SWIG_Python_AddErrMesg(mesg, 1);
  } else {
    return 0;
  }
}

SWIGRUNTIMEINLINE const char *
SwigPyObject_GetDesc(PyObject *self)
{
  SwigPyObject *v = (SwigPyObject *)self;
  swig_type_info *ty = v ? v->ty : 0;
  return ty ? ty->str : "";
}

SWIGRUNTIME void
SWIG_Python_TypeError(const char *type, PyObject *obj)
{
  if (type) {
#if defined(SWIG_COBJECT_TYPES)
    if (obj && SwigPyObject_Check(obj)) {
      const char *otype = (const char *) SwigPyObject_GetDesc(obj);
      if (otype) {
	PyErr_Format(PyExc_TypeError, "a '%s' is expected, 'SwigPyObject(%s)' is received",
		     type, otype);
	return;
      }
    } else 
#endif      
    {
      const char *otype = (obj ? obj->ob_type->tp_name : 0); 
      if (otype) {
	PyObject *str = PyObject_Str(obj);
	const char *cstr = str ? SWIG_Python_str_AsChar(str) : 0;
	if (cstr) {
	  PyErr_Format(PyExc_TypeError, "a '%s' is expected, '%s(%s)' is received",
		       type, otype, cstr);
	} else {
	  PyErr_Format(PyExc_TypeError, "a '%s' is expected, '%s' is received",
		       type, otype);
	}
	Py_XDECREF(str);
	return;
      }
    }   
    PyErr_Format(PyExc_TypeError, "a '%s' is expected", type);
  } else {
    PyErr_Format(PyExc_TypeError, "unexpected type is received");
  }
}


/* Convert a pointer value, signal an exception on a type mismatch */
SWIGRUNTIME void *
SWIG_Python_MustGetPtr(PyObject *obj, swig_type_info *ty, int SWIGUNUSEDPARM(argnum), int flags) {
  void *result;
  if (SWIG_Python_ConvertPtr(obj, &result, ty, flags) == -1) {
    PyErr_Clear();
  }
  return result;
}

#ifdef SWIGPYTHON_BUILTIN
SWIGRUNTIME int
SWIG_Python_NonDynamicSetAttr(PyObject *obj, PyObject *name, PyObject *value) {
  PyTypeObject *tp = obj->ob_type;
  PyObject *descr;
  PyObject *encoded_name;
  descrsetfunc f;
  int res = -1;

# ifdef Py_USING_UNICODE
  if (PyString_Check(name)) {
    name = PyUnicode_Decode(PyString_AsString(name), PyString_Size(name), NULL, NULL);
    if (!name)
      return -1;
  } else if (!PyUnicode_Check(name))
# else
  if (!PyString_Check(name))
# endif
  {
    PyErr_Format(PyExc_TypeError, "attribute name must be string, not '%.200s'", name->ob_type->tp_name);
    return -1;
  } else {
    Py_INCREF(name);
  }

  if (!tp->tp_dict) {
    if (PyType_Ready(tp) != 0)
      goto done;
  }

  descr = _PyType_Lookup(tp, name);
  f = NULL;
  if (descr != NULL)
    f = descr->ob_type->tp_descr_set;
  if (!f) {
    if (PyString_Check(name)) {
      encoded_name = name;
      Py_INCREF(name);
    } else {
      encoded_name = PyUnicode_AsUTF8String(name);
      if (!encoded_name)
        goto done;
    }
    PyErr_Format(PyExc_AttributeError, "'%.100s' object has no attribute '%.200s'", tp->tp_name, PyString_AsString(encoded_name));
    Py_DECREF(encoded_name);
  } else {
    res = f(descr, obj, value);
  }
  
  done:
  Py_DECREF(name);
  return res;
}
#endif


#ifdef __cplusplus
}
#endif
