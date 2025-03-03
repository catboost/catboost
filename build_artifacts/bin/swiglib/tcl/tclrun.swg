/* -----------------------------------------------------------------------------
 * tclrun.swg
 *
 * This file contains the runtime support for Tcl modules and includes
 * code for managing global variables and pointer type checking.
 * ----------------------------------------------------------------------------- */

/* Common SWIG API */

/* for raw pointers */
#define SWIG_ConvertPtr(oc, ptr, ty, flags)             SWIG_Tcl_ConvertPtr(interp, oc, ptr, ty, flags)
#define SWIG_NewPointerObj(ptr, type, flags)            SWIG_Tcl_NewPointerObj(ptr, type, flags)

/* for raw packed data */
#define SWIG_ConvertPacked(obj, ptr, sz, ty)            SWIG_Tcl_ConvertPacked(interp, obj, ptr, sz, ty)
#define SWIG_NewPackedObj(ptr, sz, type)                SWIG_Tcl_NewPackedObj(ptr, sz, type)

/* for class or struct pointers */
#define SWIG_ConvertInstance(obj, pptr, type, flags)    SWIG_Tcl_ConvertPtr(interp, obj, pptr, type, flags)
#define SWIG_NewInstanceObj(thisvalue, type, flags)     SWIG_Tcl_NewInstanceObj(interp, thisvalue, type, flags)

/* for C or C++ function pointers */
#define SWIG_ConvertFunctionPtr(obj, pptr, type)        SWIG_Tcl_ConvertPtr(interp, obj, pptr, type, 0)
#define SWIG_NewFunctionPtrObj(ptr, type)               SWIG_Tcl_NewPointerObj(ptr, type, 0)

/* for C++ member pointers, ie, member methods */
#define SWIG_ConvertMember(obj, ptr, sz, ty)            SWIG_Tcl_ConvertPacked(interp,obj, ptr, sz, ty)
#define SWIG_NewMemberObj(ptr, sz, type)                SWIG_Tcl_NewPackedObj(ptr, sz, type)


/* Runtime API */

#define SWIG_GetModule(clientdata)                      SWIG_Tcl_GetModule((Tcl_Interp *) (clientdata))	     
#define SWIG_SetModule(clientdata, pointer)          	SWIG_Tcl_SetModule((Tcl_Interp *) (clientdata), pointer)


/* Error manipulation */

#define SWIG_ErrorType(code)                            SWIG_Tcl_ErrorType(code)                                      
#define SWIG_Error(code, msg)            		SWIG_Tcl_SetErrorMsg(interp, SWIG_Tcl_ErrorType(code), msg)
#define SWIG_fail                        		goto fail						    


/* Tcl-specific SWIG API */

#define SWIG_Acquire(ptr)                               SWIG_Tcl_Acquire(ptr)                                     
#define SWIG_MethodCommand                           	SWIG_Tcl_MethodCommand				       
#define SWIG_Disown(ptr)                             	SWIG_Tcl_Disown(ptr)				       
#define SWIG_ConvertPtrFromString(c, ptr, ty, flags) 	SWIG_Tcl_ConvertPtrFromString(interp, c, ptr, ty, flags)  
#define SWIG_MakePtr(c, ptr, ty, flags)              	SWIG_Tcl_MakePtr(c, ptr, ty, flags)		       
#define SWIG_PointerTypeFromString(c)                	SWIG_Tcl_PointerTypeFromString(c)			       
#define SWIG_GetArgs                                 	SWIG_Tcl_GetArgs					       
#define SWIG_GetConstantObj(key)                     	SWIG_Tcl_GetConstantObj(key)			       
#define SWIG_ObjectConstructor                       	SWIG_Tcl_ObjectConstructor				       
#define SWIG_Thisown(ptr)                            	SWIG_Tcl_Thisown(ptr)				       
#define SWIG_ObjectDelete                            	SWIG_Tcl_ObjectDelete				       


#define SWIG_TCL_DECL_ARGS_2(arg1, arg2)                (Tcl_Interp *interp SWIGUNUSED, arg1, arg2)
#define SWIG_TCL_CALL_ARGS_2(arg1, arg2)                (interp, arg1, arg2)
/* -----------------------------------------------------------------------------
 * pointers/data manipulation
 * ----------------------------------------------------------------------------- */

/* For backward compatibility only */
#define SWIG_POINTER_EXCEPTION  0
#define SWIG_GetConstant        SWIG_GetConstantObj
#define SWIG_Tcl_GetConstant    SWIG_Tcl_GetConstantObj

#if TCL_MAJOR_VERSION >= 8 && TCL_MINOR_VERSION >= 5
#define SWIG_TCL_HASHTABLE_INIT {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
#else
#define SWIG_TCL_HASHTABLE_INIT {0}
#endif

#include "assert.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Object support */

SWIGRUNTIME Tcl_HashTable*
SWIG_Tcl_ObjectTable(void) {
  static Tcl_HashTable  swigobjectTable;
  static int            swigobjectTableinit = 0;
  if (!swigobjectTableinit) {
    Tcl_InitHashTable(&swigobjectTable, TCL_ONE_WORD_KEYS);
    swigobjectTableinit = 1;
  }
  return &swigobjectTable;
}

/* Acquire ownership of a pointer */
SWIGRUNTIME void
SWIG_Tcl_Acquire(void *ptr) {
  int newobj;
  Tcl_CreateHashEntry(SWIG_Tcl_ObjectTable(), (char *) ptr, &newobj);
}

SWIGRUNTIME int
SWIG_Tcl_Thisown(void *ptr) {
  if (Tcl_FindHashEntry(SWIG_Tcl_ObjectTable(), (char *) ptr)) {
    return 1;
  }
  return 0;
}

/* Disown a pointer.  Returns 1 if we owned it to begin with */
SWIGRUNTIME int
SWIG_Tcl_Disown(void *ptr) {
  Tcl_HashEntry *entryPtr = Tcl_FindHashEntry(SWIG_Tcl_ObjectTable(), (char *) ptr);
  if (entryPtr) {
    Tcl_DeleteHashEntry(entryPtr);
    return 1;
  }
  return 0;
}

/* Convert a pointer value */
SWIGRUNTIME int
SWIG_Tcl_ConvertPtrFromString(Tcl_Interp *interp, const char *c, void **ptr, swig_type_info *ty, int flags) {
  swig_cast_info *tc;
  /* Pointer values must start with leading underscore */
  while (*c != '_') {
    *ptr = (void *) 0;
    if (strcmp(c,"NULL") == 0)
      return (flags & SWIG_POINTER_NO_NULL) ? SWIG_NullReferenceError : SWIG_OK;

    /* Empty string: not a pointer */
    if (*c == 0) return SWIG_ERROR; 

    /* Hmmm. It could be an object name. */

    /* Check if this is a command at all. Prevents <c> cget -this         */
    /* from being called when c is not a command, firing the unknown proc */
    if (Tcl_VarEval(interp,"info commands ", c, (char *) NULL) == TCL_OK) {
      Tcl_Obj *result = Tcl_GetObjResult(interp);
      if (*(Tcl_GetStringFromObj(result, NULL)) == 0) {
        /* It's not a command, so it can't be a pointer */
        Tcl_ResetResult(interp);
        return SWIG_ERROR;
      }
    } else {
      /* This will only fail if the argument is multiple words. */
      /* Multiple words are also not commands.                  */
      Tcl_ResetResult(interp);
      return SWIG_ERROR;
    }

    /* Check if this is really a SWIG pointer */
    if (Tcl_VarEval(interp,c," cget -this", (char *) NULL) != TCL_OK) {
      Tcl_ResetResult(interp);
      return SWIG_ERROR;
    }

    c = Tcl_GetStringFromObj(Tcl_GetObjResult(interp), NULL);
  }

  c++;
  c = SWIG_UnpackData(c,ptr,sizeof(void *));
  if (ty) {
    tc = c ? SWIG_TypeCheck(c,ty) : 0;
    if (!tc) {
      return SWIG_ERROR;
    }
    if (flags & SWIG_POINTER_DISOWN) {
      SWIG_Disown((void *) *ptr);
    }
    {
      int newmemory = 0;
      *ptr = SWIG_TypeCast(tc,(void *) *ptr,&newmemory);
      assert(!newmemory); /* newmemory handling not yet implemented */
    }
  }
  return SWIG_OK;
}

/* Convert a pointer value */
SWIGRUNTIMEINLINE int
SWIG_Tcl_ConvertPtr(Tcl_Interp *interp, Tcl_Obj *oc, void **ptr, swig_type_info *ty, int flags) {
  return SWIG_Tcl_ConvertPtrFromString(interp, Tcl_GetStringFromObj(oc,NULL), ptr, ty, flags);
}

/* Convert a pointer value */
SWIGRUNTIME char *
SWIG_Tcl_PointerTypeFromString(char *c) {
  char d;
  /* Pointer values must start with leading underscore. NULL has no type */
  if (*c != '_') {
    return 0;
  }
  c++;
  /* Extract hex value from pointer */
  while ((d = *c)) {
    if (!(((d >= '0') && (d <= '9')) || ((d >= 'a') && (d <= 'f')))) break;
    c++;
  }
  return c;
}

/* Convert a packed pointer value */
SWIGRUNTIME int
SWIG_Tcl_ConvertPacked(Tcl_Interp *SWIGUNUSEDPARM(interp) , Tcl_Obj *obj, void *ptr, int sz, swig_type_info *ty) {
  swig_cast_info *tc;
  const char  *c;

  if (!obj) goto type_error;
  c = Tcl_GetStringFromObj(obj,NULL);
  /* Pointer values must start with leading underscore */
  if (*c != '_') goto type_error;
  c++;
  c = SWIG_UnpackData(c,ptr,sz);
  if (ty) {
    tc = SWIG_TypeCheck(c,ty);
    if (!tc) goto type_error;
  }
  return SWIG_OK;

 type_error:

  return SWIG_ERROR;
}


/* Take a pointer and convert it to a string */
SWIGRUNTIME void
SWIG_Tcl_MakePtr(char *c, void *ptr, swig_type_info *ty, int SWIGUNUSEDPARM(flags)) {
  if (ptr) {
    *(c++) = '_';
    c = SWIG_PackData(c,&ptr,sizeof(void *));
    strcpy(c,ty->name);
  } else {
    strcpy(c,"NULL");
  }
}

/* Create a new pointer object */
SWIGRUNTIMEINLINE Tcl_Obj *
SWIG_Tcl_NewPointerObj(void *ptr, swig_type_info *type, int flags) {
  Tcl_Obj *robj;
  char result[SWIG_BUFFER_SIZE];
  SWIG_MakePtr(result,ptr,type,flags);
  robj = Tcl_NewStringObj(result,-1);
  return robj;
}

SWIGRUNTIME Tcl_Obj *
SWIG_Tcl_NewPackedObj(void *ptr, int sz, swig_type_info *type) {
  char result[1024];
  char *r = result;
  if ((2*sz + 1 + strlen(type->name)) > 1000) return 0;
  *(r++) = '_';
  r = SWIG_PackData(r,ptr,sz);
  strcpy(r,type->name);
  return Tcl_NewStringObj(result,-1);
}

/* -----------------------------------------------------------------------------*
 *  Get type list 
 * -----------------------------------------------------------------------------*/

SWIGRUNTIME swig_module_info * 
SWIG_Tcl_GetModule(Tcl_Interp *interp) {
  const char *data;
  swig_module_info *ret = 0;
  
  /* first check if pointer already created */
  data = Tcl_GetVar(interp, (char *)"swig_runtime_data_type_pointer" SWIG_RUNTIME_VERSION SWIG_TYPE_TABLE_NAME, TCL_GLOBAL_ONLY);
  if (data) {
    SWIG_UnpackData(data, &ret, sizeof(swig_type_info **));
  }

  return ret;
}

SWIGRUNTIME void
SWIG_Tcl_SetModule(Tcl_Interp *interp, swig_module_info *module) {
  char buf[SWIG_BUFFER_SIZE];
  char *data;

  /* create a new pointer */
  data = SWIG_PackData(buf, &module, sizeof(swig_type_info **));
  *data = 0;
  Tcl_SetVar(interp, (char *)"swig_runtime_data_type_pointer" SWIG_RUNTIME_VERSION SWIG_TYPE_TABLE_NAME, buf, TCL_GLOBAL_ONLY);
}

/* -----------------------------------------------------------------------------*
 *  Object auxiliaries
 * -----------------------------------------------------------------------------*/


SWIGRUNTIME void
SWIG_Tcl_ObjectDelete(ClientData clientData) {
  swig_instance *si = (swig_instance *) clientData;
  if (!si) return;
  if (si->destroy && SWIG_Disown(si->thisvalue)) {
    if (si->classptr->destructor) {
      (si->classptr->destructor)(si->thisvalue);
    }
  }
  Tcl_DecrRefCount(si->thisptr);
  free(si);
}

/* Function to invoke object methods given an instance */
SWIGRUNTIME int
SWIG_Tcl_MethodCommand(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *CONST _objv[]) {
  char *method,   *attrname;
  swig_instance   *inst = (swig_instance *) clientData;
  swig_method     *meth;
  swig_attribute  *attr;
  Tcl_Obj         *oldarg;
  Tcl_Obj         **objv;
  int              rcode;
  swig_class      *cls;
  swig_class      *cls_stack[64];
  int              cls_stack_bi[64];
  int              cls_stack_top = 0;
  int              numconf = 2;
  int              bi;

  objv = (Tcl_Obj **) _objv;
  if (objc < 2) {
    Tcl_SetResult(interp, (char *) "wrong # args.", TCL_STATIC);
    return TCL_ERROR;
  }
  method = Tcl_GetStringFromObj(objv[1],NULL);
  if (strcmp(method,"-acquire") == 0) {
    inst->destroy = 1;
    SWIG_Acquire(inst->thisvalue);
    return TCL_OK;
  }
  if (strcmp(method,"-disown") == 0) {
    if (inst->destroy) {
      SWIG_Disown(inst->thisvalue);
    }
    inst->destroy = 0;
    return TCL_OK;
  }
  if (strcmp(method,"-delete") == 0) {
    Tcl_DeleteCommandFromToken(interp,inst->cmdtok);
    return TCL_OK;
  }
  cls_stack[cls_stack_top] = inst->classptr;
  cls_stack_bi[cls_stack_top] = -1;
  while (1) {
    Tcl_HashEntry* hashentry;
    bi = cls_stack_bi[cls_stack_top];
    cls = cls_stack[cls_stack_top];
    if (bi != -1) {
      if (!cls->bases[bi] && cls->base_names[bi]) {
        /* lookup and cache the base class */
	swig_type_info *info = SWIG_TypeQueryModule(cls->module, cls->module, cls->base_names[bi]);
	if (info) cls->bases[bi] = (swig_class *) info->clientdata;
      }
      cls = cls->bases[bi];
      if (cls) {
        cls_stack_bi[cls_stack_top]++;
        cls_stack_top++;
        cls_stack[cls_stack_top] = cls;
        cls_stack_bi[cls_stack_top] = -1;
        continue;
      }
    }
    if (!cls) {
      cls_stack_top--;
      if (cls_stack_top < 0) break;
      else continue;
    }
    cls_stack_bi[cls_stack_top]++;

    hashentry = Tcl_FindHashEntry(&(cls->hashtable), method);
    if (hashentry) {
        ClientData cd = Tcl_GetHashValue(hashentry);
        swig_wrapper method_wrapper = (swig_wrapper)cd;
        oldarg = objv[1];
        objv[1] = inst->thisptr;
        Tcl_IncrRefCount(inst->thisptr);
        rcode = (method_wrapper)(clientData,interp,objc,objv);
        objv[1] = oldarg;
        Tcl_DecrRefCount(inst->thisptr);
        return rcode;
    }
    /* Check class methods for a match */
    if (strcmp(method,"cget") == 0) {
      if (objc < 3) {
        Tcl_SetResult(interp, (char *) "wrong # args.", TCL_STATIC);
        return TCL_ERROR;
      }
      attrname = Tcl_GetStringFromObj(objv[2],NULL);
      attr = cls->attributes;
      while (attr && attr->name) {
        if ((strcmp(attr->name, attrname) == 0) && (attr->getmethod)) {
          oldarg = objv[1];
          objv[1] = inst->thisptr;
          Tcl_IncrRefCount(inst->thisptr);
          rcode = (*attr->getmethod)(clientData,interp,2, objv);
          objv[1] = oldarg;
          Tcl_DecrRefCount(inst->thisptr);
          return rcode;
        }
        attr++;
      }
      if (strcmp(attrname, "-this") == 0) {
        Tcl_SetObjResult(interp, Tcl_DuplicateObj(inst->thisptr));
        return TCL_OK;
      }
      if (strcmp(attrname, "-thisown") == 0) {
        if (SWIG_Thisown(inst->thisvalue)) {
          Tcl_SetResult(interp,(char*)"1",TCL_STATIC);
        } else {
          Tcl_SetResult(interp,(char*)"0",TCL_STATIC);
        }
        return TCL_OK;
      }
    } else if (strcmp(method, "configure") == 0) {
      int i;
      if (objc < 4) {
        Tcl_SetResult(interp, (char *) "wrong # args.", TCL_STATIC);
        return TCL_ERROR;
      }
      i = 2;
      while (i < objc) {
        attrname = Tcl_GetStringFromObj(objv[i],NULL);
        attr = cls->attributes;
        while (attr && attr->name) {
          if ((strcmp(attr->name, attrname) == 0) && (attr->setmethod)) {
            oldarg = objv[i];
            objv[i] = inst->thisptr;
            Tcl_IncrRefCount(inst->thisptr);
            rcode = (*attr->setmethod)(clientData,interp,3, &objv[i-1]);
            objv[i] = oldarg;
            Tcl_DecrRefCount(inst->thisptr);
            if (rcode != TCL_OK) return rcode;
            numconf += 2;
          }
          attr++;
        }
        i+=2;
      }
    }
  }
  if (strcmp(method,"configure") == 0) {
    if (numconf >= objc) {
      return TCL_OK;
    } else {
      Tcl_SetResult(interp,(char *) "Invalid attribute name.", TCL_STATIC);
      return TCL_ERROR;
    }
  }
  if (strcmp(method,"cget") == 0) {
    Tcl_SetResult(interp,(char *) "Invalid attribute name.", TCL_STATIC);
    return TCL_ERROR;
  }
  Tcl_SetResult(interp, (char *) "Invalid method. Must be one of: configure cget -acquire -disown -delete", TCL_STATIC);
  cls = inst->classptr;
  bi = 0;
  while (cls) {
    meth = cls->methods;
    while (meth && meth->name) {
      char *cr = (char *) Tcl_GetStringResult(interp);
      size_t meth_len = strlen(meth->name);
      char* where = strchr(cr,':');
      while(where) {
        where = strstr(where, meth->name);
        if(where) {
          if(where[-1] == ' ' && (where[meth_len] == ' ' || where[meth_len]==0)) {
            break;
          } else {
            where++;
          }
        }
      }

      if (!where)
        Tcl_AppendElement(interp, (char *) meth->name);
      meth++;
    }
    cls = inst->classptr->bases[bi++];
  }
  return TCL_ERROR;
}

/* This function takes the current result and turns it into an object command */
SWIGRUNTIME Tcl_Obj *
SWIG_Tcl_NewInstanceObj(Tcl_Interp *interp, void *thisvalue, swig_type_info *type, int flags) {
  Tcl_Obj *robj = SWIG_NewPointerObj(thisvalue, type,0);
  /* Check to see if this pointer belongs to a class or not */
  if (thisvalue && (type->clientdata) && (interp)) {
    Tcl_CmdInfo    ci;
    char          *name;
    name = Tcl_GetStringFromObj(robj,NULL);
    if (!Tcl_GetCommandInfo(interp,name, &ci) || (flags)) {
      swig_instance *newinst = (swig_instance *) malloc(sizeof(swig_instance));
      newinst->thisptr = Tcl_DuplicateObj(robj);
      Tcl_IncrRefCount(newinst->thisptr);
      newinst->thisvalue = thisvalue;
      newinst->classptr = (swig_class *) type->clientdata;
      newinst->destroy = flags;
      newinst->cmdtok = Tcl_CreateObjCommand(interp, Tcl_GetStringFromObj(robj,NULL), (swig_wrapper_func) SWIG_MethodCommand, (ClientData) newinst, (swig_delete_func) SWIG_ObjectDelete);
      if (flags) {
        SWIG_Acquire(thisvalue);
      }
    }
  }
  return robj;
}

/* Function to create objects */
SWIGRUNTIME int
SWIG_Tcl_ObjectConstructor(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *CONST objv[]) {
  Tcl_Obj          *newObj = 0;
  void             *thisvalue = 0;
  swig_instance   *newinst = 0;
  swig_class      *classptr = (swig_class *) clientData;
  swig_wrapper     cons = 0;
  char             *name = 0;
  int               firstarg = 0;
  int               thisarg = 0;
  int               destroy = 1;

  if (!classptr) {
    Tcl_SetResult(interp, (char *) "swig: internal runtime error. No class object defined.", TCL_STATIC);
    return TCL_ERROR;
  }
  cons = classptr->constructor;
  if (objc > 1) {
    char *s = Tcl_GetStringFromObj(objv[1],NULL);
    if (strcmp(s,"-this") == 0) {
      thisarg = 2;
      cons = 0;
    } else if (strcmp(s,"-args") == 0) {
      firstarg = 1;
    } else if (objc == 2) {
      firstarg = 1;
      name = s;
    } else if (objc >= 3) {
      char *s1;
      name = s;
      s1 = Tcl_GetStringFromObj(objv[2],NULL);
      if (strcmp(s1,"-this") == 0) {
	thisarg = 3;
	cons = 0;
      } else {
	firstarg = 1;
      }
    }
  }
  if (cons) {
    int result;
    result = (*cons)(0, interp, objc-firstarg, &objv[firstarg]);
    if (result != TCL_OK) {
      return result;
    }
    newObj = Tcl_DuplicateObj(Tcl_GetObjResult(interp));
    if (!name) name = Tcl_GetStringFromObj(newObj,NULL);
  } else if (thisarg > 0) {
    if (thisarg < objc) {
      destroy = 0;
      newObj = Tcl_DuplicateObj(objv[thisarg]);
      if (!name) name = Tcl_GetStringFromObj(newObj,NULL);
    } else {
      Tcl_SetResult(interp, (char *) "wrong # args.", TCL_STATIC);
      return TCL_ERROR;
    }
  } else {
    Tcl_SetResult(interp, (char *) "No constructor available.", TCL_STATIC);
    return TCL_ERROR;
  }
  if (SWIG_Tcl_ConvertPtr(interp,newObj, (void **) &thisvalue, *(classptr->type), 0) != SWIG_OK) {
    Tcl_DecrRefCount(newObj);
    return TCL_ERROR;
  }
  newinst = (swig_instance *) malloc(sizeof(swig_instance));
  newinst->thisptr = newObj;
  Tcl_IncrRefCount(newObj);
  newinst->thisvalue = thisvalue;
  newinst->classptr = classptr;
  newinst->destroy = destroy;
  if (destroy) {
    SWIG_Acquire(thisvalue);
  }
  newinst->cmdtok = Tcl_CreateObjCommand(interp,name, (swig_wrapper) SWIG_MethodCommand, (ClientData) newinst, (swig_delete_func) SWIG_ObjectDelete);
  return TCL_OK;
}

/* -----------------------------------------------------------------------------*
 *   Get arguments 
 * -----------------------------------------------------------------------------*/
SWIGRUNTIME int
SWIG_Tcl_GetArgs(Tcl_Interp *interp, int objc, Tcl_Obj *CONST objv[], const char *fmt, ...) {
  int        argno = 0, opt = 0;
  long       tempi;
  double     tempd;
  const char *c;
  va_list    ap;
  void      *vptr;
  Tcl_Obj   *obj = 0;
  swig_type_info *ty;

  va_start(ap,fmt);
  for (c = fmt; (*c && (*c != ':') && (*c != ';')); c++,argno++) {
    if (*c == '|') {
      opt = 1;
      c++;
    }
    if (argno >= (objc-1)) {
      if (!opt) {
        Tcl_SetResult(interp, (char *) "Wrong number of arguments ", TCL_STATIC);
        goto argerror;
      } else {
        va_end(ap);
        return TCL_OK;
      }
    }

    vptr = va_arg(ap,void *);
    if (vptr) {
      if (isupper(*c)) {
        obj = SWIG_Tcl_GetConstantObj(Tcl_GetStringFromObj(objv[argno+1],0));
        if (!obj) obj = objv[argno+1];
      } else {
        obj = objv[argno+1];
      }
      switch(*c) {
      case 'i': case 'I':
      case 'l': case 'L':
      case 'h': case 'H':
      case 'b': case 'B':
        if (Tcl_GetLongFromObj(interp,obj,&tempi) != TCL_OK) goto argerror;
        if ((*c == 'i') || (*c == 'I')) *((int *)vptr) = (int)tempi;
        else if ((*c == 'l') || (*c == 'L')) *((long *)vptr) = (long)tempi;
        else if ((*c == 'h') || (*c == 'H')) *((short*)vptr) = (short)tempi;
        else if ((*c == 'b') || (*c == 'B')) *((unsigned char *)vptr) = (unsigned char)tempi;
        break;
      case 'f': case 'F':
      case 'd': case 'D':
        if (Tcl_GetDoubleFromObj(interp,obj,&tempd) != TCL_OK) goto argerror;
        if ((*c == 'f') || (*c == 'F')) *((float *) vptr) = (float)tempd;
        else if ((*c == 'd') || (*c == 'D')) *((double*) vptr) = tempd;
        break;
      case 's': case 'S':
        if (*(c+1) == '#') {
          int *vlptr = (int *) va_arg(ap, void *);
          *((char **) vptr) = Tcl_GetStringFromObj(obj, vlptr);
          c++;
        } else {
          *((char **)vptr) = Tcl_GetStringFromObj(obj,NULL);
        }
        break;
      case 'c': case 'C':
        *((char *)vptr) = *(Tcl_GetStringFromObj(obj,NULL));
        break;
      case 'p': case 'P':
        ty = (swig_type_info *) va_arg(ap, void *);
        if (SWIG_Tcl_ConvertPtr(interp, obj, (void **) vptr, ty, 0) != SWIG_OK) goto argerror;
        break;
      case 'o': case 'O':
        *((Tcl_Obj **)vptr) = objv[argno+1];
        break;
      default:
        break;
      }
    }
  }

  if ((*c != ';') && ((objc-1) > argno)) {
    Tcl_SetResult(interp, (char *) "Wrong # args.", TCL_STATIC);
    goto argerror;
  }
  va_end(ap);
  return TCL_OK;

 argerror:
  {
    char temp[32];
    sprintf(temp,"%d", argno+1);
    c = strchr(fmt,':');
    if (!c) c = strchr(fmt,';');
    if (!c) c = (char *)"";
    Tcl_AppendResult(interp,c," argument ", temp, NULL);
    va_end(ap);
    return TCL_ERROR;
  }
}

#ifdef __cplusplus
}
#endif
