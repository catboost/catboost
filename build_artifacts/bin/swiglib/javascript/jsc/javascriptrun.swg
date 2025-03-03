/* ----------------------------------------------------------------------------
 *  Errors and exceptions
 *
 * ---------------------------------------------------------------------------*/

#define SWIG_Error(code, msg)     SWIG_JSC_exception(context, exception, code, msg)
#define SWIG_exception(code, msg) do { SWIG_JSC_exception(context, exception, code, msg); SWIG_fail; } while (0)
#define SWIG_fail                 goto fail

SWIGRUNTIME void SWIG_Javascript_Raise(JSContextRef context, JSValueRef *exception, const char* type) {
  JSStringRef message = JSStringCreateWithUTF8CString(type);
  JSValueRef error_arguments[1]; 
  JSObjectRef exception_object;
  JSValueRef exception_value;
  exception_value = JSValueMakeString(context, message);
  /* Converting the result to an object will let JavascriptCore add 
     "sourceURL" (file) and "line" (number) and "message" to the exception,
     instead of just returning a raw string. This is extremely important for debugging your errors.
     Using JSObjectMakeError is better than JSValueToObject because the latter only populates
     "sourceURL" and "line", but not "message" or any others I don't know about.
   */
  error_arguments[0] = exception_value;
  exception_object = JSObjectMakeError(context, 1, error_arguments, NULL);

  /* Return the exception_object */
  *exception = exception_object;

  JSStringRelease(message);
}

SWIGRUNTIME void SWIG_JSC_exception(JSContextRef context, JSValueRef *exception, int code, const char* msg) {
  SWIG_Javascript_Raise(context, exception, msg);
}

/* ----------------------------------------------------------------------------
 *  The parent class of all Proxies
 *
 * ---------------------------------------------------------------------------*/

typedef struct {
  bool swigCMemOwn;
  void *swigCObject;
  swig_type_info *info;
} SwigPrivData;

SWIGRUNTIME JSValueRef _wrap_SwigObject_disown(JSContextRef context, JSObjectRef function, JSObjectRef thisObject, size_t argc, const JSValueRef argv[], JSValueRef* exception)
{
  JSValueRef jsresult;

  JSObjectRef obj = JSValueToObject(context, thisObject, NULL);
  SwigPrivData *cdata = (SwigPrivData *) JSObjectGetPrivate(obj);

  cdata->swigCMemOwn = false;

  jsresult = JSValueMakeUndefined(context);
  return jsresult;
}

SWIGRUNTIME JSValueRef _wrap_SwigObject_getCPtr(JSContextRef context, JSObjectRef function, JSObjectRef thisObject, size_t argc, const JSValueRef argv[], JSValueRef* exception)
{
  JSValueRef jsresult;
  long result;

  JSObjectRef obj = JSValueToObject(context, thisObject, NULL);
  SwigPrivData *cdata = (SwigPrivData*) JSObjectGetPrivate(obj);

  result = (long) cdata->swigCObject;
  jsresult = JSValueMakeNumber(context, result);

  return jsresult;
}

SWIGRUNTIME JSValueRef _wrap_SwigObject_equals(JSContextRef context, JSObjectRef function, JSObjectRef thisObject, size_t argc, const JSValueRef argv[], JSValueRef* exception)
{
  JSValueRef jsresult;
  bool result;

  JSObjectRef obj = JSValueToObject(context, thisObject, NULL);
  SwigPrivData *cdata = (SwigPrivData*) JSObjectGetPrivate(obj);

  JSObjectRef obj2 = JSValueToObject(context, argv[0], NULL);
  SwigPrivData *cdata2 = (SwigPrivData*) JSObjectGetPrivate(obj2);

  result = (cdata->swigCObject == cdata2->swigCObject);
  jsresult = JSValueMakeBoolean(context, result);

  return jsresult;
}

SWIGRUNTIME JSStaticValue _SwigObject_values[] = {
  {
    0, 0, 0, 0
  }
};

SWIGRUNTIME JSStaticFunction _SwigObject_functions[] = {
  {
    "disown",_wrap_SwigObject_disown, kJSPropertyAttributeNone
  },
  {
    "equals",_wrap_SwigObject_equals, kJSPropertyAttributeNone
  },
  {
    "getCPtr",_wrap_SwigObject_getCPtr, kJSPropertyAttributeNone
  },
  {
    0, 0, 0
  }
};

SWIGRUNTIME JSClassDefinition _SwigObject_objectDefinition;

SWIGRUNTIME JSClassRef _SwigObject_classRef;


SWIGRUNTIME int SWIG_JSC_ConvertInstancePtr(JSContextRef context, JSObjectRef objRef, void** ptr, swig_type_info *info, int flags) {
  SwigPrivData *cdata;

  cdata = (SwigPrivData *) JSObjectGetPrivate(objRef);
  if(cdata == NULL) {
    return SWIG_ERROR;
  }
  if(cdata->info != info) {
    bool type_valid = false;
    swig_cast_info *t = info->cast;
    while(t != NULL) {
      if(t->type == cdata->info) {
        type_valid = true;
        break;
      }
      t = t->next;
    }
    if(!type_valid) {
      return SWIG_TypeError;
    }
  }

  *ptr = cdata->swigCObject;

  if(flags & SWIG_POINTER_DISOWN) {
    cdata->swigCMemOwn = false;
  }

  return SWIG_OK;
}

SWIGRUNTIME int SWIG_JSC_ConvertPtr(JSContextRef context, JSValueRef valRef, void** ptr, swig_type_info *info, int flags) {
  JSObjectRef objRef;

  /* special case: JavaScript null => C NULL pointer */
  if(JSValueIsNull(context, valRef)) {
    *ptr=0;
    return (flags & SWIG_POINTER_NO_NULL) ? SWIG_NullReferenceError : SWIG_OK;
  }

  if(!JSValueIsObject(context, valRef)) {
    return SWIG_TypeError;
  }

  objRef = JSValueToObject(context, valRef, NULL);
  if(objRef == NULL) {
    return SWIG_ERROR;
  }

  return SWIG_JSC_ConvertInstancePtr(context, objRef, ptr, info, flags);
}

SWIGRUNTIME JSObjectRef SWIG_JSC_NewPointerObj(JSContextRef context, void *ptr, swig_type_info *info, int flags) {
  JSClassRef classRef;
  JSObjectRef result;
  SwigPrivData *cdata;

  if (ptr == NULL) {
    // HACK: it is not possible to use JSValueToObject (causing seg-fault)
    // This static cast turned out to be a workaround
    // In future, we should change the interface of this method
    // to return JSValueRef instead of JSObjectRef.
    return (JSObjectRef) JSValueMakeNull(context);
  }

  if(info->clientdata == NULL) {
    classRef = _SwigObject_classRef;
  } else {
    classRef = (JSClassRef) info->clientdata;
  }

  result = JSObjectMake(context, classRef, NULL);

  cdata = (SwigPrivData*) malloc(sizeof(SwigPrivData));
  cdata->swigCObject = ptr;
  cdata->swigCMemOwn = (flags & SWIG_POINTER_OWN) ? 1 : 0;
  cdata->info = info;

  JSObjectSetPrivate(result, cdata);

  return result;
}

#define SWIG_ConvertPtr(obj, ptr, info, flags)    SWIG_JSC_ConvertPtr(context, obj, ptr, info, flags)
#define SWIG_NewPointerObj(ptr, info, flags)      SWIG_JSC_NewPointerObj(context, ptr, info, flags)

#define SWIG_ConvertInstance(obj, pptr, type, flags)    SWIG_JSC_ConvertInstancePtr(context, obj, pptr, type, flags)
#define SWIG_NewInstanceObj(thisvalue, type, flags)     SWIG_JSC_NewPointerObj(context, thisvalue, type, flags)

#define SWIG_ConvertFunctionPtr(obj, pptr, type)        SWIG_JSC_ConvertPtr(context, obj, pptr, type, 0)
#define SWIG_NewFunctionPtrObj(ptr, type)               SWIG_JSC_NewPointerObj(context, ptr, type, 0)

/* ----------------------------------------------------------------------------
 *  A class for packed data
 *
 * ---------------------------------------------------------------------------*/

typedef struct {
  void *data;
  size_t size;
  swig_type_info *type;
} SwigPackedData;

SWIGRUNTIME JSStaticValue _SwigPackedData_values[] = {
  {
    0, 0, 0, 0
  }
};
SWIGRUNTIME JSStaticFunction _SwigPackedData_functions[] = {
  {
    0, 0, 0
  }
};
SWIGRUNTIME JSClassDefinition _SwigPackedData_objectDefinition;
SWIGRUNTIME JSClassRef _SwigPackedData_classRef;

SWIGRUNTIMEINLINE
int SwigJSCPacked_Check(JSContextRef context, JSValueRef valRef) {
  return JSValueIsObjectOfClass(context, valRef, _SwigPackedData_classRef);
}

SWIGRUNTIME
swig_type_info* SwigJSCPacked_UnpackData(JSContextRef context, JSValueRef valRef, void *ptr, size_t size) {
  if (SwigJSCPacked_Check(context, valRef)) {
    JSObjectRef objRef = JSValueToObject(context, valRef, NULL);
    SwigPackedData *sobj = (SwigPackedData *) JSObjectGetPrivate(objRef);
    if (sobj->size != size) return 0;
    memcpy(ptr, sobj->data, size);
    return sobj->type;
  } else {
    return 0;
  }
}

SWIGRUNTIME
int SWIG_JSC_ConvertPacked(JSContextRef context, JSValueRef valRef, void *ptr, size_t sz, swig_type_info *ty) {
  swig_type_info *to = SwigJSCPacked_UnpackData(context, valRef, ptr, sz);
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

SWIGRUNTIME
JSValueRef SWIG_JSC_NewPackedObj(JSContextRef context, void *data, size_t size, swig_type_info *type) {

  JSClassRef classRef = _SwigObject_classRef;
  JSObjectRef result = JSObjectMake(context, classRef, NULL);

  SwigPackedData* cdata = (SwigPackedData*) malloc(sizeof(SwigPackedData));
  cdata->data = data;
  cdata->size = size;
  cdata->type = type;

  JSObjectSetPrivate(result, cdata);

  return result;
}

/*  SwigPackedData wrappers  */
SWIGRUNTIME
void _wrap_SwigPackedData_delete(JSObjectRef obj)
{
  SwigPackedData* cdata = (SwigPackedData*) JSObjectGetPrivate(obj);
  if (cdata) {
    free(cdata->data);
  }
}

/* for C++ member pointers, ie, member methods */

#define SWIG_ConvertMember(obj, ptr, sz, ty)            SWIG_JSC_ConvertPacked(context, obj, ptr, sz, ty)
#define SWIG_NewMemberObj(ptr, sz, type)                SWIG_JSC_NewPackedObj(context, ptr, sz, type)


/* ---------------------------------------------------------------------------
 * Support for IN/OUTPUT typemaps (see Lib/typemaps/inoutlist.swg)
 *
 * ---------------------------------------------------------------------------*/
SWIGRUNTIME
unsigned int SWIGJSC_ArrayLength(JSContextRef context, JSObjectRef arr) {
  static JSStringRef LENGTH = 0;
  JSValueRef exception = NULL;
  JSValueRef js_length;
  double length;

  if (LENGTH == 0) {
    LENGTH = JSStringCreateWithUTF8CString("length");
  }

  js_length = JSObjectGetProperty(context, arr, LENGTH, &exception);
  if (exception == 0 && JSValueIsNumber(context, js_length)) {
    length = JSValueToNumber(context, js_length, 0);
    return (unsigned int) length;
  } else {
    return 0;
  }
}

SWIGRUNTIME
JSValueRef SWIGJSC_AppendOutput(JSContextRef context, JSValueRef value, JSValueRef obj) {
  JSObjectRef arr;
  unsigned int length;

  if (JSValueIsUndefined(context, value)) {
    arr = JSObjectMakeArray(context, 0, 0, 0);
  } else {
    arr = JSValueToObject(context, value, 0);
  }

  length = SWIGJSC_ArrayLength(context, arr);
  JSObjectSetPropertyAtIndex(context, arr, length, obj, 0);
  return arr;
}
