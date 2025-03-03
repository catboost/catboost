/* ---------------------------------------------------------------------------
 * These typedefs and defines are used to deal with v8 API changes
 *
 * ---------------------------------------------------------------------------*/

// First v8 version that uses "SetWeak" and not "MakeWeak"

#define SWIGV8_SETWEAK_VERSION 0x032224

#if (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < 0x031803)
#define SWIGV8_STRING_NEW2(cstr, len) v8::String::New(cstr, len)
#else
#define SWIGV8_STRING_NEW2(cstr, len) v8::String::NewFromUtf8(v8::Isolate::GetCurrent(), cstr, v8::String::kNormalString, len)
#endif

#if (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < 0x031903)
typedef v8::Handle<v8::Value> SwigV8ReturnValue;
typedef v8::Arguments SwigV8Arguments;
typedef v8::AccessorInfo SwigV8PropertyCallbackInfo;
#define SWIGV8_RETURN(val) return scope.Close(val)
#define SWIGV8_RETURN_INFO(val, info) return scope.Close(val)
#else
typedef void SwigV8ReturnValue;
typedef v8::FunctionCallbackInfo<v8::Value> SwigV8Arguments;
typedef v8::PropertyCallbackInfo<v8::Value> SwigV8PropertyCallbackInfo;
#define SWIGV8_RETURN(val) args.GetReturnValue().Set(val); return
#define SWIGV8_RETURN_INFO(val, info) info.GetReturnValue().Set(val); return
#endif

#if (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < 0x032117)
#define SWIGV8_HANDLESCOPE() v8::HandleScope scope
#define SWIGV8_HANDLESCOPE_ESC() v8::HandleScope scope
#define SWIGV8_ESCAPE(val) return scope.Close(val)
#elif (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < 0x032224)
#define SWIGV8_HANDLESCOPE() v8::HandleScope scope(v8::Isolate::GetCurrent());
#define SWIGV8_HANDLESCOPE_ESC() v8::HandleScope scope(v8::Isolate::GetCurrent());
#define SWIGV8_ESCAPE(val) return scope.Close(val)
#else
#define SWIGV8_HANDLESCOPE() v8::HandleScope scope(v8::Isolate::GetCurrent());
#define SWIGV8_HANDLESCOPE_ESC() v8::EscapableHandleScope scope(v8::Isolate::GetCurrent());
#define SWIGV8_ESCAPE(val) return scope.Escape(val)
#endif

#if (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < 0x032224)
#define SWIGV8_ADJUST_MEMORY(size) v8::V8::AdjustAmountOfExternalAllocatedMemory(size)
#define SWIGV8_CURRENT_CONTEXT() v8::Context::GetCurrent()
#define SWIGV8_THROW_EXCEPTION(err) v8::ThrowException(err)
#define SWIGV8_STRING_NEW(str) v8::String::New(str)
#define SWIGV8_SYMBOL_NEW(sym) v8::String::NewSymbol(sym)
#else
#define SWIGV8_ADJUST_MEMORY(size) v8::Isolate::GetCurrent()->AdjustAmountOfExternalAllocatedMemory(size)
#define SWIGV8_CURRENT_CONTEXT() v8::Isolate::GetCurrent()->GetCurrentContext()
#define SWIGV8_THROW_EXCEPTION(err) v8::Isolate::GetCurrent()->ThrowException(err)
#define SWIGV8_STRING_NEW(str) v8::String::NewFromUtf8(v8::Isolate::GetCurrent(), str)
#define SWIGV8_SYMBOL_NEW(sym) v8::String::NewFromUtf8(v8::Isolate::GetCurrent(), sym)
#endif

#if (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < 0x032318)
#define SWIGV8_ARRAY_NEW() v8::Array::New()
#define SWIGV8_BOOLEAN_NEW(bool) v8::Boolean::New(bool)
#define SWIGV8_EXTERNAL_NEW(val) v8::External::New(val)
#define SWIGV8_FUNCTEMPLATE_NEW(func) v8::FunctionTemplate::New(func)
#define SWIGV8_FUNCTEMPLATE_NEW_VOID() v8::FunctionTemplate::New()
#define SWIGV8_INT32_NEW(num) v8::Int32::New(num)
#define SWIGV8_INTEGER_NEW(num) v8::Integer::New(num)
#define SWIGV8_INTEGER_NEW_UNS(num) v8::Integer::NewFromUnsigned(num)
#define SWIGV8_NUMBER_NEW(num) v8::Number::New(num)
#define SWIGV8_OBJECT_NEW() v8::Object::New()
#define SWIGV8_UNDEFINED() v8::Undefined()
#define SWIGV8_NULL() v8::Null()
#else
#define SWIGV8_ARRAY_NEW() v8::Array::New(v8::Isolate::GetCurrent())
#define SWIGV8_BOOLEAN_NEW(bool) v8::Boolean::New(v8::Isolate::GetCurrent(), bool)
#define SWIGV8_EXTERNAL_NEW(val) v8::External::New(v8::Isolate::GetCurrent(), val)
#define SWIGV8_FUNCTEMPLATE_NEW(func) v8::FunctionTemplate::New(v8::Isolate::GetCurrent(), func)
#define SWIGV8_FUNCTEMPLATE_NEW_VOID() v8::FunctionTemplate::New(v8::Isolate::GetCurrent())
#define SWIGV8_INT32_NEW(num) v8::Int32::New(v8::Isolate::GetCurrent(), num)
#define SWIGV8_INTEGER_NEW(num) v8::Integer::New(v8::Isolate::GetCurrent(), num)
#define SWIGV8_INTEGER_NEW_UNS(num) v8::Integer::NewFromUnsigned(v8::Isolate::GetCurrent(), num)
#define SWIGV8_NUMBER_NEW(num) v8::Number::New(v8::Isolate::GetCurrent(), num)
#define SWIGV8_OBJECT_NEW() v8::Object::New(v8::Isolate::GetCurrent())
#define SWIGV8_UNDEFINED() v8::Undefined(v8::Isolate::GetCurrent())
#define SWIGV8_NULL() v8::Null(v8::Isolate::GetCurrent())
#endif

#if (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < 0x031710)
#define SWIGV8_SET_CLASS_TEMPL(class_templ, class) class_templ = v8::Persistent<v8::FunctionTemplate>::New(class);
#elif (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < 0x031900)
#define SWIGV8_SET_CLASS_TEMPL(class_templ, class) class_templ = v8::Persistent<v8::FunctionTemplate>::New(v8::Isolate::GetCurrent(), class);
#else
#define SWIGV8_SET_CLASS_TEMPL(class_templ, class) class_templ.Reset(v8::Isolate::GetCurrent(), class);
#endif

#ifdef NODE_VERSION
#if NODE_VERSION_AT_LEAST(10, 12, 0)
#define SWIG_NODE_AT_LEAST_1012
#endif
#endif

//Necessary to check Node.js version because V8 API changes are backported in Node.js
#if (defined(NODE_VERSION) && !defined(SWIG_NODE_AT_LEAST_1012)) || \
    (!defined(NODE_VERSION) && (V8_MAJOR_VERSION-0) < 7)
#define SWIGV8_TO_OBJECT(handle) (handle)->ToObject()
#define SWIGV8_TO_STRING(handle) (handle)->ToString()
#define SWIGV8_NUMBER_VALUE(handle) (handle)->NumberValue()
#define SWIGV8_INTEGER_VALUE(handle) (handle)->IntegerValue()
#define SWIGV8_BOOLEAN_VALUE(handle) (handle)->BooleanValue()
#define SWIGV8_WRITE_UTF8(handle, buffer, len) (handle)->WriteUtf8(buffer, len)
#define SWIGV8_UTF8_LENGTH(handle) (handle)->Utf8Length()
#else
#define SWIGV8_TO_OBJECT(handle) (handle)->ToObject(SWIGV8_CURRENT_CONTEXT()).ToLocalChecked()
#define SWIGV8_TO_STRING(handle) (handle)->ToString(SWIGV8_CURRENT_CONTEXT()).ToLocalChecked()
#define SWIGV8_NUMBER_VALUE(handle) (handle)->NumberValue(SWIGV8_CURRENT_CONTEXT()).ToChecked()
#define SWIGV8_INTEGER_VALUE(handle) (handle)->IntegerValue(SWIGV8_CURRENT_CONTEXT()).ToChecked()
#define SWIGV8_BOOLEAN_VALUE(handle) (handle)->BooleanValue(SWIGV8_CURRENT_CONTEXT()).ToChecked()
#define SWIGV8_WRITE_UTF8(handle, buffer, len) (handle)->WriteUtf8(v8::Isolate::GetCurrent(), buffer, len)
#define SWIGV8_UTF8_LENGTH(handle) (handle)->Utf8Length(v8::Isolate::GetCurrent())
#endif

/* ---------------------------------------------------------------------------
 * Error handling
 *
 * ---------------------------------------------------------------------------*/

#define SWIG_Error(code, msg)     SWIGV8_ErrorHandler.error(code, msg)
#define SWIG_exception(code, msg) do { SWIGV8_ErrorHandler.error(code, msg); SWIG_fail; } while (0)
#define SWIG_fail                 goto fail
#define SWIGV8_OVERLOAD false

SWIGINTERN void SWIG_V8_Raise(const char *msg) {
  SWIGV8_THROW_EXCEPTION(v8::Exception::Error(SWIGV8_STRING_NEW(msg)));
}

/*
  Note: There are two contexts for handling errors.
  A static V8ErrorHandler is used in not overloaded methods.
  For overloaded methods the throwing type checking mechanism is used
  during dispatching. As V8 exceptions can not be reset properly
  the trick is to use a dynamic ErrorHandler with same local name as the global
  one.

  - See definition of SWIG_Error above.
  - See code templates 'JS_function_dispatcher', 'JS_functionwrapper_overload',
    and 'JS_function_dispatch_case' in javascriptcode.swg

*/
class V8ErrorHandler {
public:
  virtual ~V8ErrorHandler() {}
  virtual void error(int code, const char *msg) {
    SWIG_V8_Raise(msg);
  }
};
// this is used in usually
SWIGRUNTIME V8ErrorHandler SWIGV8_ErrorHandler;

// instances of this are used in overloaded functions
class OverloadErrorHandler: public V8ErrorHandler {
public:
  virtual void error(int code, const char *msg) {
    err = v8::Exception::Error(SWIGV8_STRING_NEW(msg));
    if(code != SWIG_TypeError) {
        SWIGV8_THROW_EXCEPTION(err);
    }
  }
  v8::Handle<v8::Value> err;
};

/* ---------------------------------------------------------------------------
 * Basic Proxy object
 *
 * ---------------------------------------------------------------------------*/

// Note: to trigger the v8 gc more often one can tell v8 about the memory consumption
// TODO: we could add a v8 specific parameter to control this value
#define SWIGV8_AVG_OBJ_SIZE 1000

class SWIGV8_Proxy {
public:
  SWIGV8_Proxy(): swigCMemOwn(false), swigCObject(0), info(0) {
    SWIGV8_ADJUST_MEMORY(SWIGV8_AVG_OBJ_SIZE);
  };

  ~SWIGV8_Proxy() {
#if (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < 0x031710)
    handle.ClearWeak();
    handle.Dispose();
#elif (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < 0x032100)
    handle.ClearWeak(v8::Isolate::GetCurrent());
    handle.Dispose(v8::Isolate::GetCurrent());
#elif (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < SWIGV8_SETWEAK_VERSION)
    handle.ClearWeak();
    handle.Dispose();
#else    
    handle.ClearWeak();
    handle.Reset();
#endif

#if (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < SWIGV8_SETWEAK_VERSION)
    handle.Clear();
#endif

    SWIGV8_ADJUST_MEMORY(-SWIGV8_AVG_OBJ_SIZE);
  }

  bool swigCMemOwn;
  void *swigCObject;
  swig_type_info *info;
  v8::Persistent<v8::Object> handle;
};

class SWIGV8_ClientData {
public:
  v8::Persistent<v8::FunctionTemplate> class_templ;

#if (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < 0x031710)
  void (*dtor) (v8::Persistent< v8::Value> object, void *parameter);
#elif (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < 0x031900)
  void (*dtor) (v8::Isolate *isolate, v8::Persistent< v8::Value> object, void *parameter);
#elif (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < SWIGV8_SETWEAK_VERSION)
  void (*dtor) (v8::Isolate *isolate, v8::Persistent< v8::Object > *object, SWIGV8_Proxy *proxy);
#elif (V8_MAJOR_VERSION-0) < 5
  void (*dtor) (const v8::WeakCallbackData<v8::Object, SWIGV8_Proxy> &data);
#else
  void (*dtor) (const v8::WeakCallbackInfo<SWIGV8_Proxy> &data);
#endif
};

SWIGRUNTIME v8::Persistent<v8::FunctionTemplate> SWIGV8_SWIGTYPE_Proxy_class_templ;

SWIGRUNTIME int SWIG_V8_ConvertInstancePtr(v8::Handle<v8::Object> objRef, void **ptr, swig_type_info *info, int flags) {
  SWIGV8_HANDLESCOPE();

  if(objRef->InternalFieldCount() < 1) return SWIG_ERROR;

#if (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < 0x031511)
  v8::Handle<v8::Value> cdataRef = objRef->GetInternalField(0);
  SWIGV8_Proxy *cdata = static_cast<SWIGV8_Proxy *>(v8::External::Unwrap(cdataRef));
#else
  SWIGV8_Proxy *cdata = static_cast<SWIGV8_Proxy *>(objRef->GetAlignedPointerFromInternalField(0));
#endif

  if(cdata == NULL) {
    return SWIG_ERROR;
  }
  if(cdata->info != info) {
    swig_cast_info *tc = SWIG_TypeCheckStruct(cdata->info, info);
    if (!tc && cdata->info->name) {
      tc = SWIG_TypeCheck(cdata->info->name, info);
    }
    bool type_valid = tc != 0;
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


#if (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < 0x031710)
SWIGRUNTIME void SWIGV8_Proxy_DefaultDtor(v8::Persistent< v8::Value > object, void *parameter) {
  SWIGV8_Proxy *proxy = static_cast<SWIGV8_Proxy *>(parameter);
#elif (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < 0x031900)
SWIGRUNTIME void SWIGV8_Proxy_DefaultDtor(v8::Isolate *, v8::Persistent< v8::Value > object, void *parameter) {
  SWIGV8_Proxy *proxy = static_cast<SWIGV8_Proxy *>(parameter);
#elif (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < SWIGV8_SETWEAK_VERSION)
SWIGRUNTIME void SWIGV8_Proxy_DefaultDtor(v8::Isolate *, v8::Persistent< v8::Object > *object, SWIGV8_Proxy *proxy) {
#elif (V8_MAJOR_VERSION-0) < 5
SWIGRUNTIME void SWIGV8_Proxy_DefaultDtor(const v8::WeakCallbackData<v8::Object, SWIGV8_Proxy> &data) {
  SWIGV8_Proxy *proxy = data.GetParameter();
#else
SWIGRUNTIME void SWIGV8_Proxy_DefaultDtor(const v8::WeakCallbackInfo<SWIGV8_Proxy> &data) {
  SWIGV8_Proxy *proxy = data.GetParameter();
#endif

  delete proxy;
}

SWIGRUNTIME int SWIG_V8_GetInstancePtr(v8::Handle<v8::Value> valRef, void **ptr) {
  if(!valRef->IsObject()) {
    return SWIG_TypeError;
  }
  v8::Handle<v8::Object> objRef = SWIGV8_TO_OBJECT(valRef);

  if(objRef->InternalFieldCount() < 1) return SWIG_ERROR;

#if (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < 0x031511)
  v8::Handle<v8::Value> cdataRef = objRef->GetInternalField(0);
  SWIGV8_Proxy *cdata = static_cast<SWIGV8_Proxy *>(v8::External::Unwrap(cdataRef));
#else
  SWIGV8_Proxy *cdata = static_cast<SWIGV8_Proxy *>(objRef->GetAlignedPointerFromInternalField(0));
#endif

  if(cdata == NULL) {
    return SWIG_ERROR;
  }

  *ptr = cdata->swigCObject;

  return SWIG_OK;
}

SWIGRUNTIME void SWIGV8_SetPrivateData(v8::Handle<v8::Object> obj, void *ptr, swig_type_info *info, int flags) {
  SWIGV8_Proxy *cdata = new SWIGV8_Proxy();
  cdata->swigCObject = ptr;
  cdata->swigCMemOwn = (flags & SWIG_POINTER_OWN) ? 1 : 0;
  cdata->info = info;

#if (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < 0x031511)
  obj->SetPointerInInternalField(0, cdata);
#else
  obj->SetAlignedPointerInInternalField(0, cdata);
#endif

#if (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < 0x031710)
  cdata->handle = v8::Persistent<v8::Object>::New(obj);
#elif (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < 0x031900)
  cdata->handle = v8::Persistent<v8::Object>::New(v8::Isolate::GetCurrent(), obj);
#else
  cdata->handle.Reset(v8::Isolate::GetCurrent(), obj);
#endif

#if (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < 0x031710)
  // clientdata must be set for owned data as we need to register the dtor
  if(cdata->swigCMemOwn && (SWIGV8_ClientData*)info->clientdata) {
    cdata->handle.MakeWeak(cdata, ((SWIGV8_ClientData*)info->clientdata)->dtor);
  } else {
    cdata->handle.MakeWeak(cdata, SWIGV8_Proxy_DefaultDtor);
  }
#elif (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < 0x031918)
  if(cdata->swigCMemOwn && (SWIGV8_ClientData*)info->clientdata) {
    cdata->handle.MakeWeak(v8::Isolate::GetCurrent(), cdata, ((SWIGV8_ClientData*)info->clientdata)->dtor);
  } else {
    cdata->handle.MakeWeak(v8::Isolate::GetCurrent(), cdata, SWIGV8_Proxy_DefaultDtor);
  }
#elif (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < SWIGV8_SETWEAK_VERSION)
  if(cdata->swigCMemOwn && (SWIGV8_ClientData*)info->clientdata) {
    cdata->handle.MakeWeak(cdata, ((SWIGV8_ClientData*)info->clientdata)->dtor);
  } else {
    cdata->handle.MakeWeak(cdata, SWIGV8_Proxy_DefaultDtor);
  }
#elif (V8_MAJOR_VERSION-0) < 5
  if(cdata->swigCMemOwn && (SWIGV8_ClientData*)info->clientdata) {
    cdata->handle.SetWeak(cdata, ((SWIGV8_ClientData*)info->clientdata)->dtor);
  } else {
    cdata->handle.SetWeak(cdata, SWIGV8_Proxy_DefaultDtor);
  }
#else
  if(cdata->swigCMemOwn && (SWIGV8_ClientData*)info->clientdata) {
    cdata->handle.SetWeak(cdata, ((SWIGV8_ClientData*)info->clientdata)->dtor, v8::WeakCallbackType::kParameter);
  } else {
    cdata->handle.SetWeak(cdata, SWIGV8_Proxy_DefaultDtor, v8::WeakCallbackType::kParameter);
  }
#endif

#if (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < 0x031710)
  cdata->handle.MarkIndependent();
#elif (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < 0x032100)
  cdata->handle.MarkIndependent(v8::Isolate::GetCurrent());
#else
  cdata->handle.MarkIndependent();
#endif

}

SWIGRUNTIME int SWIG_V8_ConvertPtr(v8::Handle<v8::Value> valRef, void **ptr, swig_type_info *info, int flags) {
  SWIGV8_HANDLESCOPE();
  
  /* special case: JavaScript null => C NULL pointer */
  if(valRef->IsNull()) {
    *ptr=0;
    return (flags & SWIG_POINTER_NO_NULL) ? SWIG_NullReferenceError : SWIG_OK;
  }
  if(!valRef->IsObject()) {
    return SWIG_TypeError;
  }
  v8::Handle<v8::Object> objRef = SWIGV8_TO_OBJECT(valRef);
  return SWIG_V8_ConvertInstancePtr(objRef, ptr, info, flags);
}

SWIGRUNTIME v8::Handle<v8::Value> SWIG_V8_NewPointerObj(void *ptr, swig_type_info *info, int flags) {
  SWIGV8_HANDLESCOPE_ESC();
  
  v8::Handle<v8::FunctionTemplate> class_templ;

  if (ptr == NULL) {
#if (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < 0x031903)
    SWIGV8_ESCAPE(SWIGV8_NULL());
#else    
    v8::Local<v8::Primitive> result = SWIGV8_NULL();
    SWIGV8_ESCAPE(result);
#endif
  }

#if (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < 0x031903)
  if(info->clientdata != 0) {
    class_templ = ((SWIGV8_ClientData*) info->clientdata)->class_templ;
  } else {
    class_templ = SWIGV8_SWIGTYPE_Proxy_class_templ;
  }
#else
  v8::Isolate *isolate = v8::Isolate::GetCurrent();

  if(info->clientdata != 0) {
    class_templ = v8::Local<v8::FunctionTemplate>::New(isolate, ((SWIGV8_ClientData*) info->clientdata)->class_templ);
  } else {
    class_templ = v8::Local<v8::FunctionTemplate>::New(isolate, SWIGV8_SWIGTYPE_Proxy_class_templ);
  }
#endif

//  v8::Handle<v8::Object> result = class_templ->InstanceTemplate()->NewInstance();
  v8::Local<v8::Object> result = class_templ->InstanceTemplate()->NewInstance();
  SWIGV8_SetPrivateData(result, ptr, info, flags);

  SWIGV8_ESCAPE(result);
}

#define SWIG_ConvertPtr(obj, ptr, info, flags)    SWIG_V8_ConvertPtr(obj, ptr, info, flags)
#define SWIG_NewPointerObj(ptr, info, flags)      SWIG_V8_NewPointerObj(ptr, info, flags)

#define SWIG_ConvertInstance(obj, pptr, type, flags)    SWIG_V8_ConvertInstancePtr(obj, pptr, type, flags)
#define SWIG_NewInstanceObj(thisvalue, type, flags)     SWIG_V8_NewPointerObj(thisvalue, type, flags)

#define SWIG_ConvertFunctionPtr(obj, pptr, type)        SWIG_V8_ConvertPtr(obj, pptr, type, 0)
#define SWIG_NewFunctionPtrObj(ptr, type)               SWIG_V8_NewPointerObj(ptr, type, 0)

#define SWIG_GetInstancePtr(obj, ptr)    SWIG_V8_GetInstancePtr(obj, ptr)

SWIGRUNTIME SwigV8ReturnValue _SWIGV8_wrap_equals(const SwigV8Arguments &args) {
  SWIGV8_HANDLESCOPE();
  
  v8::Handle<v8::Value> jsresult;
  void *arg1 = (void *) 0 ;
  void *arg2 = (void *) 0 ;
  bool result;
  int res1;
  int res2;

  if(args.Length() != 1) SWIG_exception_fail(SWIG_ERROR, "Illegal number of arguments for equals.");

  res1 = SWIG_GetInstancePtr(args.Holder(), &arg1);
  if (!SWIG_IsOK(res1)) {
    SWIG_exception_fail(SWIG_ERROR, "Could not get pointer from 'this' object for equals.");
  }
  res2 = SWIG_GetInstancePtr(args[0], &arg2);
  if (!SWIG_IsOK(res2)) {
    SWIG_exception_fail(SWIG_ArgError(res2), "in method '" "equals" "', argument " "1"" of type '" "void *""'");
  }

  result = (bool)(arg1 == arg2);
  jsresult =  SWIGV8_BOOLEAN_NEW(result);

  SWIGV8_RETURN(jsresult);
  goto fail;
fail:
  SWIGV8_RETURN(SWIGV8_UNDEFINED());
}

SWIGRUNTIME SwigV8ReturnValue _wrap_getCPtr(const SwigV8Arguments &args) {
  SWIGV8_HANDLESCOPE();
  
  v8::Handle<v8::Value> jsresult;
  void *arg1 = (void *) 0 ;
  long result;
  int res1;

  res1 = SWIG_GetInstancePtr(args.Holder(), &arg1);
  if (!SWIG_IsOK(res1)) {
    SWIG_exception_fail(SWIG_ArgError(res1), "in method '" "getCPtr" "', argument " "1"" of type '" "void *""'");
  }

  result = (long)arg1;
  jsresult = SWIGV8_NUMBER_NEW(result);

  SWIGV8_RETURN(jsresult);
  goto fail;
fail:
  SWIGV8_RETURN(SWIGV8_UNDEFINED());
}

/* ---------------------------------------------------------------------------
 * PackedData object
 *
 * ---------------------------------------------------------------------------*/

class SwigV8PackedData {
public:
  SwigV8PackedData(void *data, size_t size, swig_type_info *type): data(data), size(size), type(type) {};

  ~SwigV8PackedData() {
  };

  void *data;
  size_t size;
  swig_type_info *type;

  v8::Persistent<v8::Object> handle;
};

SWIGRUNTIMEINLINE
int SwigV8Packed_Check(v8::Handle<v8::Value> valRef) {
  SWIGV8_HANDLESCOPE();
  
  v8::Handle<v8::Object> objRef = SWIGV8_TO_OBJECT(valRef);
  if(objRef->InternalFieldCount() < 1) return false;
#if (V8_MAJOR_VERSION-0) < 5
  v8::Handle<v8::Value> flag = objRef->GetHiddenValue(SWIGV8_STRING_NEW("__swig__packed_data__"));
#else
  v8::Local<v8::Private> privateKey = v8::Private::ForApi(v8::Isolate::GetCurrent(), SWIGV8_STRING_NEW("__swig__packed_data__"));
  v8::Local<v8::Value> flag;
  if (!objRef->GetPrivate(SWIGV8_CURRENT_CONTEXT(), privateKey).ToLocal(&flag))
    return false;
#endif
  return (flag->IsBoolean() && SWIGV8_BOOLEAN_VALUE(flag));
}

SWIGRUNTIME
swig_type_info *SwigV8Packed_UnpackData(v8::Handle<v8::Value> valRef, void *ptr, size_t size) {
  if (SwigV8Packed_Check(valRef)) {
    SWIGV8_HANDLESCOPE();
    
    SwigV8PackedData *sobj;

    v8::Handle<v8::Object> objRef = SWIGV8_TO_OBJECT(valRef);

#if (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < 0x031511)
    v8::Handle<v8::Value> cdataRef = objRef->GetInternalField(0);
    sobj = static_cast<SwigV8PackedData*>(v8::External::Unwrap(cdataRef));
#else
    sobj = static_cast<SwigV8PackedData*>(objRef->GetAlignedPointerFromInternalField(0));
#endif
    if (sobj == NULL || sobj->size != size) return 0;
    memcpy(ptr, sobj->data, size);
    return sobj->type;
  } else {
    return 0;
  }
}

SWIGRUNTIME
int SWIGV8_ConvertPacked(v8::Handle<v8::Value> valRef, void *ptr, size_t sz, swig_type_info *ty) {
  swig_type_info *to = SwigV8Packed_UnpackData(valRef, ptr, sz);
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

#if (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < 0x031710)
SWIGRUNTIME void _wrap_SwigV8PackedData_delete(v8::Persistent< v8::Value > object, void *parameter) {
  SwigV8PackedData *cdata = static_cast<SwigV8PackedData *>(parameter);
#elif (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < 0x031900)
SWIGRUNTIME void _wrap_SwigV8PackedData_delete(v8::Isolate *isolate, v8::Persistent<v8::Value> object, void *parameter) {
  SwigV8PackedData *cdata = static_cast<SwigV8PackedData *>(parameter);
#elif (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < SWIGV8_SETWEAK_VERSION)
SWIGRUNTIME void _wrap_SwigV8PackedData_delete(v8::Isolate *isolate, v8::Persistent<v8::Object> *object, SwigV8PackedData *cdata) {
#elif (V8_MAJOR_VERSION-0) < 5
SWIGRUNTIME void _wrap_SwigV8PackedData_delete(const v8::WeakCallbackData<v8::Object, SwigV8PackedData> &data) {
  v8::Local<v8::Object> object = data.GetValue();
  SwigV8PackedData *cdata = data.GetParameter();
#else
SWIGRUNTIME void _wrap_SwigV8PackedData_delete(const v8::WeakCallbackInfo<SwigV8PackedData> &data) {
  SwigV8PackedData *cdata = data.GetParameter();
#endif

  delete cdata;

#if (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < 0x031710)
  object.Clear();
  object.Dispose();
#elif (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < 0x031900)
  object.Clear();
  object.Dispose(isolate);
#elif (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < 0x032100)
  object->Dispose(isolate);
#elif (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < SWIGV8_SETWEAK_VERSION)
  object->Dispose();
#elif (V8_MAJOR_VERSION-0) < 5
  object.Clear();
#endif
}

SWIGRUNTIME
v8::Handle<v8::Value> SWIGV8_NewPackedObj(void *data, size_t size, swig_type_info *type) {
  SWIGV8_HANDLESCOPE_ESC();

  SwigV8PackedData *cdata = new SwigV8PackedData(data, size, type);
//  v8::Handle<v8::Object> obj = SWIGV8_OBJECT_NEW();
  v8::Local<v8::Object> obj = SWIGV8_OBJECT_NEW();

#if (V8_MAJOR_VERSION-0) < 5
  obj->SetHiddenValue(SWIGV8_STRING_NEW("__swig__packed_data__"), SWIGV8_BOOLEAN_NEW(true));
#else
  v8::Local<v8::Private> privateKey = v8::Private::ForApi(v8::Isolate::GetCurrent(), SWIGV8_STRING_NEW("__swig__packed_data__"));
  obj->SetPrivate(SWIGV8_CURRENT_CONTEXT(), privateKey, SWIGV8_BOOLEAN_NEW(true));
#endif

#if (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < 0x031511)
  obj->SetPointerInInternalField(0, cdata);
#else
  obj->SetAlignedPointerInInternalField(0, cdata);
#endif

#if (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < 0x031710)
  cdata->handle = v8::Persistent<v8::Object>::New(obj);
#elif (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < 0x031900)
  cdata->handle = v8::Persistent<v8::Object>::New(v8::Isolate::GetCurrent(), obj);
#else
  cdata->handle.Reset(v8::Isolate::GetCurrent(), obj);
#endif


#if (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < 0x031710)
  cdata->handle.MakeWeak(cdata, _wrap_SwigV8PackedData_delete);
#elif (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < 0x031918)
  cdata->handle.MakeWeak(v8::Isolate::GetCurrent(), cdata, _wrap_SwigV8PackedData_delete);
#elif (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < SWIGV8_SETWEAK_VERSION)
  cdata->handle.MakeWeak(cdata, _wrap_SwigV8PackedData_delete);
#elif (V8_MAJOR_VERSION-0) < 5
  cdata->handle.SetWeak(cdata, _wrap_SwigV8PackedData_delete);
//  v8::V8::SetWeak(&cdata->handle, cdata, _wrap_SwigV8PackedData_delete);
#else
  cdata->handle.SetWeak(cdata, _wrap_SwigV8PackedData_delete, v8::WeakCallbackType::kParameter);
#endif

#if (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < 0x031710)
  cdata->handle.MarkIndependent();
#elif (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < 0x032100)
  cdata->handle.MarkIndependent(v8::Isolate::GetCurrent());
#else
  cdata->handle.MarkIndependent();
#endif

  SWIGV8_ESCAPE(obj);
}

#define SWIG_ConvertMember(obj, ptr, sz, ty)            SWIGV8_ConvertPacked(obj, ptr, sz, ty)
#define SWIG_NewMemberObj(ptr, sz, type)                SWIGV8_NewPackedObj(ptr, sz, type)


/* ---------------------------------------------------------------------------
 * Support for IN/OUTPUT typemaps (see Lib/typemaps/inoutlist.swg)
 *
 * ---------------------------------------------------------------------------*/

SWIGRUNTIME

#if (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < 0x031903)
v8::Handle<v8::Value> SWIGV8_AppendOutput(v8::Handle<v8::Value> result, v8::Handle<v8::Value> obj) {
#else
v8::Handle<v8::Value> SWIGV8_AppendOutput(v8::Local<v8::Value> result, v8::Handle<v8::Value> obj) {
#endif
  SWIGV8_HANDLESCOPE_ESC();
  
  if (result->IsUndefined()) {
    result = SWIGV8_ARRAY_NEW();
  }
#if (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < 0x031903)
  v8::Handle<v8::Array> arr = v8::Handle<v8::Array>::Cast(result);
#else  
  v8::Local<v8::Array> arr = v8::Local<v8::Array>::Cast(result);
#endif
  arr->Set(arr->Length(), obj);

  SWIGV8_ESCAPE(arr);
}
