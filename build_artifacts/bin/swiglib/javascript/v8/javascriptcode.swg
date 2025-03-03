/* -----------------------------------------------------------------------------
 * js_ctor:  template for wrapping a ctor.
 *   - $jswrapper:        wrapper of called ctor
 *   - $jslocals:         locals part of wrapper
 *   - $jscode:           code part of wrapper
 *   - $jsargcount:       number of arguments
 *   - $jsmangledtype:    mangled type of class
 * ----------------------------------------------------------------------------- */

%fragment("js_ctor", "templates") %{
static SwigV8ReturnValue $jswrapper(const SwigV8Arguments &args) {
  SWIGV8_HANDLESCOPE();
  
  v8::Handle<v8::Object> self = args.Holder();
  $jslocals
  if(args.Length() != $jsargcount) SWIG_exception_fail(SWIG_ERROR, "Illegal number of arguments for $jswrapper.");
  $jscode

  SWIGV8_SetPrivateData(self, result, SWIGTYPE_$jsmangledtype, SWIG_POINTER_OWN);
  SWIGV8_RETURN(self);

  goto fail;
fail:
  SWIGV8_RETURN(SWIGV8_UNDEFINED());
}
%}

/* -----------------------------------------------------------------------------
 * js_veto_ctor:  a vetoing ctor for abstract classes
 *   - $jswrapper:        name of wrapper
 *   - $jsname:           class name
 * ----------------------------------------------------------------------------- */
%fragment ("js_veto_ctor", "templates")
%{
static SwigV8ReturnValue $jswrapper(const SwigV8Arguments &args) {
  SWIGV8_HANDLESCOPE();
  
  SWIG_exception(SWIG_ERROR, "Class $jsname can not be instantiated");
fail:
  SWIGV8_RETURN(SWIGV8_UNDEFINED());
}
%}

/* -----------------------------------------------------------------------------
 * js_ctor_dispatcher:  dispatcher for overloaded constructors
 *   - $jswrapper:        name of wrapper
 *   - $jsname:           class name
 *   - $jsdispatchcases:  part containing code for dispatching
 * ----------------------------------------------------------------------------- */
%fragment ("js_ctor_dispatcher", "templates")
%{
static SwigV8ReturnValue $jswrapper(const SwigV8Arguments &args) {
  SWIGV8_HANDLESCOPE();
  
  OverloadErrorHandler errorHandler;
  v8::Handle<v8::Value> self;

  // switch all cases by means of series of if-returns.
  $jsdispatchcases

  // default:
  SWIG_exception_fail(SWIG_ERROR, "Illegal arguments for construction of $jsmangledname");

fail:
  SWIGV8_RETURN(SWIGV8_UNDEFINED());
}
%}

/* -----------------------------------------------------------------------------
 * js_overloaded_ctor:  template for wrapping a ctor.
 *   - $jswrapper:        wrapper of called ctor
 *   - $jslocals:         locals part of wrapper
 *   - $jscode:           code part of wrapper
 *   - $jsargcount:       number of arguments
 *   - $jsmangledtype:    mangled type of class
 * ----------------------------------------------------------------------------- */
%fragment("js_overloaded_ctor", "templates") %{
static SwigV8ReturnValue $jswrapper(const SwigV8Arguments &args, V8ErrorHandler &SWIGV8_ErrorHandler) {
  SWIGV8_HANDLESCOPE();
  
  v8::Handle<v8::Object> self = args.Holder();
  $jslocals
  if(args.Length() != $jsargcount) SWIG_exception_fail(SWIG_ERROR, "Illegal number of arguments for $jswrapper.");
  $jscode

  SWIGV8_SetPrivateData(self, result, SWIGTYPE_$jsmangledtype, SWIG_POINTER_OWN);
  SWIGV8_RETURN(self);

  goto fail;
fail:
  SWIGV8_RETURN(SWIGV8_UNDEFINED());
}
%}

/* -----------------------------------------------------------------------------
 * js_ctor_dispatch_case:  template for a dispatch case for calling an overloaded ctor.
 *   - $jsargcount:       number of arguments of called ctor
 *   - $jswrapper:        wrapper of called ctor
 *
 *  Note: a try-catch-like mechanism is used to switch cases
 * ----------------------------------------------------------------------------- */
%fragment ("js_ctor_dispatch_case", "templates")
%{
  if(args.Length() == $jsargcount) {
    errorHandler.err.Clear();
#if (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < 0x031903)
    self = $jswrapper(args, errorHandler);
    if(errorHandler.err.IsEmpty()) {
      SWIGV8_ESCAPE(self);
    }
#else
    $jswrapper(args, errorHandler);
    if(errorHandler.err.IsEmpty()) {
      return;
    }
#endif
  }
%}

/* -----------------------------------------------------------------------------
 * js_dtor:  template for a destructor wrapper
 *   - $jsmangledname:  mangled class name
 *   - $jstype:         class type
 * ----------------------------------------------------------------------------- */
%fragment ("js_dtor", "templates")
%{

#if (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < 0x031710)
static void $jswrapper(v8::Persistent< v8::Value > object, void *parameter) {
  SWIGV8_Proxy *proxy = static_cast<SWIGV8_Proxy *>(parameter);
#elif (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < 0x031900)
static void $jswrapper(v8::Isolate *isolate, v8::Persistent<v8::Value> object, void *parameter) {
  SWIGV8_Proxy *proxy = static_cast<SWIGV8_Proxy *>(parameter);
#elif (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < SWIGV8_SETWEAK_VERSION)
static void $jswrapper(v8::Isolate *isolate, v8::Persistent<v8::Object> *object, SWIGV8_Proxy *proxy) {
#elif (V8_MAJOR_VERSION-0) < 5
static void $jswrapper(const v8::WeakCallbackData<v8::Object, SWIGV8_Proxy> &data) {
  v8::Local<v8::Object> object = data.GetValue();
  SWIGV8_Proxy *proxy = data.GetParameter();
#else
  static void $jswrapper(const v8::WeakCallbackInfo<SWIGV8_Proxy> &data) {
  SWIGV8_Proxy *proxy = data.GetParameter();
#endif

  if(proxy->swigCMemOwn && proxy->swigCObject) {
#ifdef SWIGRUNTIME_DEBUG
    printf("Deleting wrapped instance: %s\n", proxy->info->name);
#endif
    $jsfree proxy->swigCObject;
  }
  delete proxy;

#if (V8_MAJOR_VERSION-0) < 5
  object.Clear();
#endif
  
#if (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < 0x031710)
  object.Dispose();
#elif (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < 0x031900)
  object.Dispose(isolate);
#elif (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < 0x032100)
  object->Dispose(isolate);
#elif (V8_MAJOR_VERSION-0) < 5
  object->Dispose();
#endif
}
%}

/* -----------------------------------------------------------------------------
 * js_dtoroverride:  template for a destructor wrapper
 *   - $jsmangledname:  mangled class name
 *   - $jstype:         class type
 *   - ${destructor_action}: The custom destructor action to invoke.
 * ----------------------------------------------------------------------------- */
%fragment ("js_dtoroverride", "templates")
%{
#if (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < 0x031710)
static void $jswrapper(v8::Persistent<v8::Value> object, void *parameter) {
  SWIGV8_Proxy *proxy = static_cast<SWIGV8_Proxy *>(parameter);
#elif (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < 0x031900)
static void $jswrapper(v8::Isolate *isolate, v8::Persistent<v8::Value> object, void *parameter) {
  SWIGV8_Proxy *proxy = static_cast<SWIGV8_Proxy *>(parameter);
#elif (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < SWIGV8_SETWEAK_VERSION)
static void $jswrapper(v8::Isolate *isolate, v8::Persistent< v8::Object> *object, SWIGV8_Proxy *proxy) {
#elif (V8_MAJOR_VERSION-0) < 5
static void $jswrapper(const v8::WeakCallbackData<v8::Object, SWIGV8_Proxy> &data) {
  v8::Local<v8::Object> object = data.GetValue();
  SWIGV8_Proxy *proxy = data.GetParameter();
#else
static void $jswrapper(const v8::WeakCallbackInfo<SWIGV8_Proxy> &data) {
  SWIGV8_Proxy *proxy = data.GetParameter();
#endif

  if(proxy->swigCMemOwn && proxy->swigCObject) {
    $jstype arg1 = ($jstype)proxy->swigCObject;
    ${destructor_action}
  }
  delete proxy;

#if (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < 0x031710)
  object.Dispose();
#elif (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < 0x031900)
  object.Dispose(isolate);
#elif (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < 0x032100)
  object->Dispose(isolate);
#elif (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < SWIGV8_SETWEAK_VERSION)
  object->Dispose();
#elif (V8_MAJOR_VERSION-0) < 5
  object.Clear();
#endif
}
%}

/* -----------------------------------------------------------------------------
 * js_getter:  template for getter function wrappers
 *   - $jswrapper:  wrapper function name
 *   - $jslocals:   locals part of wrapper
 *   - $jscode:     code part of wrapper
 * ----------------------------------------------------------------------------- */
%fragment("js_getter", "templates")
%{
#if (V8_MAJOR_VERSION-0) < 5
static SwigV8ReturnValue $jswrapper(v8::Local<v8::String> property, const SwigV8PropertyCallbackInfo &info) {
#else
static SwigV8ReturnValue $jswrapper(v8::Local<v8::Name> property, const SwigV8PropertyCallbackInfo &info) {
#endif
  SWIGV8_HANDLESCOPE();
  
  v8::Handle<v8::Value> jsresult;
  $jslocals
  $jscode
  SWIGV8_RETURN_INFO(jsresult, info);

  goto fail;
fail:
  SWIGV8_RETURN_INFO(SWIGV8_UNDEFINED(), info);
}
%}

/* -----------------------------------------------------------------------------
 * js_setter:  template for setter function wrappers
 *   - $jswrapper:  wrapper function name
 *   - $jslocals:   locals part of wrapper
 *   - $jscode:     code part of wrapper
 * ----------------------------------------------------------------------------- */
%fragment("js_setter", "templates")
%{
#if (V8_MAJOR_VERSION-0) < 5
static void $jswrapper(v8::Local<v8::String> property, v8::Local<v8::Value> value, const SwigV8PropertyCallbackInfoVoid &info) {
#else
static void $jswrapper(v8::Local<v8::Name> property, v8::Local<v8::Value> value, const SwigV8PropertyCallbackInfoVoid &info) {
#endif
  SWIGV8_HANDLESCOPE();
  
  $jslocals
  $jscode
  goto fail;
fail:
  return;
}
%}

/* -----------------------------------------------------------------------------
 * js_function:  template for function wrappers
 *   - $jswrapper:  wrapper function name
 *   - $jslocals:   locals part of wrapper
 *   - $jscode:     code part of wrapper
 * ----------------------------------------------------------------------------- */
%fragment("js_function", "templates")
%{
static SwigV8ReturnValue $jswrapper(const SwigV8Arguments &args) {
  SWIGV8_HANDLESCOPE();
  
  v8::Handle<v8::Value> jsresult;
  $jslocals
  if(args.Length() != $jsargcount) SWIG_exception_fail(SWIG_ERROR, "Illegal number of arguments for $jswrapper.");

  $jscode
  SWIGV8_RETURN(jsresult);

  goto fail;
fail:
  SWIGV8_RETURN(SWIGV8_UNDEFINED());
}
%}

/* -----------------------------------------------------------------------------
 * js_function_dispatcher:  template for a function dispatcher for overloaded functions
 *   - $jswrapper:  wrapper function name
 *   - $jsname:     name of the wrapped function
 *   - $jslocals:   locals part of wrapper
 *   - $jscode:     code part of wrapper
 * ----------------------------------------------------------------------------- */
%fragment("js_function_dispatcher", "templates")
%{
static SwigV8ReturnValue $jswrapper(const SwigV8Arguments &args) {
  SWIGV8_HANDLESCOPE();
  
  v8::Handle<v8::Value> jsresult;
  OverloadErrorHandler errorHandler;
  $jscode

  SWIG_exception_fail(SWIG_ERROR, "Illegal arguments for function $jsname.");

  goto fail;
fail:
  SWIGV8_RETURN(SWIGV8_UNDEFINED());
}
%}

/* -----------------------------------------------------------------------------
 * js_overloaded_function:  template for a overloaded function
 *   - $jswrapper:  wrapper function name
 *   - $jslocals:   locals part of wrapper
 *   - $jscode:     code part of wrapper
 * ----------------------------------------------------------------------------- */
%fragment ("js_overloaded_function", "templates")
%{
static SwigV8ReturnValue $jswrapper(const SwigV8Arguments &args, V8ErrorHandler &SWIGV8_ErrorHandler)
{
  SWIGV8_HANDLESCOPE();
  
  v8::Handle<v8::Value> jsresult;
  $jslocals
  $jscode
  SWIGV8_RETURN(jsresult);

  goto fail;
fail:
  SWIGV8_RETURN(SWIGV8_UNDEFINED());
}
%}

/* -----------------------------------------------------------------------------
 * js_function_dispatch_case:  template for a case used in the function dispatcher
 *   - $jswrapper:  wrapper function name
 *   - $jsargcount: number of arguments of overloaded function
 *   - $jscode:     code part of wrapper
 * ----------------------------------------------------------------------------- */
%fragment ("js_function_dispatch_case", "templates")
%{

  if(args.Length() == $jsargcount) {
    errorHandler.err.Clear();
#if (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < 0x031903)
    jsresult = $jswrapper(args, errorHandler);
    if(errorHandler.err.IsEmpty()) {
      SWIGV8_ESCAPE(jsresult);
    }
#else
    $jswrapper(args, errorHandler);
    if(errorHandler.err.IsEmpty()) {
      return;
    }
#endif
  }
%}

/* -----------------------------------------------------------------------------
 * jsv8_declare_class_template:  template for a class template declaration.
 *   - $jsmangledname:  mangled class name
 * ----------------------------------------------------------------------------- */
%fragment("jsv8_declare_class_template", "templates")
%{
  SWIGV8_ClientData $jsmangledname_clientData;
%}

/* -----------------------------------------------------------------------------
 * jsv8_define_class_template:  template for a class template definition.
 *   - $jsmangledname:  mangled class name
 *   - $jsmangledtype:  mangled class type
 *   - $jsdtor:         the dtor wrapper
 * ----------------------------------------------------------------------------- */
%fragment("jsv8_define_class_template", "templates")
%{
  /* Name: $jsmangledname, Type: $jsmangledtype, Dtor: $jsdtor */
  v8::Handle<v8::FunctionTemplate> $jsmangledname_class = SWIGV8_CreateClassTemplate("$jsmangledname");
  SWIGV8_SET_CLASS_TEMPL($jsmangledname_clientData.class_templ, $jsmangledname_class);
  $jsmangledname_clientData.dtor = $jsdtor;
  if (SWIGTYPE_$jsmangledtype->clientdata == 0) {
    SWIGTYPE_$jsmangledtype->clientdata = &$jsmangledname_clientData;
  }
%}


/* -----------------------------------------------------------------------------
 * jsv8_inherit:  template for an class inherit statement.
 *   - $jsmangledname:  mangled class name
 *   - $jsbaseclass:    mangled name of the base class
 * ----------------------------------------------------------------------------- */
%fragment("jsv8_inherit", "templates")
%{
  if (SWIGTYPE_p$jsbaseclass->clientdata && !(static_cast<SWIGV8_ClientData *>(SWIGTYPE_p$jsbaseclass->clientdata)->class_templ.IsEmpty()))
  {
#if (V8_MAJOR_VERSION-0) < 4 && (SWIG_V8_VERSION < 0x031903)
    $jsmangledname_class->Inherit(static_cast<SWIGV8_ClientData *>(SWIGTYPE_p$jsbaseclass->clientdata)->class_templ);
#else
    $jsmangledname_class->Inherit(
      v8::Local<v8::FunctionTemplate>::New(
        v8::Isolate::GetCurrent(),
        static_cast<SWIGV8_ClientData *>(SWIGTYPE_p$jsbaseclass->clientdata)->class_templ)
     );
#endif

#ifdef SWIGRUNTIME_DEBUG
    printf("Inheritance successful $jsmangledname $jsbaseclass\n");
#endif
  } else {
#ifdef SWIGRUNTIME_DEBUG
    printf("Unable to inherit baseclass, it didn't exist $jsmangledname $jsbaseclass\n");
#endif
  }
%}

/* -----------------------------------------------------------------------------
 * jsv8_create_class_instance:  template for creating an class object.
 *   - $jsname:         class name
 *   - $jsmangledname:  mangled class name
 * ----------------------------------------------------------------------------- */
%fragment("jsv8_create_class_instance", "templates")
%{
  /* Class: $jsname ($jsmangledname) */
  v8::Handle<v8::FunctionTemplate> $jsmangledname_class_0 = SWIGV8_CreateClassTemplate("$jsname");
  $jsmangledname_class_0->SetCallHandler($jsctor);
  $jsmangledname_class_0->Inherit($jsmangledname_class);
  $jsmangledname_class_0->SetHiddenPrototype(true);
  v8::Handle<v8::Object> $jsmangledname_obj = $jsmangledname_class_0->GetFunction();
%}

/* -----------------------------------------------------------------------------
 * jsv8_register_class:  template for a statement that registers a class in a parent namespace.
 *   - $jsname:         class name
 *   - $jsmangledname:  mangled class name
 *   - $jsparent:       mangled name of parent namespace
 * ----------------------------------------------------------------------------- */
%fragment("jsv8_register_class", "templates")
%{
  $jsparent_obj->Set(SWIGV8_SYMBOL_NEW("$jsname"), $jsmangledname_obj);
%}

/* -----------------------------------------------------------------------------
 * jsv8_create_namespace:  template for a statement that creates a namespace object.
 *   - $jsmangledname:  mangled namespace name
 * ----------------------------------------------------------------------------- */
%fragment("jsv8_create_namespace", "templates")
%{
  v8::Handle<v8::Object> $jsmangledname_obj = SWIGV8_OBJECT_NEW();
%}

/* -----------------------------------------------------------------------------
 * jsv8_register_namespace:  template for a statement that registers a namespace in a parent namespace.
 *   - $jsname:         name of namespace
 *   - $jsmangledname:  mangled name of namespace
 *   - $jsparent:       mangled name of parent namespace
 * ----------------------------------------------------------------------------- */
%fragment("jsv8_register_namespace", "templates")
%{
  $jsparent_obj->Set(SWIGV8_SYMBOL_NEW("$jsname"), $jsmangledname_obj);
%}

/* -----------------------------------------------------------------------------
 * jsv8_register_member_function:  template for a statement that registers a member function.
 *   - $jsmangledname:  mangled class name
 *   - $jsname:         name of the function
 *   - $jswrapper:      wrapper of the member function
 * ----------------------------------------------------------------------------- */
%fragment("jsv8_register_member_function", "templates")
%{
  SWIGV8_AddMemberFunction($jsmangledname_class, "$jsname", $jswrapper);
%}

/* -----------------------------------------------------------------------------
 * jsv8_register_member_variable:  template for a statement that registers a member variable.
 *   - $jsmangledname:  mangled class name
 *   - $jsname:         name of the function
 *   - $jsgetter:       wrapper of the getter function
 *   - $jssetter:       wrapper of the setter function
 * ----------------------------------------------------------------------------- */
%fragment("jsv8_register_member_variable", "templates")
%{
  SWIGV8_AddMemberVariable($jsmangledname_class, "$jsname", $jsgetter, $jssetter);
%}

/* -----------------------------------------------------------------------------
 * jsv8_register_static_function:  template for a statement that registers a static class function.
 *   - $jsname:         function name
 *   - $jswrapper:      wrapper of the function
 *   - $jsparent:       mangled name of parent namespace
 *
 * Note: this template is also used for global functions.
 * ----------------------------------------------------------------------------- */
%fragment("jsv8_register_static_function", "templates")
%{
  SWIGV8_AddStaticFunction($jsparent_obj, "$jsname", $jswrapper);
%}

/* -----------------------------------------------------------------------------
 * jsv8_register_static_variable:  template for a statement that registers a static variable.
 *   - $jsname:         variable name
 *   - $jsparent:       mangled name of parent namespace
 *   - $jsgetter:       wrapper of the getter function
 *   - $jssetter:       wrapper of the setter function
 *
 * Note: this template is also used for global variables.
 * ----------------------------------------------------------------------------- */
%fragment("jsv8_register_static_variable", "templates")
%{
  SWIGV8_AddStaticVariable($jsparent_obj, "$jsname", $jsgetter, $jssetter);
%}
