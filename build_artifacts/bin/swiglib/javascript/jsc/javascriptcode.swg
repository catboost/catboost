/* -----------------------------------------------------------------------------
 * js_ctor:  template for wrapping a ctor.
 *   - $jswrapper:        wrapper of called ctor
 *   - $jslocals:         locals part of wrapper
 *   - $jscode:           code part of wrapper
 *   - $jsargcount:       number of arguments
 *   - $jsmangledtype:    mangled type of class
 * ----------------------------------------------------------------------------- */
%fragment ("js_ctor", "templates")
%{
static JSObjectRef $jswrapper(JSContextRef context, JSObjectRef thisObject, size_t argc, const JSValueRef argv[], JSValueRef* exception)
{
    $jslocals
    if(argc != $jsargcount) SWIG_exception_fail(SWIG_ERROR, "Illegal number of arguments for $jswrapper.");

    $jscode
    return SWIG_JSC_NewPointerObj(context, result, SWIGTYPE_$jsmangledtype, SWIG_POINTER_OWN);
    goto fail;
    fail:
    return NULL;
}
%}

/* -----------------------------------------------------------------------------
 * js_veto_ctor:  a vetoing ctor for abstract classes
 *   - $jswrapper:        name of wrapper
 *   - $jsname:           class name
 * ----------------------------------------------------------------------------- */
%fragment ("js_veto_ctor", "templates")
%{
static JSObjectRef $jswrapper(JSContextRef context, JSObjectRef ctorObject,
    size_t argc, const JSValueRef argv[], JSValueRef* exception)
{
    SWIG_exception(SWIG_ERROR, "Class $jsname can not be instantiated");
fail:
    return 0;
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
static JSObjectRef $jswrapper(JSContextRef context, JSObjectRef ctorObject,
    size_t argc, const JSValueRef argv[], JSValueRef* exception)
{
    JSObjectRef thisObject = NULL;

    // switch all cases by means of series of if-returns.
    $jsdispatchcases

    // default:
    SWIG_exception_fail(SWIG_ERROR, "Illegal arguments for construction of $jsname");

    fail:
    return thisObject;
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
%fragment ("js_overloaded_ctor", "templates")
%{
static JSObjectRef $jswrapper(JSContextRef context, JSObjectRef thisObject, size_t argc, const JSValueRef argv[], JSValueRef* exception)
{
    $jslocals
    $jscode
    return SWIG_JSC_NewPointerObj(context, result, SWIGTYPE_$jsmangledtype, SWIG_POINTER_OWN);

    goto fail;
    fail:
    return NULL;
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
  if(argc == $jsargcount) {
    thisObject = $jswrapper(context, NULL, argc, argv, exception);
    if(thisObject != NULL) { *exception=0; return thisObject; } /* reset exception and return */
  }
%}


/* -----------------------------------------------------------------------------
 * js_dtor:  template for a destructor wrapper
 *   - $jsmangledname:  mangled class name
 *   - $jstype:         class type
 * ----------------------------------------------------------------------------- */
%fragment ("js_dtor", "templates")
%{
static void $jswrapper(JSObjectRef thisObject)
{
  SwigPrivData* t = (SwigPrivData*) JSObjectGetPrivate(thisObject);
  if(t) {
    if (t->swigCMemOwn) {
      free (($jstype)t->swigCObject);
    }
    JSObjectSetPrivate(thisObject, NULL);
    free(t);
  }
}
%}

/* -----------------------------------------------------------------------------
 * js_dtor:  template for a destructor wrapper
 *   - $jsmangledname:  mangled class name
 *   - $jstype:         class type
 *   - ${destructor_action}: The custom destructor action to invoke.
 * ----------------------------------------------------------------------------- */
%fragment ("js_dtoroverride", "templates")
%{
static void $jswrapper(JSObjectRef thisObject)
{
  SwigPrivData* t = (SwigPrivData*) JSObjectGetPrivate(thisObject);
  if(t) {
    if (t->swigCMemOwn) {
      $jstype arg1 = ($jstype)t->swigCObject;
      ${destructor_action}
    }
    /* remove the private data to make sure that it isn't accessed elsewhere */
    JSObjectSetPrivate(thisObject, NULL);
    free(t);
  }
}
%}

/* -----------------------------------------------------------------------------
 * js_getter:  template for getter function wrappers
 *   - $jswrapper:  wrapper function name
 *   - $jslocals:   locals part of wrapper
 *   - $jscode:     code part of wrapper
 * ----------------------------------------------------------------------------- */
%fragment ("js_getter", "templates")
%{
static JSValueRef $jswrapper(JSContextRef context, JSObjectRef thisObject, JSStringRef propertyName, JSValueRef* exception)
{
    $jslocals
    JSValueRef jsresult;

    $jscode
    return jsresult;

    goto fail;
    fail:
    return JSValueMakeUndefined(context);
}
%}

/* -----------------------------------------------------------------------------
 * js_setter:  template for setter function wrappers
 *   - $jswrapper:  wrapper function name
 *   - $jslocals:   locals part of wrapper
 *   - $jscode:     code part of wrapper
 * ----------------------------------------------------------------------------- */
%fragment ("js_setter", "templates")
%{
static bool $jswrapper(JSContextRef context, JSObjectRef thisObject, JSStringRef propertyName, JSValueRef value, JSValueRef* exception)
{
    $jslocals
    $jscode

    return true;

    goto fail;
    fail:
    return false;
}
%}

/* -----------------------------------------------------------------------------
 * js_function:  template for function wrappers
 *   - $jswrapper:  wrapper function name
 *   - $jslocals:   locals part of wrapper
 *   - $jscode:     code part of wrapper
 * ----------------------------------------------------------------------------- */
%fragment ("js_function", "templates")
%{
static JSValueRef $jswrapper(JSContextRef context, JSObjectRef function, JSObjectRef thisObject, size_t argc, const JSValueRef argv[], JSValueRef* exception)
{
  $jslocals
  JSValueRef jsresult;

  if(argc != $jsargcount) SWIG_exception_fail(SWIG_ERROR, "Illegal number of arguments for $jswrapper.");

  $jscode
  return jsresult;

  goto fail;
  fail:
  return JSValueMakeUndefined(context);
}
%}

/* -----------------------------------------------------------------------------
 * js_function_dispatcher:  template for a function dispatcher for overloaded functions
 *   - $jswrapper:  wrapper function name
 *   - $jsname:     name of the wrapped function
 *   - $jslocals:   locals part of wrapper
 *   - $jscode:     code part of wrapper
 * ----------------------------------------------------------------------------- */
%fragment ("js_function_dispatcher", "templates")
%{
static JSValueRef $jswrapper(JSContextRef context, JSObjectRef function, JSObjectRef thisObject, size_t argc, const JSValueRef argv[], JSValueRef* exception)
{
  $jslocals
  JSValueRef jsresult;
  int res;
  $jscode

  SWIG_exception_fail(SWIG_ERROR, "Illegal arguments for function $jsname.");
  return jsresult;

  goto fail;
  fail:
  return JSValueMakeUndefined(context);
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
static int $jswrapper(JSContextRef context, JSObjectRef function, JSObjectRef thisObject, size_t argc, const JSValueRef argv[], JSValueRef* exception, JSValueRef* p_result)
{
  $jslocals
  JSValueRef jsresult;

  if(argc != $jsargcount) SWIG_exception_fail(SWIG_ERROR, "Illegal number of arguments for $jswrapper.");

  $jscode
  *p_result = jsresult;
  return SWIG_OK;

  goto fail;
  fail:
  return SWIG_TypeError;
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
  if(argc == $jsargcount) {
     res = $jswrapper(context, function, thisObject, argc, argv, exception, &jsresult);
     if(res == SWIG_OK) { *exception = 0; return jsresult; }
  }
%}

/* -----------------------------------------------------------------------------
 * jsc_variable_declaration:  template for a variable table entry
 *   - $jsname:       name of the variable
 *   - $jsgetter:     wrapper of getter function
 *   - $jssetter:     wrapper of setter function
 * ----------------------------------------------------------------------------- */
%fragment ("jsc_variable_declaration", "templates")
%{
  {"$jsname", $jsgetter, $jssetter, kJSPropertyAttributeNone},
%}


/* -----------------------------------------------------------------------------
 * jsc_function_declaration:  template for a function table entry
 *   - $jsname:       name of the variable
 *   - $jswrapper:    wrapper function
 * ----------------------------------------------------------------------------- */
%fragment ("jsc_function_declaration", "templates")
%{
  {"$jsname", $jswrapper, kJSPropertyAttributeNone},
%}

/* -----------------------------------------------------------------------------
 * jsc_classtemplate_declaration:  template for a namespace declaration
 *   - $jsmangledname:      mangled class name
 * ----------------------------------------------------------------------------- */
%fragment ("jsc_class_declaration", "templates")
%{
static JSClassDefinition $jsmangledname_classDefinition;

static JSClassDefinition $jsmangledname_objectDefinition;

static JSClassRef $jsmangledname_classRef;
%}

/* -----------------------------------------------------------------------------
 * jsc_class_tables:  template for a namespace declaration
 *   - $jsmangledname:            mangled class name
 *   - $jsstaticclassvariables:   list of static variable entries
 *   - $jsstaticclassfunctions:   list of static function entries
 *   - $jsclassvariables:         list of member variable entries
 *   - $jsclassfunctions:         list of member function entries
 * ----------------------------------------------------------------------------- */
%fragment ("jsc_class_tables", "templates")
%{
static JSStaticValue $jsmangledname_staticValues[] = {
  $jsstaticclassvariables
  { 0, 0, 0, 0 }
};

static JSStaticFunction $jsmangledname_staticFunctions[] = {
  $jsstaticclassfunctions
  { 0, 0, 0 }
};

static JSStaticValue $jsmangledname_values[] = {
  $jsclassvariables
  { 0, 0, 0, 0 }
};

static JSStaticFunction $jsmangledname_functions[] = {
  $jsclassfunctions
  { 0, 0, 0 }
};
%}

/* -----------------------------------------------------------------------------
 * jsc_define_class_template:  template for defining a class template
 *   - $jsmangledname:            mangled class name
 *   - $jsmangledtype:            mangled class type
 *   - $jsctor:                   wrapper of ctor
 *   - $jsbaseclass:              mangled name of base class
 * ----------------------------------------------------------------------------- */
%fragment ("jsc_class_definition", "templates")
%{
  $jsmangledname_classDefinition.staticFunctions = $jsmangledname_staticFunctions;
  $jsmangledname_classDefinition.staticValues = $jsmangledname_staticValues;
  $jsmangledname_classDefinition.callAsConstructor = $jsctor;
  $jsmangledname_objectDefinition.finalize = $jsdtor;
  $jsmangledname_objectDefinition.staticValues = $jsmangledname_values;
  $jsmangledname_objectDefinition.staticFunctions = $jsmangledname_functions;
  $jsclass_inheritance
  $jsmangledname_classRef = JSClassCreate(&$jsmangledname_objectDefinition);
  SWIGTYPE_$jsmangledtype->clientdata = $jsmangledname_classRef;
%}

%fragment ("jsc_class_inherit", templates)
%{
    if (SWIGTYPE_p$jsbaseclassmangled != NULL) {
      $jsmangledname_objectDefinition.parentClass = (JSClassRef) SWIGTYPE_p$jsbaseclassmangled->clientdata;
    }
%}

%fragment ("jsc_class_noinherit", templates)
%{
    $jsmangledname_objectDefinition.parentClass = _SwigObject_classRef;
%}


/* -----------------------------------------------------------------------------
 * jsc_register_class:  template for registration of a class
 *   - $jsname:                   class name
 *   - $jsmangledname:            mangled class name
 *   - $jsnspace:                 mangled name of namespace
 * ----------------------------------------------------------------------------- */
%fragment ("jsc_class_registration", "templates")
%{
  JS_registerClass(context, $jsnspace_object, "$jsname", &$jsmangledname_classDefinition);
%}


/* -----------------------------------------------------------------------------
 * jsc_nspace_declaration:  template for a namespace declaration
 *   - $jsnspace:           mangled name of the namespace
 *   - $jsglobalvariables:  list of variable entries
 *   - $jsglobalfunctions:  list of function entries
 * ----------------------------------------------------------------------------- */
%fragment ("jsc_nspace_declaration", "templates")
%{
static JSStaticValue $jsnspace_values[] = {
  $jsglobalvariables
  { 0, 0, 0, 0 }
};

static JSStaticFunction $jsnspace_functions[] = {
  $jsglobalfunctions
  { 0, 0, 0 }
};

static JSClassDefinition $jsnspace_classDefinition;
static JSObjectRef $jsmangledname_object;
%}

/* -----------------------------------------------------------------------------
 * jsc_nspace_definition:  template for definition of a namespace object
 *   - $jsmangledname:            mangled name of namespace
 * ----------------------------------------------------------------------------- */
%fragment ("jsc_nspace_definition", "templates")
%{
  $jsmangledname_classDefinition.staticFunctions = $jsmangledname_functions;
  $jsmangledname_classDefinition.staticValues = $jsmangledname_values;
  $jsmangledname_object = JSObjectMake(context, JSClassCreate(&$jsmangledname_classDefinition), NULL);
%}

/* -----------------------------------------------------------------------------
 * jsc_nspace_registration:  template for registration of a namespace object
 *   - $jsname:                   name of namespace
 *   - $jsmangledname:            mangled name of namespace
 *   - $jsparent:                 mangled name of parent namespace
 * ----------------------------------------------------------------------------- */
%fragment ("jsc_nspace_registration", "templates")
%{
  JS_registerNamespace(context, $jsmangledname_object, $jsparent_object, "$jsname");
%}
