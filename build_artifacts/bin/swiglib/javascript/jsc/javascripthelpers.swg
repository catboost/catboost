%insert(wrapper) %{
 
SWIGINTERN bool JS_registerClass(JSGlobalContextRef context, JSObjectRef parentObject,
                             const char* className,
                             JSClassDefinition* definition) {

    JSStringRef js_className = JSStringCreateWithUTF8CString(className);
    JSObjectRef classObject = JSObjectMake(context, JSClassCreate(definition), NULL);
    JSObjectSetProperty(context, parentObject,
                        js_className, classObject,
                        kJSPropertyAttributeNone, NULL);
    JSStringRelease(js_className);
    
    return true;
}

SWIGINTERN bool JS_registerNamespace(JSGlobalContextRef context, 
                           JSObjectRef namespaceObj, JSObjectRef parentNamespace,
                           const char* name)
{
    JSStringRef js_name = JSStringCreateWithUTF8CString(name);
    JSObjectSetProperty(context, parentNamespace,
                        js_name, namespaceObj,
                        kJSPropertyAttributeNone, NULL);
    JSStringRelease(js_name);
    
    return true;
}


SWIGINTERN bool JS_registerFunction(JSGlobalContextRef context, JSObjectRef object, 
                        const char* functionName, JSObjectCallAsFunctionCallback callback)
{
    JSStringRef js_functionName = JSStringCreateWithUTF8CString(functionName);
    JSObjectSetProperty(context, object, js_functionName,
                        JSObjectMakeFunctionWithCallback(context, js_functionName, callback), 
                        kJSPropertyAttributeNone, NULL);
    JSStringRelease(js_functionName);
    return true;
}

SWIGINTERN bool JS_veto_set_variable(JSContextRef context, JSObjectRef thisObject, JSStringRef propertyName, JSValueRef value, JSValueRef* exception)
{
    char buffer[256];
    char msg[512];
    int res;
    
    JSStringGetUTF8CString(propertyName, buffer, 256);
    res = sprintf(msg, "Tried to write read-only variable: %s.", buffer);
    
    if(res<0) {
      SWIG_exception(SWIG_ERROR, "Tried to write read-only variable.");
    } else {
      SWIG_exception(SWIG_ERROR, msg);
    }
fail:    
    return false;
}

SWIGINTERN JSValueRef JS_CharPtrToJSValue(JSContextRef context, char* cstr) {
  JSValueRef val;
  
  JSStringRef jsstring = JSStringCreateWithUTF8CString((char*) cstr);
  val = JSValueMakeString(context, jsstring);
  JSStringRelease(jsstring);

  return val;
}
%}
