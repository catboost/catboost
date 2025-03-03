%insert(init) %{
SWIGRUNTIME void
SWIG_JSC_SetModule(swig_module_info *swig_module) {}

SWIGRUNTIME swig_module_info *
SWIG_JSC_GetModule(void) {
  return 0;
}

#define SWIG_GetModule(clientdata)                SWIG_JSC_GetModule()
#define SWIG_SetModule(clientdata, pointer)       SWIG_JSC_SetModule(pointer)
%}

%insert(init) "swiginit.swg"

%fragment ("js_initializer_define", "templates") %{
#define SWIGJSC_INIT $jsname_initialize
%}

// Open the initializer function
%insert(init)
%{

#ifdef __cplusplus
extern "C" {
#endif

bool SWIGJSC_INIT (JSGlobalContextRef context, JSObjectRef *exports) {
    SWIG_InitializeModule(0);
%}

/* -----------------------------------------------------------------------------
 * js_initializer:  template for the module initializer function
 *   - $jsname:                   module name
 *   - $jscreatenamespaces:       part with code for creating namespace objects
 *   - $jscreateclasses:          part with code for creating classes
 *   - $jsregisternamespaces:     part with code for registration of namespaces
 * ----------------------------------------------------------------------------- */
%fragment ("js_initializer", "templates") %{
    /* Initialize the base swig type object */
    _SwigObject_objectDefinition.staticFunctions = _SwigObject_functions;
    _SwigObject_objectDefinition.staticValues = _SwigObject_values;
    _SwigObject_classRef = JSClassCreate(&_SwigObject_objectDefinition);

    /* Initialize the PackedData class */
    _SwigPackedData_objectDefinition.staticFunctions = _SwigPackedData_functions;
    _SwigPackedData_objectDefinition.staticValues = _SwigPackedData_values;
    _SwigPackedData_objectDefinition.finalize = _wrap_SwigPackedData_delete;
    _SwigPackedData_classRef = JSClassCreate(&_SwigPackedData_objectDefinition);

    /* Create objects for namespaces */
    $jscreatenamespaces

    /* Register classes */
    $jsregisterclasses

    /* Register namespaces */
    $jsregisternamespaces

    *exports = exports_object;

    return true;
}
#ifdef __cplusplus
}
#endif
%}
