%insert(runtime) "swigrun.swg";
%insert(runtime) "swigerrors.swg";

%insert(runtime) "scirun.swg";

%insert(init) %{
/* Module management functions */

#define SWIG_GetModule(clientdata) SWIG_Scilab_GetModule()
#define SWIG_SetModule(clientdata, pointer) SWIG_Scilab_SetModule(pointer)

SWIGRUNTIME swig_module_info*
SWIG_Scilab_GetModule(void) {
  return NULL;
}

SWIGRUNTIME void
SWIG_Scilab_SetModule(swig_module_info *swig_module) {
}
%}

%insert(init) "swiginit.swg"

%insert(init) %{
SWIGRUNTIME swig_type_info *
SWIG_Scilab_TypeQuery(const char *name) {
  if (SWIG_Module_Initialized()) {
    if (name) {
      return SWIG_TypeQuery(name);
    }
  }
  else {
    SWIG_Error(SWIG_RuntimeError, "the module is not initialized");
  }
  return NULL;
}
%}

%insert(init) %{
#ifdef __cplusplus
extern "C"
#endif
int <module>_Init(SWIG_GatewayParameters) {
  SWIG_InitializeModule(NULL);
  SWIG_CreateScilabVariables(pvApiCtx);
  swig_module_initialized = 1;
%}
