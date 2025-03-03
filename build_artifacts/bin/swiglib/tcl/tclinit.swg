/* ------------------------------------------------------------
 * The start of the Tcl initialization function 
 * ------------------------------------------------------------ */

%insert(init) "swiginit.swg"

/* This initialization code exports the module initialization function */

%header %{

#ifdef __cplusplus
extern "C" {
#endif
#ifdef MAC_TCL
#pragma export on
#endif
SWIGEXPORT int SWIG_init(Tcl_Interp *);
#ifdef MAC_TCL
#pragma export off
#endif
#ifdef __cplusplus
}
#endif

/* Compatibility version for TCL stubs */
#ifndef SWIG_TCL_STUBS_VERSION
#define SWIG_TCL_STUBS_VERSION "8.1"
#endif

%}

%init %{
#ifdef __cplusplus
extern "C" {   
#endif

/* -----------------------------------------------------------------------------
 * constants/methods manipulation
 * ----------------------------------------------------------------------------- */

/* Install Constants */

SWIGINTERN void
SWIG_Tcl_InstallConstants(Tcl_Interp *interp, swig_const_info constants[]) {
  size_t i;
  Tcl_Obj *obj;

  if (!swigconstTableinit) {
    Tcl_InitHashTable(&swigconstTable, TCL_STRING_KEYS);
    swigconstTableinit = 1;
  }
  for (i = 0; constants[i].type; i++) {
    switch(constants[i].type) {
    case SWIG_TCL_POINTER:
      obj = SWIG_NewPointerObj(constants[i].pvalue, *(constants[i]).ptype,0);
      break;
    case SWIG_TCL_BINARY:
      obj = SWIG_NewPackedObj(constants[i].pvalue, constants[i].lvalue, *(constants[i].ptype));
      break;
    default:
      obj = 0;
      break;
    }
    if (obj) {
      SWIG_Tcl_SetConstantObj(interp, constants[i].name, obj);
    }
  }
}

/* Create fast method lookup tables */

SWIGINTERN void
SWIG_Tcl_InstallMethodLookupTables(void) {
  size_t i;

  for (i = 0; i < swig_module.size; ++i) {
    swig_type_info *type = swig_module.type_initial[i];
    if (type->clientdata) {
      swig_class* klass = (swig_class*) type->clientdata;
      swig_method* meth;
      Tcl_InitHashTable(&(klass->hashtable), TCL_STRING_KEYS);
      for (meth = klass->methods; meth && meth->name; ++meth) {
        int newEntry;
        Tcl_HashEntry* hashentry = Tcl_CreateHashEntry(&(klass->hashtable), meth->name, &newEntry);
        Tcl_SetHashValue(hashentry, (ClientData)meth->method);
      }
    }
  }
}

#ifdef __cplusplus
}
#endif

/* -----------------------------------------------------------------------------*
 *  Partial Init method
 * -----------------------------------------------------------------------------*/

SWIGEXPORT int SWIG_init(Tcl_Interp *interp) {
  size_t i;
  if (interp == 0) return TCL_ERROR;
#ifdef USE_TCL_STUBS
  /* (char*) cast is required to avoid compiler warning/error for Tcl < 8.4. */
  if (Tcl_InitStubs(interp, (char*)SWIG_TCL_STUBS_VERSION, 0) == NULL) {
    return TCL_ERROR;
  }
#endif  
#ifdef USE_TK_STUBS
  /* (char*) cast is required to avoid compiler warning/error. */
  if (Tk_InitStubs(interp, (char*)SWIG_TCL_STUBS_VERSION, 0) == NULL) {
    return TCL_ERROR;
  }
#endif

  Tcl_PkgProvide(interp, (char*)SWIG_name, (char*)SWIG_version);
  
#ifdef SWIG_namespace
  Tcl_Eval(interp, "namespace eval " SWIG_namespace " { }");
#endif
  
  SWIG_InitializeModule((void *) interp);
  SWIG_PropagateClientData();
  
  for (i = 0; swig_commands[i].name; i++) {
    Tcl_CreateObjCommand(interp, (char *) swig_commands[i].name, (swig_wrapper_func) swig_commands[i].wrapper,
			 swig_commands[i].clientdata, NULL);
  }
  for (i = 0; swig_variables[i].name; i++) {
    Tcl_SetVar(interp, (char *) swig_variables[i].name, (char *) "", TCL_GLOBAL_ONLY);
    Tcl_TraceVar(interp, (char *) swig_variables[i].name, TCL_TRACE_READS | TCL_GLOBAL_ONLY, 
		 (Tcl_VarTraceProc *) swig_variables[i].get, (ClientData) swig_variables[i].addr);
    Tcl_TraceVar(interp, (char *) swig_variables[i].name, TCL_TRACE_WRITES | TCL_GLOBAL_ONLY, 
		 (Tcl_VarTraceProc *) swig_variables[i].set, (ClientData) swig_variables[i].addr);
  }

  SWIG_Tcl_InstallConstants(interp, swig_constants);
  SWIG_Tcl_InstallMethodLookupTables();

%}

/* Note: the initialization function is closed after all code is generated */
