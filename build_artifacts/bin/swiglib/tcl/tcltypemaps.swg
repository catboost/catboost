/* ------------------------------------------------------------
 *  Typemap specializations for Tcl
 * ------------------------------------------------------------ */

/* ------------------------------------------------------------
 *  Fragment section
 * ------------------------------------------------------------ */

/*
  In Tcl we need to pass the interp value, so we define the decl/call
  macros as needed.
*/

#define SWIG_AS_DECL_ARGS SWIG_TCL_DECL_ARGS_2
#define SWIG_AS_CALL_ARGS SWIG_TCL_CALL_ARGS_2


/* Include fundamental fragment definitions */
%include <typemaps/fragments.swg>

/* Look for user fragments file. */
%include <tclfragments.swg>

/* Tcl fragments for primitive types */
%include <tclprimtypes.swg>

/* Tcl fragments for char* strings */
%include <tclstrings.swg>


/* ------------------------------------------------------------
 *  Unified typemap section
 * ------------------------------------------------------------ */

/* No director support in Tcl */
#ifdef SWIG_DIRECTOR_TYPEMAPS
#undef SWIG_DIRECTOR_TYPEMAPS
#endif


/* Tcl types */
#define SWIG_Object                      Tcl_Obj *

/* Overload of the output/constant/exception handling */

/* output */
#define %set_output(obj)                 Tcl_SetObjResult(interp,obj)

/* append output */
#define %append_output(obj)              Tcl_ListObjAppendElement(interp,Tcl_GetObjResult(interp),obj)

/* set constant */
#define SWIG_SetConstant(name, obj)      SWIG_Tcl_SetConstantObj(interp, name, obj)

/* raise */
#define SWIG_Raise(obj,type,desc)        SWIG_Tcl_SetErrorObj(interp,type,obj)


/* Include the unified typemap library */
%include <typemaps/swigtypemaps.swg>


/* ------------------------------------------------------------
 *  Tcl extra typemaps / typemap overrides
 * ------------------------------------------------------------ */

#if 1
// Old 1.3.25 typemaps needed to avoid premature object deletion
%typemap(out,noblock=1) SWIGTYPE *INSTANCE, SWIGTYPE &INSTANCE, SWIGTYPE &&INSTANCE, SWIGTYPE INSTANCE[] {
  Tcl_SetObjResult(interp, SWIG_NewInstanceObj( %as_voidptr($1), $1_descriptor,0));
}

%typemap(out) SWIGTYPE *DYNAMIC, SWIGTYPE &DYNAMIC {
  swig_type_info *ty = SWIG_TypeDynamicCast($1_descriptor,%as_voidptrptr(&$1));
  Tcl_SetObjResult(interp,SWIG_NewInstanceObj(%as_voidptr($1), ty,0));
}

#endif

%typemap(throws,noblock=1) SWIGTYPE CLASS {
  SWIG_set_result(SWIG_NewInstanceObj(%as_voidptr(SWIG_new_copy($1, $1_ltype)), $&1_descriptor, 1));
  SWIG_fail;
}

%typemap(out)    SWIGTYPE    =  SWIGTYPE INSTANCE;
%typemap(out)    SWIGTYPE *  =  SWIGTYPE *INSTANCE;
%typemap(out)    SWIGTYPE *const  =  SWIGTYPE *;
%typemap(out)    SWIGTYPE &  =  SWIGTYPE &INSTANCE;
%typemap(out)    SWIGTYPE && =  SWIGTYPE &&INSTANCE;
%typemap(out)    SWIGTYPE [] =  SWIGTYPE INSTANCE[];
%typemap(varout) SWIGTYPE    =  SWIGTYPE INSTANCE;
