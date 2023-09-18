/* ------------------------------------------------------------
 *  Typemap specializations for Perl
 * ------------------------------------------------------------ */

/* ------------------------------------------------------------
 *  Fragment section
 * ------------------------------------------------------------ */

/*
  in Perl we need to pass the CPerlObj value, sometimes, so, we define
  the decl/call macros as needed.
*/

#define SWIG_AS_DECL_ARGS SWIG_PERL_DECL_ARGS_2
#define SWIG_AS_CALL_ARGS SWIG_PERL_CALL_ARGS_2

#define SWIG_FROM_DECL_ARGS  SWIG_PERL_DECL_ARGS_1
#define SWIG_FROM_CALL_ARGS  SWIG_PERL_CALL_ARGS_1


/* Include fundamental fragment definitions */
%include <typemaps/fragments.swg>

/* Look for user fragments file. */
%include <perlfragments.swg>

/* Perl fragments for primitive types */
%include <perlprimtypes.swg>

/* Perl fragments for char* strings */
%include <perlstrings.swg>


/* ------------------------------------------------------------
 *  Unified typemap section
 * ------------------------------------------------------------ */

/* director support in Perl is experimental */
#ifndef SWIG_DIRECTOR_TYPEMAPS
#define SWIG_DIRECTOR_TYPEMAPS
#endif


/* Perl types */
#define SWIG_Object                      SV *
#define VOID_Object                      &PL_sv_undef

/* Perl $shadow flag */
#define %newpointer_flags                $shadow
#define %newinstance_flags               $shadow


/* Complete overload of the output/constant/exception macros */

/* output */
%define %set_output(obj) $result = obj; argvi++ %enddef

/* append output */
%define %append_output(obj)
if (argvi >= items) EXTEND(sp, argvi+1);
%set_output(obj) %enddef

/* variable output */
%define %set_varoutput(obj) sv_setsv($result,obj)  %enddef

/* constant */
%define %set_constant(name, obj) %begin_block
  SV *sv = get_sv((char*) SWIG_prefix name, TRUE | 0x2 | GV_ADDMULTI);
  sv_setsv(sv, obj);
  SvREADONLY_on(sv);
%end_block %enddef

/* raise exception */
%define %raise(obj, type, desc) sv_setsv(get_sv("@", GV_ADD), obj); SWIG_fail %enddef

/* For directors to raise/throw the original exception */
%typemap(throws) Swig::DirectorException
%{ sv_setsv(ERRSV, $1.getNative()); SWIG_fail; %}

/* Include the unified typemap library */
%include <typemaps/swigtypemaps.swg>

/*  ------------------------------------------------------------
 *  Perl extra typemaps / typemap overrides
 * ------------------------------------------------------------ */

%typemap(varout,type="$1_descriptor") SWIGTYPE *, SWIGTYPE []  
    "sv_setiv(SvRV($result),PTR2IV($1));"; 

%typemap(varout,type="$1_descriptor") SWIGTYPE &
    "sv_setiv(SvRV($result),PTR2IV(&$1));"; 

%typemap(varout,type="$1_descriptor") SWIGTYPE &&
    "sv_setiv(SvRV($result),PTR2IV(&$1));"; 

%typemap(varout,type="$&1_descriptor") SWIGTYPE
    "sv_setiv(SvRV($result), PTR2IV(&$1));";

%typemap(varout,type="$1_descriptor") SWIGTYPE (CLASS::*) {
  SWIG_MakePackedObj($result, (void *) &$1, sizeof($1), $1_descriptor);
}

%typemap(varout) SWIGTYPE *const = SWIGTYPE *;

