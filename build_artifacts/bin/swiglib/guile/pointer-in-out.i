/* -----------------------------------------------------------------------------
 * pointer-in-out.i
 *
 * Guile typemaps for passing pointers indirectly 
 * ----------------------------------------------------------------------------- */

/* Here is a macro that will define typemaps for passing C pointers indirectly.
  
   TYPEMAP_POINTER_INPUT_OUTPUT(PTRTYPE, SCM_TYPE)

   Supported calling conventions (in this example, PTRTYPE is int *):

   func(int **INPUT)

       Scheme wrapper will take one argument, a wrapped C pointer.
       The address of a variable containing this pointer will be
       passed to the function.

   func(int **INPUT_CONSUMED)

       Likewise, but mark the pointer object as not garbage
       collectable.

   func(int **INPUT_DESTROYED)

       Likewise, but mark the pointer object as destroyed.
       
   func(int **OUTPUT)

       Scheme wrapper will take no arguments.  The address of an int *
       variable will be passed to the function.  The function is
       expected to modify the variable; its value is wrapped and
       becomes an extra return value.  (See the documentation on how
       to deal with multiple values.)
   
   func(int **OUTPUT_NONCOLLECTABLE)

       Likewise, but make the pointer object not garbage collectable.
   
   func(int **BOTH)
   func(int **INOUT)

       This annotation combines INPUT and OUTPUT.

*/

%define TYPEMAP_POINTER_INPUT_OUTPUT(PTRTYPE, SCM_TYPE)

%typemap(in, doc="$NAME is of type <" #SCM_TYPE ">") PTRTYPE *INPUT(PTRTYPE temp)
{
    if (SWIG_ConvertPtr($input, (void **) &temp, $*descriptor, 0)) {
	scm_wrong_type_arg(FUNC_NAME, $argnum, $input);
    }
    $1 = &temp;
}

%typemap(in, doc="$NAME is of type <" #SCM_TYPE "> and is consumed by the function") PTRTYPE *INPUT_CONSUMED(PTRTYPE temp)
{
    if (SWIG_ConvertPtr($input, (void **) &temp, $*descriptor, 0)) {
	scm_wrong_type_arg(FUNC_NAME, $argnum, $input);
    }
    SWIG_Guile_MarkPointerNoncollectable($input);
    $1 = &temp;
}

%typemap(in, doc="$NAME is of type <" #SCM_TYPE "> and is consumed by the function") PTRTYPE *INPUT_DESTROYED(PTRTYPE temp)
{
    if (SWIG_ConvertPtr($input, (void **) &temp, $*descriptor, 0)) {
	scm_wrong_type_arg(FUNC_NAME, $argnum, $input);
    }
    SWIG_Guile_MarkPointerDestroyed($input);
    $1 = &temp;
}

%typemap(in, numinputs=0) PTRTYPE *OUTPUT(PTRTYPE temp),
                          PTRTYPE *OUTPUT_NONCOLLECTABLE(PTRTYPE temp)
     "$1 = &temp;";

%typemap(argout, doc="<" #SCM_TYPE ">") PTRTYPE *OUTPUT
     "SWIG_APPEND_VALUE(SWIG_NewPointerObj(*$1, $*descriptor, 1));"; 

%typemap(argout, doc="<" #SCM_TYPE ">") PTRTYPE *OUTPUT_NONCOLLECTABLE
     "SWIG_APPEND_VALUE(SWIG_NewPointerObj(*$1, $*descriptor, 0));"; 

%typemap(in) PTRTYPE *BOTH = PTRTYPE *INPUT;
%typemap(argout) PTRTYPE *BOTH = PTRTYPE *OUTPUT;
%typemap(in) PTRTYPE *INOUT = PTRTYPE *INPUT;
%typemap(argout) PTRTYPE *INOUT = PTRTYPE *OUTPUT;

/* As a special convenience measure, also attach docs involving
   SCM_TYPE to the standard pointer typemaps */

%typemap(in, doc="$NAME is of type <" #SCM_TYPE ">") PTRTYPE {
  if (SWIG_ConvertPtr($input, (void **) &$1, $descriptor, 0))
    scm_wrong_type_arg(FUNC_NAME, $argnum, $input);
}

%typemap(out, doc="<" #SCM_TYPE ">") PTRTYPE {
    $result = SWIG_NewPointerObj ($1, $descriptor, $owner);
}

%enddef
