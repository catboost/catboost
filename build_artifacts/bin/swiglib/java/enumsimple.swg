/* -----------------------------------------------------------------------------
 * enumsimple.swg
 *
 * This file provides backwards compatible enum wrapping. SWIG versions 1.3.21
 * and earlier wrapped global enums with constant integers in the module class
 * or Constants interface. Enums declared within a C++ class were wrapped by
 * constant integers in the Java proxy class.
 * ----------------------------------------------------------------------------- */

// const enum SWIGTYPE & typemaps
%typemap(jni) const enum SWIGTYPE & "jint"
%typemap(jtype) const enum SWIGTYPE & "int"
%typemap(jstype) const enum SWIGTYPE & "int"

%typemap(in) const enum SWIGTYPE & ($*1_ltype temp)
%{ temp = ($*1_ltype)$input; 
   $1 = &temp; %}
%typemap(out) const enum SWIGTYPE &  %{ $result = (jint)*$1; %}

%typemap(directorout,warning=SWIGWARN_TYPEMAP_THREAD_UNSAFE_MSG) const enum SWIGTYPE &
%{ static $*1_ltype temp = ($*1_ltype)$input; 
   $result = &temp; %}
%typemap(directorin, descriptor="I") const enum SWIGTYPE & "$input = (jint)$1;"
%typemap(javadirectorin) const enum SWIGTYPE & "$jniinput"
%typemap(javadirectorout) const enum SWIGTYPE & "$javacall"

%typecheck(SWIG_TYPECHECK_INT32) const enum SWIGTYPE & ""

%typemap(throws) const enum SWIGTYPE &
%{ (void)$1;
   SWIG_JavaThrowException(jenv, SWIG_JavaRuntimeException, "C++ $1_type exception thrown"); %}

%typemap(javain) const enum SWIGTYPE & "$javainput"
%typemap(javaout) const enum SWIGTYPE & {
    return $jnicall;
  }


// enum SWIGTYPE typemaps
%typemap(jni) enum SWIGTYPE "jint"
%typemap(jtype) enum SWIGTYPE "int"
%typemap(jstype) enum SWIGTYPE "int"

%typemap(in) enum SWIGTYPE  %{ $1 = ($1_ltype)$input; %}
%typemap(out) enum SWIGTYPE  %{ $result = (jint)$1; %}

%typemap(directorout) enum SWIGTYPE  %{ $result = ($1_ltype)$input; %}
%typemap(directorin, descriptor="I") enum SWIGTYPE "$input = (jint) $1;"
%typemap(javadirectorin) enum SWIGTYPE "$jniinput"
%typemap(javadirectorout) enum SWIGTYPE "$javacall"

%typecheck(SWIG_TYPECHECK_INT32) enum SWIGTYPE ""

%typemap(throws) enum SWIGTYPE
%{ (void)$1;
   SWIG_JavaThrowException(jenv, SWIG_JavaRuntimeException, "C++ $1_type exception thrown"); %}

%typemap(javain) enum SWIGTYPE "$javainput"
%typemap(javaout) enum SWIGTYPE {
    return $jnicall;
  }

%typemap(javaclassmodifiers) enum SWIGTYPE ""
%typemap(javabase)           enum SWIGTYPE ""
%typemap(javacode)           enum SWIGTYPE ""
%typemap(javaimports)        enum SWIGTYPE ""
%typemap(javainterfaces)     enum SWIGTYPE ""
%typemap(javabody)           enum SWIGTYPE ""

%javaenum(simple);

