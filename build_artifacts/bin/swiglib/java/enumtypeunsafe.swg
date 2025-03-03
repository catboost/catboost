/* -----------------------------------------------------------------------------
 * enumtypeunsafe.swg
 *
 * Include this file in order for C/C++ enums to be wrapped by integers values.
 * Each enum has an equivalent class named after the enum and the enum items are
 * wrapped by constant integers within this class. The enum items are not
 * typesafe as they are all integers.
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

// '$static' will be replaced with either 'static' or nothing depending on whether the enum is an inner Java class or not
%typemap(javaclassmodifiers) enum SWIGTYPE "public final $static class"
%typemap(javabase)           enum SWIGTYPE ""
%typemap(javacode)           enum SWIGTYPE ""
%typemap(javaimports)        enum SWIGTYPE ""
%typemap(javainterfaces)     enum SWIGTYPE ""
%typemap(javabody)           enum SWIGTYPE ""

%javaenum(typeunsafe);

