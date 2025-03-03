/* -----------------------------------------------------------------------------
 * enumsimple.swg
 *
 * This file provides backwards compatible enum wrapping. SWIG versions 1.3.21
 * and earlier wrapped global enums with constant integers in the module
 * class. Enums declared within a C++ class were wrapped by constant integers
 * in the C# proxy class.
 * ----------------------------------------------------------------------------- */

// const enum SWIGTYPE & typemaps
%typemap(ctype) const enum SWIGTYPE & "int"
%typemap(imtype) const enum SWIGTYPE & "int"
%typemap(cstype) const enum SWIGTYPE & "int"

%typemap(in) const enum SWIGTYPE & ($*1_ltype temp)
%{ temp = ($*1_ltype)$input; 
   $1 = &temp; %}
%typemap(out) const enum SWIGTYPE & %{ $result = (int)*$1; %}

%typemap(directorout,warning=SWIGWARN_TYPEMAP_THREAD_UNSAFE_MSG) const enum SWIGTYPE &
%{ static $*1_ltype temp = ($*1_ltype)$input; 
   $result = &temp; %}
%typemap(directorin) const enum SWIGTYPE & "$input = (int)$1;"
%typemap(csdirectorin) const enum SWIGTYPE & "$iminput"
%typemap(csdirectorout) const enum SWIGTYPE & "$cscall"

%typecheck(SWIG_TYPECHECK_INT32) const enum SWIGTYPE & ""

%typemap(throws, canthrow=1) const enum SWIGTYPE &
%{ (void)$1;
   SWIG_CSharpSetPendingException(SWIG_CSharpApplicationException, "C++ $1_type exception thrown");
   return $null; %}

%typemap(csin) const enum SWIGTYPE & "$csinput"
%typemap(csout, excode=SWIGEXCODE) const enum SWIGTYPE & {
    int ret = $imcall;$excode
    return ret;
  }

%typemap(csvarout, excode=SWIGEXCODE2) const enum SWIGTYPE & %{
    get {
      int ret = $imcall;$excode
      return ret;
    } %}


// enum SWIGTYPE typemaps
%typemap(ctype) enum SWIGTYPE "int"
%typemap(imtype) enum SWIGTYPE "int"
%typemap(cstype) enum SWIGTYPE "int"

%typemap(in) enum SWIGTYPE %{ $1 = ($1_ltype)$input; %}
%typemap(out) enum SWIGTYPE %{ $result = (int)$1; %}

%typemap(directorout) enum SWIGTYPE  %{ $result = ($1_ltype)$input; %}
%typemap(directorin) enum SWIGTYPE "$input = (int)$1;"
%typemap(csdirectorin) enum SWIGTYPE "$iminput"
%typemap(csdirectorout) enum SWIGTYPE "$cscall"

%typecheck(SWIG_TYPECHECK_INT32) enum SWIGTYPE ""

%typemap(throws, canthrow=1) enum SWIGTYPE
%{ (void)$1;
   SWIG_CSharpSetPendingException(SWIG_CSharpApplicationException, "C++ $1_type exception thrown");
   return $null; %}

%typemap(csin) enum SWIGTYPE "$csinput"
%typemap(csout, excode=SWIGEXCODE) enum SWIGTYPE {
    int ret = $imcall;$excode
    return ret;
  }

%typemap(csvarout, excode=SWIGEXCODE2) enum SWIGTYPE %{
    get {
      int ret = $imcall;$excode
      return ret;
    } %}

%typemap(csbase)           enum SWIGTYPE ""
%typemap(csclassmodifiers) enum SWIGTYPE ""
%typemap(cscode)           enum SWIGTYPE ""
%typemap(csimports)        enum SWIGTYPE ""
%typemap(csinterfaces)     enum SWIGTYPE ""

%typemap(csbody) enum SWIGTYPE ""

%csenum(simple);

