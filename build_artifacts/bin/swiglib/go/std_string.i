/* -----------------------------------------------------------------------------
 * std_string.i
 *
 * Typemaps for std::string and const std::string&
 * These are mapped to a Go string and are passed around by value.
 *
 * To use non-const std::string references use the following %apply.  Note 
 * that they are passed by value.
 * %apply const std::string & {std::string &};
 * ----------------------------------------------------------------------------- */

%{
#include <string>
%}

namespace std {

%naturalvar string;

class string;

%typemap(gotype) string, const string & "string"

%typemap(in) string
%{ $1.assign($input.p, $input.n); %}

%typemap(godirectorout) string
%{
  {
    p := Swig_malloc(len($input))
    s := (*[1<<30]byte)(unsafe.Pointer(p))[:len($input)]
    copy(s, $input)
    $result = *(*string)(unsafe.Pointer(&s))
  }
%}

%typemap(directorout) string
%{
  $result.assign($input.p, $input.n);
  free($input.p);
%}

%typemap(out,fragment="AllocateString") string
%{ $result = Swig_AllocateString($1.data(), $1.length()); %}

%typemap(goout,fragment="CopyString") string
%{ $result = swigCopyString($1) %}

%typemap(directorin,fragment="AllocateString") string
%{ $input = Swig_AllocateString($1.data(), $1.length()); %}

%typemap(godirectorin,fragment="CopyString") string
%{ $result = swigCopyString($input) %}

%typemap(in) const string &
%{
  $*1_ltype $1_str($input.p, $input.n);
  $1 = &$1_str;
%}

%typemap(godirectorout) const string &
%{
  {
    p := Swig_malloc(len($input))
    s := (*[1<<30]byte)(unsafe.Pointer(p))[:len($input)]
    copy(s, $input)
    $result = *(*string)(unsafe.Pointer(&s))
  }
%}

%typemap(directorout,warning=SWIGWARN_TYPEMAP_THREAD_UNSAFE_MSG) const string &
%{
  static $*1_ltype $1_str;
  $1_str.assign($input.p, $input.n);
  free($input.p);
  $result = &$1_str;
%}

%typemap(out,fragment="AllocateString") const string &
%{ $result = Swig_AllocateString((*$1).data(), (*$1).length()); %}

%typemap(goout,fragment="CopyString") const string &
%{ $result = swigCopyString($1) %}

%typemap(directorin,fragment="AllocateString") const string &
%{ $input = Swig_AllocateString($1.data(), $1.length()); %}

%typemap(godirectorin,fragment="CopyString") const string &
%{ $result = swigCopyString($input) %}

}
