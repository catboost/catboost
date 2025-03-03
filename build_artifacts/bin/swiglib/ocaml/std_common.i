/* -----------------------------------------------------------------------------
 * std_common.i
 *
 * SWIG typemaps for STL - common utilities
 * ----------------------------------------------------------------------------- */

%include <std/std_except.i>

%apply size_t { std::size_t };
%apply const size_t& { const std::size_t& };

%{
#include <string>
SWIGINTERNINLINE
CAML_VALUE SwigString_FromString(const std::string &s) {
  return caml_val_string((char *)s.c_str());
}

SWIGINTERNINLINE
std::string SwigString_AsString(CAML_VALUE o) {
  return std::string((char *)caml_ptr_val(o,0));
}
%}
