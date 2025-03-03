/* -----------------------------------------------------------------------------
 * std_common.i
 *
 * SWIG typemaps for STL - common utilities
 * ----------------------------------------------------------------------------- */

%include <std/std_except.i>

%apply size_t { std::size_t };

%{
#include <string>

SWIGINTERNINLINE
std::string swig_scm_to_string(Scheme_Object *x) {
    return std::string(SCHEME_STR_VAL(x));
}

SWIGINTERNINLINE
Scheme_Object *swig_make_string(const std::string &s) {
    return scheme_make_string(s.c_str());
}
%}
