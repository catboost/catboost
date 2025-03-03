/* -----------------------------------------------------------------------------
 * std_string.i
 *
 * SWIG typemaps for std::string types
 * ----------------------------------------------------------------------------- */

// ------------------------------------------------------------------------
// std::string is typemapped by value
// This can prevent exporting methods which return a string
// in order for the user to modify it.
// However, I think I'll wait until someone asks for it...
// ------------------------------------------------------------------------

%include <exception.i>

%{
#include <string>
%}

namespace std {

    %naturalvar string;

    class string;

    %typemap(typecheck,precedence=SWIG_TYPECHECK_STRING) string, const string& %{
        $1 = (Z_TYPE($input) == IS_STRING) ? 1 : 0;
    %}

    %typemap(in) string %{
        convert_to_string(&$input);
        $1.assign(Z_STRVAL($input), Z_STRLEN($input));
    %}

    %typemap(directorout) string %{
      if (!EG(exception)) {
        convert_to_string($input);
        $result.assign(Z_STRVAL_P($input), Z_STRLEN_P($input));
      }
    %}

    %typemap(out) string %{
        ZVAL_STRINGL($result, $1.data(), $1.size());
    %}

    %typemap(directorin) string, const string& %{
        ZVAL_STRINGL($input, $1.data(), $1.size());
    %}

    %typemap(out) const string & %{
        ZVAL_STRINGL($result, $1->data(), $1->size());
    %}

    %typemap(throws) string, const string& %{
        zend_throw_exception(NULL, $1.c_str(), 0);
        return;
    %}

    %typemap(in) const string & ($*1_ltype temp) %{
        convert_to_string(&$input);
        temp.assign(Z_STRVAL($input), Z_STRLEN($input));
        $1 = &temp;
    %}

    /* These next two handle a function which takes a non-const reference to
     * a std::string and modifies the string. */
    %typemap(in,byref=1) string & ($*1_ltype temp) %{
        {
          zval * p = Z_ISREF($input) ? Z_REFVAL($input) : &$input;
          convert_to_string(p);
          temp.assign(Z_STRVAL_P(p), Z_STRLEN_P(p));
          $1 = &temp;
        }
    %}

    %typemap(directorout) string & ($*1_ltype *temp) %{
      if (!EG(exception)) {
        convert_to_string($input);
        temp = new $*1_ltype(Z_STRVAL_P($input), Z_STRLEN_P($input));
        swig_acquire_ownership(temp);
        $result = temp;
      }
    %}

    %typemap(argout) string & %{
      if (Z_ISREF($input)) {
        ZVAL_STRINGL(Z_REFVAL($input), $1->data(), $1->size());
      }
    %}

    /* SWIG will apply the non-const typemap above to const string& without
     * this more specific typemap. */
    %typemap(argout) const string & "";
}
