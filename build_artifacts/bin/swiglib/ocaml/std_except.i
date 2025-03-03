%{
#include <typeinfo>
#include <stdexcept>
%}

namespace std
{
  %ignore exception;
  struct exception {};
}

%typemap(throws) std::bad_cast          "SWIG_OCamlThrowException(SWIG_OCamlRuntimeException, $1.what());"
%typemap(throws) std::bad_exception     "SWIG_OCamlThrowException(SWIG_OCamlRuntimeException, $1.what());"
%typemap(throws) std::domain_error      "SWIG_OCamlThrowException(SWIG_OCamlRuntimeException, $1.what());"
%typemap(throws) std::exception         "SWIG_OCamlThrowException(SWIG_OCamlRuntimeException, $1.what());"
%typemap(throws) std::invalid_argument  "SWIG_OCamlThrowException(SWIG_OCamlIllegalArgumentException, $1.what());"
%typemap(throws) std::length_error      "SWIG_OCamlThrowException(SWIG_OCamlIndexOutOfBoundsException, $1.what());"
%typemap(throws) std::logic_error       "SWIG_OCamlThrowException(SWIG_OCamlRuntimeException, $1.what());"
%typemap(throws) std::out_of_range      "SWIG_OCamlThrowException(SWIG_OCamlIndexOutOfBoundsException, $1.what());"
%typemap(throws) std::overflow_error    "SWIG_OCamlThrowException(SWIG_OCamlArithmeticException, $1.what());"
%typemap(throws) std::range_error       "SWIG_OCamlThrowException(SWIG_OCamlIndexOutOfBoundsException, $1.what());"
%typemap(throws) std::runtime_error     "SWIG_OCamlThrowException(SWIG_OCamlRuntimeException, $1.what());"
%typemap(throws) std::underflow_error   "SWIG_OCamlThrowException(SWIG_OCamlArithmeticException, $1.what());"
