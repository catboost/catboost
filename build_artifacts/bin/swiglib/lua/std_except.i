/* -----------------------------------------------------------------------------
 * Typemaps used by the STL wrappers that throw exceptions.
 * These typemaps are used when methods are declared with an STL exception
 * specification, such as:
 *   size_t at() const throw (std::out_of_range);
 *
 * std_except.i
 * ----------------------------------------------------------------------------- */

%{
#include <typeinfo>
#include <stdexcept>
%}
%include <exception.i>

namespace std 
{
  %ignore exception; // not sure if I should ignore this...
  class exception 
  {
  public:
    exception() throw() { }
    virtual ~exception() throw();
    virtual const char* what() const throw();
  }; 
}

// normally objects which are thrown are returned to the interpreter as errors
// (which potentially may have problems if they are not copied)
// therefore all classes based upon std::exception are converted to their strings & returned as errors
%typemap(throws) std::bad_cast          "SWIG_exception(SWIG_TypeError, $1.what());"
%typemap(throws) std::bad_exception     "SWIG_exception(SWIG_RuntimeError, $1.what());"
%typemap(throws) std::domain_error      "SWIG_exception(SWIG_ValueError, $1.what());"
%typemap(throws) std::exception         "SWIG_exception(SWIG_SystemError, $1.what());"
%typemap(throws) std::invalid_argument  "SWIG_exception(SWIG_ValueError, $1.what());"
%typemap(throws) std::length_error      "SWIG_exception(SWIG_IndexError, $1.what());"
%typemap(throws) std::logic_error       "SWIG_exception(SWIG_RuntimeError, $1.what());"
%typemap(throws) std::out_of_range      "SWIG_exception(SWIG_IndexError, $1.what());"
%typemap(throws) std::overflow_error    "SWIG_exception(SWIG_OverflowError, $1.what());"
%typemap(throws) std::range_error       "SWIG_exception(SWIG_IndexError, $1.what());"
%typemap(throws) std::runtime_error     "SWIG_exception(SWIG_RuntimeError, $1.what());"
%typemap(throws) std::underflow_error   "SWIG_exception(SWIG_RuntimeError, $1.what());"
