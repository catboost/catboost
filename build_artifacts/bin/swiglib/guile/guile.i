/* -----------------------------------------------------------------------------
 * guile.i
 *
 * SWIG Configuration File for Guile.
 * ----------------------------------------------------------------------------- */

/* Macro for inserting Scheme code into the stub */
#define %scheme	    %insert("scheme")
#define %goops      %insert("goops")

/* Return-styles */
%pragma(guile) return_nothing_doc = "Returns unspecified."
%pragma(guile) return_one_doc = "Returns $values."

%define %values_as_list
  %pragma(guile) beforereturn = ""
  %pragma(guile) return_multi_doc = "Returns a list of $num_values values: $values."
%enddef
%values_as_list /* the default style */

%define %values_as_vector
  %pragma(guile) beforereturn = "GUILE_MAYBE_VECTOR"
  %pragma(guile) return_multi_doc = "Returns a vector of $num_values values: $values."
%enddef

%define %multiple_values
  %pragma(guile) beforereturn = "GUILE_MAYBE_VALUES"
  %pragma(guile) return_multi_doc = "Returns $num_values values: $values."
%enddef

#define GUILE_APPEND_RESULT SWIG_APPEND_VALUE

%include <typemaps.i>
