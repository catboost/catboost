
// Documentation for use with the autodoc feature.

#ifdef SWIG_DOC_DOXYGEN_STYLE
%typemap(doc) SWIGTYPE "@param $1_name $1_type"
%typemap(doc) SWIGTYPE * "@param $1_name $1_type"
%typemap(doc) const SWIGTYPE & "@param $1_name $1_type"
%typemap(doc) const SWIGTYPE && "@param $1_name $1_type"
%typemap(doc) enum SWIGTYPE "@param $1_name enum $1_type"

%typemap(doc) SWIGTYPE *INOUT, SWIGTYPE &INOUT "@param $1_name $1_type (input/output)"
%typemap(doc) SWIGTYPE *INPUT, SWIGTYPE &INPUT "@param $1_name $1_type (input)"
%typemap(doc) SWIGTYPE *OUTPUT, SWIGTYPE &OUTPUT "@param $1_name $1_type (output)"
#else
%typemap(doc) SWIGTYPE "$1_name: $1_type"
%typemap(doc) SWIGTYPE * "$1_name: $1_type"
%typemap(doc) const SWIGTYPE & "$1_name: $1_type"
%typemap(doc) const SWIGTYPE && "$1_name: $1_type"
%typemap(doc) enum SWIGTYPE "$1_name: enum $1_type"

%typemap(doc) SWIGTYPE *INOUT, SWIGTYPE &INOUT "$1_name: $1_type (input/output)"
%typemap(doc) SWIGTYPE *INPUT, SWIGTYPE &INPUT "$1_name: $1_type (input)"
%typemap(doc) SWIGTYPE *OUTPUT, SWIGTYPE &OUTPUT "$1_name: $1_type (output)"
#endif


// Types to use in Python documentation for the parameters of the given C++ type.
%typemap(doctype) bool "boolean"

%define int_doctype_for_cppint_type(cppint_type)
    %typemap(doctype) cppint_type, unsigned cppint_type "int"
%enddef
%formacro(int_doctype_for_cppint_type, short, int, long, long long)

%typemap(doctype) size_t "int"

%typemap(doctype) enum SWIGTYPE "int"

%typemap(doctype) float, double, long double "float"

%typemap(doctype) char*, std::string "string"

%typemap(doctype) SWIGTYPE "$1_basetype"
%typemap(doctype) SWIGTYPE * "$typemap(doctype, $*1_ltype)"
%typemap(doctype) SWIGTYPE & "$typemap(doctype, $*1_ltype)"
