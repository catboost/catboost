#if !defined(SWIG_STD_STRING) 
#define SWIG_STD_BASIC_STRING
#define SWIG_STD_MODERN_STL

%include <octcontainer.swg>

#define %swig_basic_string(Type...)  %swig_sequence_methods_val(Type)


%fragment(SWIG_AsPtr_frag(std::basic_string<char>),"header",
	  fragment="SWIG_AsCharPtrAndSize") {
SWIGINTERN int
SWIG_AsPtr(std::basic_string<char>)(octave_value obj, std::string **val) {
  if (obj.is_string()) {
    if (val)
      *val = new std::string(obj.string_value());
    return SWIG_NEWOBJ;
  }
  return SWIG_ERROR;
}
}

%fragment(SWIG_From_frag(std::basic_string<char>),"header",
	  fragment="SWIG_FromCharPtrAndSize") {
SWIGINTERNINLINE octave_value
  SWIG_From(std::basic_string<char>)(const std::string& s) {
    return SWIG_FromCharPtrAndSize(s.data(), s.size());
  }
}

%ignore std::basic_string::operator +=;

%include <std/std_basic_string.i>
%typemaps_asptrfromn(%checkcode(STRING), std::basic_string<char>);

#endif


#if !defined(SWIG_STD_WSTRING)

%fragment(SWIG_AsPtr_frag(std::basic_string<wchar_t>),"header",
	  fragment="SWIG_AsWCharPtrAndSize") {
SWIGINTERN int
SWIG_AsPtr(std::basic_string<wchar_t>)(octave_value obj, std::wstring **val) {
  if (obj.is_string()) {
    if (val)
      *val = new std::wstring(obj.string_value());
    return SWIG_NEWOBJ;
  }
  return SWIG_ERROR;
}
}

%fragment(SWIG_From_frag(std::basic_string<wchar_t>),"header",
	  fragment="SWIG_FromWCharPtrAndSize") {
SWIGINTERNINLINE PyObject*
  SWIG_From(std::basic_string<wchar_t>)(const std::wstring& s) {
    return SWIG_FromWCharPtrAndSize(s.data(), s.size());
  }
}

%typemaps_asptrfromn(%checkcode(UNISTRING), std::basic_string<wchar_t>);

#endif
