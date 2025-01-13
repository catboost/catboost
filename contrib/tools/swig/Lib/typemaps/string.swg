%ensure_fragment(SWIG_AsCharPtrAndSize)
%ensure_fragment(SWIG_FromCharPtrAndSize)

%types(char *);

%fragment("SWIG_pchar_descriptor","header") {
SWIGINTERN swig_type_info*
SWIG_pchar_descriptor(void)
{
  static swig_type_info* info = 0;
  if (!info) {
    info = SWIG_TypeQuery("_p_char");
  }
  return info;
}
}

%fragment("SWIG_strnlen","header",fragment="SWIG_FromCharPtrAndSize") {
SWIGINTERN size_t
SWIG_strnlen(const char* s, size_t maxlen)
{
  const char *p;
  for (p = s; maxlen-- && *p; p++)
    ;
  return p - s;
}
}

%include <typemaps/strings.swg>
%typemaps_string(%checkcode(STRING), %checkcode(CHAR),
		 SWIGWARN_TYPEMAP_CHARLEAK_MSG,
		 char, Char, SWIG_AsCharPtrAndSize, SWIG_FromCharPtrAndSize,
		 strlen, SWIG_strnlen,
		"<limits.h>", CHAR_MIN, CHAR_MAX)
