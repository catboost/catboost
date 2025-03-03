/* -----------------------------------------------------------------------------
 * dmemberfunctionpointers.swg
 *
 * Typemaps for member function pointers.
 * ----------------------------------------------------------------------------- */


%typemap(ctype) SWIGTYPE (CLASS::*) "char *"
%typemap(imtype) SWIGTYPE (CLASS::*) "char*"
%typemap(dtype) SWIGTYPE (CLASS::*) "$dclassname"

%typecheck(SWIG_TYPECHECK_POINTER)
  SWIGTYPE (CLASS::*)
  ""


/*
 * Conversion generation typemaps.
 */

%typemap(in, fragment="SWIG_UnPackData") SWIGTYPE (CLASS::*) %{
  SWIG_UnpackData($input, (void *)&$1, sizeof($1));
%}
%typemap(out, fragment="SWIG_PackData") SWIGTYPE (CLASS::*) %{
  char buf[128];
  char *data = SWIG_PackData(buf, (void *)&$1, sizeof($1));
  *data = '\0';
  $result = SWIG_d_string_callback(buf);
%}

%typemap(directorin) SWIGTYPE (CLASS::*) "$input = (void *) $1;"
%typemap(directorout, warning=SWIGWARN_TYPEMAP_DIRECTOROUT_PTR_MSG) SWIGTYPE (CLASS::*)
  "$result = ($1_ltype)$input;"

%typemap(ddirectorin) SWIGTYPE (CLASS::*)
  "($winput is null) ? null : new $dclassname($winput, false)"
%typemap(ddirectorout) SWIGTYPE (CLASS::*) "$dclassname.swigGetCPtr($dcall)"

%typemap(din) SWIGTYPE (CLASS::*) "$dclassname.swigGetCMemberPtr($dinput)"
%typemap(dout, excode=SWIGEXCODE) SWIGTYPE (CLASS::*) {
  char* cMemberPtr = $imcall;
  $dclassname ret = (cMemberPtr is null) ? null : new $dclassname(cMemberPtr, $owner);$excode
  return ret;
}

%apply SWIGTYPE (CLASS::*) { SWIGTYPE (CLASS::*const) }
%apply SWIGTYPE & { SWIGTYPE (CLASS::*const&) }

/*
 * Helper functions to pack/unpack arbitrary binary data (member function
 * pointers in this case) into a string.
 */

%fragment("SWIG_PackData", "header") {
/* Pack binary data into a string */
SWIGINTERN char * SWIG_PackData(char *c, void *ptr, size_t sz) {
  static const char hex[17] = "0123456789abcdef";
  const unsigned char *u = (unsigned char *) ptr;
  const unsigned char *eu =  u + sz;
  for (; u != eu; ++u) {
    unsigned char uu = *u;
    *(c++) = hex[(uu & 0xf0) >> 4];
    *(c++) = hex[uu & 0xf];
  }
  return c;
}
}

%fragment("SWIG_UnPackData", "header") {
/* Unpack binary data from a string */
SWIGINTERN const char * SWIG_UnpackData(const char *c, void *ptr, size_t sz) {
  unsigned char *u = (unsigned char *) ptr;
  const unsigned char *eu = u + sz;
  for (; u != eu; ++u) {
    char d = *(c++);
    unsigned char uu;
    if ((d >= '0') && (d <= '9'))
      uu = ((d - '0') << 4);
    else if ((d >= 'a') && (d <= 'f'))
      uu = ((d - ('a'-10)) << 4);
    else
      return (char *) 0;
    d = *(c++);
    if ((d >= '0') && (d <= '9'))
      uu |= (d - '0');
    else if ((d >= 'a') && (d <= 'f'))
      uu |= (d - ('a'-10));
    else
      return (char *) 0;
    *u = uu;
  }
  return c;
}
}
