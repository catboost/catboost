/* -----------------------------------------------------------------------------
 * typemaps.i
 * ----------------------------------------------------------------------------- */

/* The MzScheme module handles all types uniformly via typemaps. Here
   are the definitions.  */

/* Pointers */

%typemap(in) SWIGTYPE * {
  $1 = ($ltype) SWIG_MustGetPtr($input, $descriptor, $argnum, 0);
}

%typemap(in) void * {
  $1 = SWIG_MustGetPtr($input, NULL, $argnum, 0);
}

%typemap(varin) SWIGTYPE * {
  $1 = ($ltype) SWIG_MustGetPtr($input, $descriptor, 1, 0);
}

%typemap(varin) SWIGTYPE & {
  $1 = *(($1_ltype)SWIG_MustGetPtr($input, $descriptor, 1, 0));
}

%typemap(varin) SWIGTYPE && {
  $1 = *(($1_ltype)SWIG_MustGetPtr($input, $descriptor, 1, 0));
}

%typemap(varin) SWIGTYPE [ANY] {
  void *temp;
  int ii;
  $1_basetype *b = 0;
  temp = SWIG_MustGetPtr($input, $1_descriptor, 1, 0);
  b = ($1_basetype *) $1;
  for (ii = 0; ii < $1_size; ii++) b[ii] = *(($1_basetype *) temp + ii);
}
  

%typemap(varin) void * {
  $1 = SWIG_MustGetPtr($input, NULL, 1, 0);
}

%typemap(out) SWIGTYPE * {
  $result = SWIG_NewPointerObj ($1, $descriptor, $owner);
}

%typemap(out) SWIGTYPE *DYNAMIC {
  swig_type_info *ty = SWIG_TypeDynamicCast($1_descriptor,(void **) &$1);
  $result = SWIG_NewPointerObj ($1, ty, $owner);
}
    
%typemap(varout) SWIGTYPE *, SWIGTYPE [] {
  $result = SWIG_NewPointerObj ($1, $descriptor, 0);
}

%typemap(varout) SWIGTYPE & {
  $result = SWIG_NewPointerObj((void *) &$1, $1_descriptor, 0);
}

%typemap(varout) SWIGTYPE && {
  $result = SWIG_NewPointerObj((void *) &$1, $1_descriptor, 0);
}

/* C++ References */

#ifdef __cplusplus

%typemap(in) SWIGTYPE &, SWIGTYPE && { 
  $1 = ($ltype) SWIG_MustGetPtr($input, $descriptor, $argnum, 0);
  if ($1 == NULL) scheme_signal_error("swig-type-error (null reference)");
}

%typemap(out) SWIGTYPE &, SWIGTYPE && {
  $result = SWIG_NewPointerObj ($1, $descriptor, $owner);
}

%typemap(out) SWIGTYPE &DYNAMIC {
  swig_type_info *ty = SWIG_TypeDynamicCast($1_descriptor,(void **) &$1);
  $result = SWIG_NewPointerObj ($1, ty, $owner);
}

#endif

/* Arrays */

%typemap(in) SWIGTYPE[] {
  $1 = ($ltype) SWIG_MustGetPtr($input, $descriptor, $argnum, 0);
}

%typemap(out) SWIGTYPE[] {
  $result = SWIG_NewPointerObj ($1, $descriptor, $owner);
}

/* Enums */
%typemap(in) enum SWIGTYPE {
  if (!SWIG_is_integer($input)) 
      scheme_wrong_type(FUNC_NAME, "integer", $argnum - 1, argc, argv);
  $1 = ($1_type) SWIG_convert_int($input);
}

%typemap(varin) enum SWIGTYPE {
  if (!SWIG_is_integer($input)) 
      scheme_wrong_type(FUNC_NAME, "integer", 0, argc, argv);
  $1 = ($1_type) SWIG_convert_int($input);
}

%typemap(out) enum SWIGTYPE "$result = scheme_make_integer_value($1);";
%typemap(varout) enum SWIGTYPE "$result = scheme_make_integer_value($1);";


/* Pass-by-value */

%typemap(in) SWIGTYPE($&1_ltype argp) {
  argp = ($&1_ltype) SWIG_MustGetPtr($input, $&1_descriptor, $argnum, 0);
  $1 = *argp;
}

%typemap(varin) SWIGTYPE {
  $&1_ltype argp;
  argp = ($&1_ltype) SWIG_MustGetPtr($input, $&1_descriptor, 1, 0);
  $1 = *argp;
}


%typemap(out) SWIGTYPE 
#ifdef __cplusplus
{
  $&1_ltype resultptr;
  resultptr = new $1_ltype(($1_ltype &) $1);
  $result =  SWIG_NewPointerObj (resultptr, $&1_descriptor, 1);
} 
#else
{
  $&1_ltype resultptr;
  resultptr = ($&1_ltype) malloc(sizeof($1_type));
  memmove(resultptr, &$1, sizeof($1_type));
  $result = SWIG_NewPointerObj(resultptr, $&1_descriptor, 1);
}
#endif

%typemap(varout) SWIGTYPE 
#ifdef __cplusplus
{
  $&1_ltype resultptr;
  resultptr = new $1_ltype(($1_ltype &) $1);
  $result =  SWIG_NewPointerObj (resultptr, $&1_descriptor, 0);
} 
#else
{
  $&1_ltype resultptr;
  resultptr = ($&1_ltype) malloc(sizeof($1_type));
  memmove(resultptr, &$1, sizeof($1_type));
  $result = SWIG_NewPointerObj(resultptr, $&1_descriptor, 0);
}
#endif

/* The SIMPLE_MAP macro below defines the whole set of typemaps needed
   for simple types. */

%define SIMPLE_MAP(C_NAME, MZ_PREDICATE, MZ_TO_C, C_TO_MZ, MZ_NAME)
%typemap(in) C_NAME {
    if (!MZ_PREDICATE($input))
	scheme_wrong_type(FUNC_NAME, #MZ_NAME, $argnum - 1, argc, argv);
    $1 = MZ_TO_C($input);
}
%typemap(varin) C_NAME {
    if (!MZ_PREDICATE($input))
	scheme_wrong_type(FUNC_NAME, #MZ_NAME, 0, argc, argv);
    $1 = MZ_TO_C($input);
}
%typemap(out) C_NAME {
    $result = C_TO_MZ($1);
}
%typemap(varout) C_NAME {
    $result = C_TO_MZ($1);
}
%typemap(in) C_NAME *INPUT (C_NAME temp) {
    temp = (C_NAME) MZ_TO_C($input);
    $1 = &temp;
}
%typemap(in,numinputs=0) C_NAME *OUTPUT (C_NAME temp) {
    $1 = &temp;
}
%typemap(argout) C_NAME *OUTPUT {
    Scheme_Object *s;
    s = C_TO_MZ(*$1);
    SWIG_APPEND_VALUE(s);
}
%typemap(in) C_NAME *BOTH = C_NAME *INPUT;
%typemap(argout) C_NAME *BOTH = C_NAME *OUTPUT;
%typemap(in) C_NAME *INOUT = C_NAME *INPUT;
%typemap(argout) C_NAME *INOUT = C_NAME *OUTPUT;
%enddef

SIMPLE_MAP(bool, SCHEME_BOOLP, SCHEME_TRUEP,
	   swig_make_boolean, boolean);
SIMPLE_MAP(char, SCHEME_CHARP, SCHEME_CHAR_VAL,
	   scheme_make_character, character);
SIMPLE_MAP(unsigned char, SCHEME_CHARP, SCHEME_CHAR_VAL,
	   scheme_make_character, character);
SIMPLE_MAP(int, SWIG_is_integer, SWIG_convert_int,
	   scheme_make_integer_value, integer);
SIMPLE_MAP(short, SWIG_is_integer, SWIG_convert_short,
	   scheme_make_integer_value, integer);
SIMPLE_MAP(long, SWIG_is_integer, SWIG_convert_long,
	   scheme_make_integer_value, integer);
SIMPLE_MAP(ptrdiff_t, SWIG_is_integer, SWIG_convert_long,
	   scheme_make_integer_value, integer);
SIMPLE_MAP(unsigned int, SWIG_is_unsigned_integer, SWIG_convert_unsigned_int,
	   scheme_make_integer_value_from_unsigned, integer);
SIMPLE_MAP(unsigned short, SWIG_is_unsigned_integer, SWIG_convert_unsigned_short,
	   scheme_make_integer_value_from_unsigned, integer);
SIMPLE_MAP(unsigned long, SWIG_is_unsigned_integer, SWIG_convert_unsigned_long,
	   scheme_make_integer_value_from_unsigned, integer);
SIMPLE_MAP(size_t, SWIG_is_unsigned_integer, SWIG_convert_unsigned_long,
	   scheme_make_integer_value_from_unsigned, integer);
SIMPLE_MAP(float, SCHEME_REALP, scheme_real_to_double,
	   scheme_make_double, real);
SIMPLE_MAP(double, SCHEME_REALP, scheme_real_to_double,
	   scheme_make_double, real);

SIMPLE_MAP(char *, SCHEME_STRINGP, SCHEME_STR_VAL, 
	   SCHEME_MAKE_STRING, string);
SIMPLE_MAP(const char *, SCHEME_STRINGP, SCHEME_STR_VAL, 
	   SCHEME_MAKE_STRING, string);

/* For MzScheme 30x:  Use these typemaps if you are not going to use
   UTF8 encodings in your C code. 
 SIMPLE_MAP(char *,SCHEME_BYTE_STRINGP, SCHEME_BYTE_STR_VAL,
 	   scheme_make_byte_string_without_copying,bytestring);
 SIMPLE_MAP(const char *,SCHEME_BYTE_STRINGP, SCHEME_BYTE_STR_VAL,
 	   scheme_make_byte_string_without_copying,bytestring);
*/

/* Const primitive references.  Passed by value */

%define REF_MAP(C_NAME, MZ_PREDICATE, MZ_TO_C, C_TO_MZ, MZ_NAME)
  %typemap(in) const C_NAME & (C_NAME temp) {
     if (!MZ_PREDICATE($input))
        scheme_wrong_type(FUNC_NAME, #MZ_NAME, $argnum - 1, argc, argv);
     temp = MZ_TO_C($input);
     $1 = &temp;
  }
  %typemap(out) const C_NAME & {
    $result = C_TO_MZ(*$1);
  }
%enddef

REF_MAP(bool, SCHEME_BOOLP, SCHEME_TRUEP,
	   swig_make_boolean, boolean);
REF_MAP(char, SCHEME_CHARP, SCHEME_CHAR_VAL,
	   scheme_make_character, character);
REF_MAP(unsigned char, SCHEME_CHARP, SCHEME_CHAR_VAL,
	   scheme_make_character, character);
REF_MAP(int, SWIG_is_integer, SWIG_convert_int,
	   scheme_make_integer_value, integer);
REF_MAP(short, SWIG_is_integer, SWIG_convert_short,
	   scheme_make_integer_value, integer);
REF_MAP(long, SWIG_is_integer, SWIG_convert_long,
	   scheme_make_integer_value, integer);
REF_MAP(unsigned int, SWIG_is_unsigned_integer, SWIG_convert_unsigned_int,
	   scheme_make_integer_value_from_unsigned, integer);
REF_MAP(unsigned short, SWIG_is_unsigned_integer, SWIG_convert_unsigned_short,
	   scheme_make_integer_value_from_unsigned, integer);
REF_MAP(unsigned long, SWIG_is_unsigned_integer, SWIG_convert_unsigned_long,
	   scheme_make_integer_value_from_unsigned, integer);
REF_MAP(float, SCHEME_REALP, scheme_real_to_double,
	   scheme_make_double, real);
REF_MAP(double, SCHEME_REALP, scheme_real_to_double,
	   scheme_make_double, real);

/* Void */

%typemap(out) void "$result = scheme_void;";

/* Pass through Scheme_Object * */

%typemap (in) Scheme_Object * "$1=$input;";
%typemap (out) Scheme_Object * "$result=$1;";
%typecheck(SWIG_TYPECHECK_POINTER) Scheme_Object * "$1=1;";


/* ------------------------------------------------------------
 * String & length
 * ------------------------------------------------------------ */

//%typemap(in) (char *STRING, int LENGTH) {
//    int temp;
//    $1 = ($1_ltype) SWIG_Guile_scm2newstr($input, &temp);
//    $2 = ($2_ltype) temp;
//}


/* ------------------------------------------------------------
 * Typechecking rules
 * ------------------------------------------------------------ */

%typecheck(SWIG_TYPECHECK_INTEGER)
	 int, short, long,
 	 unsigned int, unsigned short, unsigned long,
	 signed char, unsigned char,
	 long long, unsigned long long,
	 const int &, const short &, const long &,
 	 const unsigned int &, const unsigned short &, const unsigned long &,
	 const long long &, const unsigned long long &,
	 enum SWIGTYPE
{
  $1 = (SWIG_is_integer($input)) ? 1 : 0;
}

%typecheck(SWIG_TYPECHECK_BOOL) bool, bool &, const bool &
{
  $1 = (SCHEME_BOOLP($input)) ? 1 : 0;
}

%typecheck(SWIG_TYPECHECK_DOUBLE)
	float, double,
	const float &, const double &
{
  $1 = (SCHEME_REALP($input)) ? 1 : 0;
}

%typecheck(SWIG_TYPECHECK_STRING) char {
  $1 = (SCHEME_STRINGP($input)) ? 1 : 0;
}

%typecheck(SWIG_TYPECHECK_STRING) char * {
  $1 = (SCHEME_STRINGP($input)) ? 1 : 0;
}

%typecheck(SWIG_TYPECHECK_POINTER) SWIGTYPE *, SWIGTYPE [] {
  void *ptr;
  if (SWIG_ConvertPtr($input, (void **) &ptr, $1_descriptor, 0)) {
    $1 = 0;
  } else {
    $1 = 1;
  }
}

%typecheck(SWIG_TYPECHECK_POINTER) SWIGTYPE &, SWIGTYPE && {
  void *ptr;
  if (SWIG_ConvertPtr($input, (void **) &ptr, $1_descriptor, SWIG_POINTER_NO_NULL)) {
    $1 = 0;
  } else {
    $1 = 1;
  }
}

%typecheck(SWIG_TYPECHECK_POINTER) SWIGTYPE {
  void *ptr;
  if (SWIG_ConvertPtr($input, (void **) &ptr, $&1_descriptor, SWIG_POINTER_NO_NULL)) {
    $1 = 0;
  } else {
    $1 = 1;
  }
}

%typecheck(SWIG_TYPECHECK_VOIDPTR) void * {
  void *ptr;
  if (SWIG_ConvertPtr($input, (void **) &ptr, 0, 0)) {
    $1 = 0;
  } else {
    $1 = 1;
  }
}


/* Array reference typemaps */
%apply SWIGTYPE & { SWIGTYPE ((&)[ANY]) }
%apply SWIGTYPE && { SWIGTYPE ((&&)[ANY]) }

/* const pointers */
%apply SWIGTYPE * { SWIGTYPE *const }


