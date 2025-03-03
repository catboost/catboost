/* -----------------------------------------------------------------------------
 * typemaps.i
 *
 * Guile-specific typemaps
 * ----------------------------------------------------------------------------- */

/* Pointers */

%typemap(in) SWIGTYPE *, SWIGTYPE &, SWIGTYPE &&, SWIGTYPE [] {
  $1 = ($1_ltype)SWIG_MustGetPtr($input, $descriptor, $argnum, 0);
}
%typemap(freearg) SWIGTYPE *, SWIGTYPE &, SWIGTYPE &&, SWIGTYPE [] "";

%typemap(in) void * {
  $1 = ($1_ltype)SWIG_MustGetPtr($input, NULL, $argnum, 0);
}
%typemap(freearg) void * "";

%typemap(varin) SWIGTYPE * {
  $1 = ($1_ltype)SWIG_MustGetPtr($input, $descriptor, 1, 0);
}

%typemap(varin) SWIGTYPE & {
  $1 = *(($1_ltype)SWIG_MustGetPtr($input, $descriptor, 1, 0));
}

%typemap(varin) SWIGTYPE && {
  $1 = *(($1_ltype)SWIG_MustGetPtr($input, $descriptor, 1, 0));
}

%typemap(varin) SWIGTYPE [] {
  scm_wrong_type_arg((char *) FUNC_NAME, 1, $input);
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

%typemap(out) SWIGTYPE *, SWIGTYPE &, SWIGTYPE &&, SWIGTYPE [] {
  $result = SWIG_NewPointerObj ($1, $descriptor, $owner);
}

%typemap(out) SWIGTYPE *DYNAMIC, SWIGTYPE &DYNAMIC {
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

%typemap(throws) SWIGTYPE {
  $&ltype temp = new $ltype($1);
  scm_throw(scm_from_locale_symbol((char *) "swig-exception"),
	    scm_list_n(SWIG_NewPointerObj(temp, $&descriptor, 1),
		    SCM_UNDEFINED));
}

%typemap(throws) SWIGTYPE & {
  scm_throw(scm_from_locale_symbol((char *) "swig-exception"),
	    scm_list_n(SWIG_NewPointerObj(&$1, $descriptor, 1),
		    SCM_UNDEFINED));
}

%typemap(throws) SWIGTYPE && {
  scm_throw(gh_symbol2scm((char *) "swig-exception"),
	    gh_list(SWIG_NewPointerObj(&$1, $descriptor, 1),
		    SCM_UNDEFINED));
}

%typemap(throws) SWIGTYPE * {
  scm_throw(scm_from_locale_symbol((char *) "swig-exception"),
	    scm_list_n(SWIG_NewPointerObj($1, $descriptor, 1),
		    SCM_UNDEFINED));
}

%typemap(throws) SWIGTYPE [] {
  scm_throw(scm_from_locale_symbol((char *) "swig-exception"),
	    scm_list_n(SWIG_NewPointerObj($1, $descriptor, 1),
		    SCM_UNDEFINED));
}

/* Change of object ownership, and interaction of destructor-like functions and the
   garbage-collector */

%typemap(in, doc="$NAME is of type <$type> and gets destroyed by the function") SWIGTYPE *DESTROYED {
  $1 = ($1_ltype)SWIG_MustGetPtr($input, $descriptor, $argnum, 0);
}

%typemap(freearg) SWIGTYPE *DESTROYED {
  SWIG_Guile_MarkPointerDestroyed($input);
}

%typemap(in, doc="$NAME is of type <$type> and is consumed by the function") SWIGTYPE *CONSUMED {
  $1 = ($1_ltype)SWIG_MustGetPtr($input, $descriptor, $argnum, 0);
  SWIG_Guile_MarkPointerNoncollectable($input);
}

/* Pass-by-value */

%typemap(in) SWIGTYPE($&1_ltype argp) {
  argp = ($&1_ltype)SWIG_MustGetPtr($input, $&1_descriptor, $argnum, 0);
  $1 = *argp;
}

%typemap(varin) SWIGTYPE {
  $&1_ltype argp;
  argp = ($&1_ltype)SWIG_MustGetPtr($input, $&1_descriptor, 1, 0);
  $1 = *argp;
}

%typemap(out) SWIGTYPE 
#ifdef __cplusplus
{
  $&1_ltype resultptr;
  resultptr = new $1_ltype((const $1_ltype &) $1);
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
  resultptr = new $1_ltype((const $1_ltype&) $1);
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

/* Enums */

%typemap(in)     enum SWIGTYPE  { $1 = ($1_type) scm_to_int($input); }
/* The complicated construction below needed to deal with anonymous
   enums, which cannot be cast to. */
%typemap(varin)  enum SWIGTYPE  {
  if (sizeof(int) != sizeof($1)) {
    scm_error(scm_from_locale_symbol("swig-error"),
	      (char *) FUNC_NAME,
	      (char *) "enum variable '$name' cannot be set",
	      SCM_EOL, SCM_BOOL_F); 
  }
  * (int *) &($1) = scm_to_int($input);
}
%typemap(out)    enum SWIGTYPE  { $result = scm_from_long((int)$1); }
%typemap(varout) enum SWIGTYPE  { $result = scm_from_long((int)$1); }
%typemap(throws) enum SWIGTYPE {
  scm_throw(scm_from_locale_symbol((char *) "swig-exception"),
     scm_list_n(scm_from_long((int)$1), SCM_UNDEFINED));
}

/* The SIMPLE_MAP_WITH_EXPR macro below defines the whole set of
   typemaps needed for simple types.
   -- SCM_TO_C_EXPR is a C expression that translates the Scheme value
      "swig_scm_value" to a C value.
   -- C_TO_SCM_EXPR is a C expression that translates the C value
      "swig_c_value" to a Scheme value. */

%define SIMPLE_MAP_WITH_EXPR(C_NAME, SCM_TO_C_EXPR, C_TO_SCM_EXPR, SCM_NAME)
 %typemap (in,     doc="$NAME is of type <" #SCM_NAME ">") C_NAME
     { SCM swig_scm_value = $input;
       $1 = SCM_TO_C_EXPR; }
 %typemap (varin,  doc="NEW-VALUE is of type <" #SCM_NAME ">") C_NAME
     { SCM swig_scm_value = $input;
       $1 = SCM_TO_C_EXPR; }
 %typemap (out,    doc="<" #SCM_NAME ">") C_NAME
     { C_NAME swig_c_value = $1;
       $result = C_TO_SCM_EXPR; }
 %typemap (varout, doc="<" #SCM_NAME ">") C_NAME
     { C_NAME swig_c_value = $1;
       $result = C_TO_SCM_EXPR; }
 /* INPUT and OUTPUT */
 %typemap (in, doc="$NAME is of type <" #SCM_NAME ">)")
     C_NAME *INPUT(C_NAME temp) {
       SCM swig_scm_value = $input;
       temp = (C_NAME) SCM_TO_C_EXPR; $1 = &temp; }
 %typemap (in,numinputs=0)      C_NAME *OUTPUT (C_NAME temp)
     {$1 = &temp;}
 %typemap (argout,doc="$name (of type <" #SCM_NAME ">)") C_NAME *OUTPUT
     { C_NAME swig_c_value = *$1;
       SWIG_APPEND_VALUE(C_TO_SCM_EXPR); }
 %typemap (in)          C_NAME *BOTH = C_NAME *INPUT;
 %typemap (argout)      C_NAME *BOTH = C_NAME *OUTPUT;
 %typemap (in)          C_NAME *INOUT = C_NAME *INPUT;
 %typemap (argout)      C_NAME *INOUT = C_NAME *OUTPUT;
 /* Const primitive references.  Passed by value */
 %typemap(in, doc="$NAME is of type <" #SCM_NAME ">") const C_NAME & (C_NAME temp)
     { SCM swig_scm_value = $input;
       temp = SCM_TO_C_EXPR;
       $1 = &temp; }
 %typemap(out, doc="<" #SCM_NAME ">")  const C_NAME &
     { C_NAME swig_c_value = *$1;
       $result = C_TO_SCM_EXPR; }
 /* Throw typemap */
 %typemap(throws) C_NAME {
   C_NAME swig_c_value = $1;
   scm_throw(scm_from_locale_symbol((char *) "swig-exception"),
	     scm_list_n(C_TO_SCM_EXPR, SCM_UNDEFINED));
 }
%enddef

/* The SIMPLE_MAP macro below defines the whole set of typemaps needed
   for simple types.  It generates slightly simpler code than the
   macro above, but it is only suitable for very simple conversion
   expressions. */

%define SIMPLE_MAP(C_NAME, SCM_TO_C, C_TO_SCM, SCM_NAME)
 %typemap (in,     doc="$NAME is of type <" #SCM_NAME ">")
     C_NAME {$1 = ($1_ltype) SCM_TO_C($input);}
 %typemap (varin,  doc="NEW-VALUE is of type <" #SCM_NAME ">")
     C_NAME {$1 = ($1_ltype) SCM_TO_C($input);}
 %typemap (out,    doc="<" #SCM_NAME ">")
     C_NAME {$result = C_TO_SCM($1);}
 %typemap (varout, doc="<" #SCM_NAME ">")
     C_NAME {$result = C_TO_SCM($1);}
 /* INPUT and OUTPUT */
 %typemap (in, doc="$NAME is of type <" #SCM_NAME ">)")
     C_NAME *INPUT(C_NAME temp), C_NAME &INPUT(C_NAME temp) {
   temp = (C_NAME) SCM_TO_C($input); $1 = &temp;
 }
 %typemap (in,numinputs=0)      C_NAME *OUTPUT (C_NAME temp), C_NAME &OUTPUT(C_NAME temp)
   {$1 = &temp;}
 %typemap (argout,doc="$name (of type <" #SCM_NAME ">)") C_NAME *OUTPUT, C_NAME &OUTPUT
   {SWIG_APPEND_VALUE(C_TO_SCM(*$1));}
 %typemap (in)          C_NAME *BOTH = C_NAME *INPUT;
 %typemap (argout)      C_NAME *BOTH = C_NAME *OUTPUT;
 %typemap (in)          C_NAME *INOUT = C_NAME *INPUT;
 %typemap (argout)      C_NAME *INOUT = C_NAME *OUTPUT;
 %typemap (in)          C_NAME &INOUT = C_NAME &INPUT;
 %typemap (argout)      C_NAME &INOUT = C_NAME &OUTPUT;
 /* Const primitive references.  Passed by value */
 %typemap(in, doc="$NAME is of type <" #SCM_NAME ">") const C_NAME & (C_NAME temp) {
   temp = SCM_TO_C($input);
   $1 = ($1_ltype) &temp;
 }
 %typemap(out, doc="<" #SCM_NAME ">")  const C_NAME & {
   $result = C_TO_SCM(*$1);
 }
 /* Throw typemap */
 %typemap(throws) C_NAME {
   scm_throw(scm_from_locale_symbol((char *) "swig-exception"),
	     scm_list_n(C_TO_SCM($1), SCM_UNDEFINED));
 }
%enddef

 SIMPLE_MAP(bool, scm_is_true, scm_from_bool, boolean);
 SIMPLE_MAP(char, SCM_CHAR, SCM_MAKE_CHAR, char);
 SIMPLE_MAP(unsigned char, SCM_CHAR, SCM_MAKE_CHAR, char);
 SIMPLE_MAP(signed char, SCM_CHAR, SCM_MAKE_CHAR, char);
 SIMPLE_MAP(int, scm_to_int, scm_from_long, integer);
 SIMPLE_MAP(short, scm_to_short, scm_from_long, integer);
 SIMPLE_MAP(long, scm_to_long, scm_from_long, integer);
 SIMPLE_MAP(ptrdiff_t, scm_to_long, scm_from_long, integer);
 SIMPLE_MAP(unsigned int, scm_to_uint, scm_from_ulong, integer);
 SIMPLE_MAP(unsigned short, scm_to_ushort, scm_from_ulong, integer);
 SIMPLE_MAP(unsigned long, scm_to_ulong, scm_from_ulong, integer);
 SIMPLE_MAP(size_t, scm_to_ulong, scm_from_ulong, integer);
 SIMPLE_MAP(float, scm_to_double, scm_from_double, real);
 SIMPLE_MAP(double, scm_to_double, scm_from_double, real);
// SIMPLE_MAP(char *, SWIG_scm2str, SWIG_str02scm, string);
// SIMPLE_MAP(const char *, SWIG_scm2str, SWIG_str02scm, string);

/* Define long long typemaps -- uses functions that are only defined
   in recent versions of Guile, availability also depends on Guile's
   configuration. */

SIMPLE_MAP(long long, scm_to_long_long, scm_from_long_long, integer);
SIMPLE_MAP(unsigned long long, scm_to_ulong_long, scm_from_ulong_long, integer);

/* Strings */

 %typemap (in,     doc="$NAME is a string")      char *(int must_free = 0) {
  $1 = ($1_ltype)SWIG_scm2str($input);
  must_free = 1;
 }
 %typemap (varin,  doc="NEW-VALUE is a string")  char * {$1 = ($1_ltype)SWIG_scm2str($input);}
 %typemap (out,    doc="<string>")              char * {$result = SWIG_str02scm((const char *)$1);}
 %typemap (varout, doc="<string>")              char * {$result = SWIG_str02scm($1);}
 %typemap (in, doc="$NAME is a string")          char **INPUT(char * temp, int must_free = 0) {
   temp = (char *) SWIG_scm2str($input); $1 = &temp;
   must_free = 1;
 }
 %typemap (in,numinputs=0)  char **OUTPUT (char * temp)
   {$1 = &temp;}
 %typemap (argout,doc="$NAME (a string)") char **OUTPUT
   {SWIG_APPEND_VALUE(SWIG_str02scm(*$1));}
 %typemap (in)          char **BOTH = char **INPUT;
 %typemap (argout)      char **BOTH = char **OUTPUT;
 %typemap (in)          char **INOUT = char **INPUT;
 %typemap (argout)      char **INOUT = char **OUTPUT;

/* SWIG_scm2str makes a malloc'ed copy of the string, so get rid of it after
   the function call. */

%typemap (freearg) char * "if (must_free$argnum && $1) SWIG_free($1);";
%typemap (freearg) char **INPUT, char **BOTH "if (must_free$argnum && (*$1)) SWIG_free(*$1);"
%typemap (freearg) char **OUTPUT "SWIG_free(*$1);"
  
/* But this shall not apply if we try to pass a single char by
   reference. */

%typemap (freearg) char *OUTPUT, char *BOTH "";

/* If we set a string variable, delete the old result first, unless const. */

%typemap (varin) char * {
    if ($1) free($1);
    $1 = ($1_ltype) SWIG_scm2str($input);
}

%typemap (varin) const char * {
    $1 = ($1_ltype) SWIG_scm2str($input);
}

%typemap(throws) char * {
  scm_throw(scm_from_locale_symbol((char *) "swig-exception"),
	    scm_list_n(SWIG_str02scm($1), SCM_UNDEFINED));
}

/* Void */

%typemap (out,doc="") void "gswig_result = SCM_UNSPECIFIED;";

/* SCM is passed through */

typedef unsigned long SCM;
%typemap (in) SCM "$1=$input;";
%typemap (out) SCM "$result=$1;";
%typecheck(SWIG_TYPECHECK_POINTER) SCM "$1=1;";

/* ------------------------------------------------------------
 * String & length
 * ------------------------------------------------------------ */

%typemap(in) (char *STRING, int LENGTH), (char *STRING, size_t LENGTH) {
    size_t temp;
    $1 = ($1_ltype) SWIG_Guile_scm2newstr($input, &temp);
    $2 = ($2_ltype) temp;
}

/* ------------------------------------------------------------
 * CLASS::* (member function pointer) typemaps
 * taken from typemaps/swigtype.swg
 * ------------------------------------------------------------ */

#define %set_output(obj)                  $result = obj
#define %set_varoutput(obj)               $result = obj
#define %argument_fail(code, type, name, argn)	scm_wrong_type_arg((char *) FUNC_NAME, argn, $input);
#define %as_voidptr(ptr)		(void*)(ptr)

%typemap(in) SWIGTYPE (CLASS::*) {  
  int res = SWIG_ConvertMember($input, %as_voidptr(&$1), sizeof($type),$descriptor);
  if (!SWIG_IsOK(res)) {
    %argument_fail(res,"$type",$symname, $argnum); 
  }
}

%typemap(out,noblock=1) SWIGTYPE (CLASS::*) {
  %set_output(SWIG_NewMemberObj(%as_voidptr(&$1), sizeof($type), $descriptor));
}

%typemap(varin) SWIGTYPE (CLASS::*) {
  int res = SWIG_ConvertMember($input,%as_voidptr(&$1), sizeof($type), $descriptor);
  if (!SWIG_IsOK(res)) {
    scm_wrong_type_arg((char *) FUNC_NAME, 1, $input);
  }
}

%typemap(varout,noblock=1) SWIGTYPE (CLASS::*) {
  %set_varoutput(SWIG_NewMemberObj(%as_voidptr(&$1), sizeof($type), $descriptor));
}

/* ------------------------------------------------------------
 * Typechecking rules
 * ------------------------------------------------------------ */

/* adapted from python.swg */

%typecheck(SWIG_TYPECHECK_INTEGER)
	 int, short, long,
 	 unsigned int, unsigned short, unsigned long,
	 signed char, unsigned char,
	 long long, unsigned long long,
         size_t, ptrdiff_t,
         std::size_t, std::ptrdiff_t,
	 const int &, const short &, const long &,
 	 const unsigned int &, const unsigned short &, const unsigned long &,
	 const long long &, const unsigned long long &,
         const size_t &, const ptrdiff_t &,
         const std::size_t &, const std::ptrdiff_t &,
	 enum SWIGTYPE
{
  $1 = scm_is_true(scm_integer_p($input)) && scm_is_true(scm_exact_p($input))? 1 : 0;
}

%typecheck(SWIG_TYPECHECK_BOOL)
	bool, bool&, const bool&
{
  $1 = SCM_BOOLP($input) ? 1 : 0;
}

%typecheck(SWIG_TYPECHECK_DOUBLE)
	float, double,
	const float &, const double &
{
  $1 = scm_is_true(scm_real_p($input)) ? 1 : 0;
}

%typecheck(SWIG_TYPECHECK_CHAR) char {
  $1 = SCM_CHARP($input) ? 1 : 0;
}

%typecheck(SWIG_TYPECHECK_STRING) char * {
  $1 = scm_is_string($input) ? 1 : 0;
}

%typecheck(SWIG_TYPECHECK_POINTER) SWIGTYPE *, SWIGTYPE [] {
  void *ptr;
  int res = SWIG_ConvertPtr($input, &ptr, $1_descriptor, 0);
  $1 = SWIG_CheckState(res);
}

%typecheck(SWIG_TYPECHECK_POINTER) SWIGTYPE &, SWIGTYPE && {
  void *ptr;
  int res = SWIG_ConvertPtr($input, &ptr, $1_descriptor, SWIG_POINTER_NO_NULL);
  $1 = SWIG_CheckState(res);
}

%typecheck(SWIG_TYPECHECK_POINTER) SWIGTYPE {
  void *ptr;
  int res = SWIG_ConvertPtr($input, &ptr, $&descriptor, SWIG_POINTER_NO_NULL);
  $1 = SWIG_CheckState(res);
}

%typecheck(SWIG_TYPECHECK_VOIDPTR) void * {
  void *ptr;
  int res = SWIG_ConvertPtr($input, &ptr, 0, 0);
  $1 = SWIG_CheckState(res);
}

/* Array reference typemaps */
%apply SWIGTYPE & { SWIGTYPE ((&)[ANY]) }
%apply SWIGTYPE && { SWIGTYPE ((&&)[ANY]) }

/* const pointers */
%apply SWIGTYPE * { SWIGTYPE *const }
%apply SWIGTYPE (CLASS::*) { SWIGTYPE (CLASS::*const) }
%apply SWIGTYPE & { SWIGTYPE (CLASS::*const&) }

/* typemaps.i ends here */
