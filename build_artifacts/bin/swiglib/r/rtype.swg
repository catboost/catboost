
/* These map the primitive C types to the appropriate R type
   for use in class representations. 
 */

%typemap("rtype") int, int *, int &      "integer";
%typemap("rtype") long, long *, long &      "integer";
%typemap("rtype") float, float*, float & "numeric";
%typemap("rtype") double, double*, double & "numeric";
%typemap("rtype") char *, char ** "character";
%typemap("rtype") char            "character";
%typemap("rtype") string, string *, string & "character";
%typemap("rtype") std::string, std::string *, std::string & "character";
%typemap("rtype") bool, bool *    "logical";
%typemap("rtype") enum SWIGTYPE   "character";
%typemap("rtype") enum SWIGTYPE *   "character";
%typemap("rtype") enum SWIGTYPE *const   "character";
%typemap("rtype") enum SWIGTYPE &  "character";
%typemap("rtype") const enum SWIGTYPE &  "character";
%typemap("rtype") enum SWIGTYPE &&  "character";
%typemap("rtype") SWIGTYPE * "$R_class";
%typemap("rtype") SWIGTYPE *const "$R_class";
%typemap("rtype") SWIGTYPE & "$R_class";
%typemap("rtype") SWIGTYPE && "$R_class";
%typemap("rtype") SWIGTYPE "$&R_class";

%typemap("rtypecheck") int, int &, long, long &
  %{ (is.integer($arg) || is.numeric($arg)) && length($arg) == 1 %}
%typemap("rtypecheck") int *, long *
  %{ is.integer($arg) || is.numeric($arg) %}


%typemap("rtypecheck") float, double
  %{ is.numeric($arg) && length($arg) == 1 %}
%typemap("rtypecheck") float *, double *
  %{ is.numeric($arg) %}

%typemap("rtypecheck") bool, bool &
  %{ is.logical($arg) && length($arg) == 1 %}
%typemap("rtypecheck") bool *
  %{ is.logical($arg) %}

/*
  Set up type checks to insure overloading precedence.
  We would like non pointer items to shadow pointer items, so that 
  they get called if length = 1
*/

%typecheck(SWIG_TYPECHECK_BOOL) bool {}
%typecheck(SWIG_TYPECHECK_UINT32) unsigned int {}
%typecheck(SWIG_TYPECHECK_INTEGER) int {}
%typecheck(SWIG_TYPECHECK_FLOAT) float {}
%typecheck(SWIG_TYPECHECK_DOUBLE) double {}

%typecheck(SWIG_TYPECHECK_BOOL_PTR) bool * {}
%typecheck(SWIG_TYPECHECK_INT32_PTR) int * {}
%typecheck(SWIG_TYPECHECK_FLOAT_PTR) float * {}
%typecheck(SWIG_TYPECHECK_DOUBLE_PTR) double * {}
%typecheck(SWIG_TYPECHECK_CHAR_PTR) char * {}

%typecheck(SWIG_TYPECHECK_INT32_ARRAY) int[ANY] {}
%typecheck(SWIG_TYPECHECK_FLOAT_ARRAY) float[ANY] {}
%typecheck(SWIG_TYPECHECK_DOUBLE_ARRAY) double [ANY] {}

/* Have to be careful that as(x, "numeric") is different from as.numeric(x).  
   The latter makes a REALSXP, whereas the former leaves an INTSXP as an
   INTSXP.
*/

/* Force coercion of integer, since by default R sets all constants to
   numeric, which means that you can't directly call a function with an 
   integer using an R numercal literal */

%typemap(scoercein) int, int *, int &
  %{  $input = as.integer($input);     %}
%typemap(scoercein) long, long *, long &
  %{  $input = as.integer($input);     %}
%typemap(scoercein) float, float*, float &,
  double, double *, double &
  %{ %}
%typemap(scoercein) char, char *, char &
  %{  $input = as($input, "character");     %}
%typemap(scoercein) string, string *, string &
  %{  $input = as($input, "character");     %}
%typemap(scoercein) std::string, std::string *, std::string &
  %{  $input = as($input, "character");     %}
%typemap(scoercein) enum SWIGTYPE 
  %{  $input = enumToInteger($input, "$R_class"); %}
%typemap(scoercein) enum SWIGTYPE &
  %{  $input = enumToInteger($input, "$*R_class"); %}
%typemap(scoercein) enum SWIGTYPE *
  %{  $input = enumToInteger($input, "$R_class"); %}
%typemap(scoercein) enum SWIGTYPE *const
  %{  $input = enumToInteger($input, "$R_class"); %}

%typemap(scoercein) SWIGTYPE, SWIGTYPE *, SWIGTYPE *const, SWIGTYPE &, SWIGTYPE &&
 %{ if (inherits($input, "ExternalReference")) $input = slot($input,"ref"); %}

/*
%typemap(scoercein) SWIGTYPE *, SWIGTYPE *const
  %{ $input = coerceIfNotSubclass($input, "$R_class"); %}

%typemap(scoercein) SWIGTYPE & 
  %{ $input = coerceIfNotSubclass($input, "$R_class"); %}

%typemap(scoercein) SWIGTYPE && 
  %{ $input = coerceIfNotSubclass($input, "$R_class"); %}

%typemap(scoercein) SWIGTYPE  
  %{ $input = coerceIfNotSubclass($input, "$&R_class"); %}
*/

%typemap(scoercein) SWIGTYPE[ANY]  
 %{
    if(is.list($input))
      assert(all(sapply($input, class) == "$R_class"));
 %}


/* **************************************************************** */

%typemap(scoercein) bool, bool *, bool & 
		    "$input = as.logical($input);";
%typemap(scoercein) int, 
                    int *, 
		    int &,
		    long,
		    long *,
		    long &
  "$input = as.integer($input);";

%typemap(scoercein) char *, string, std::string,
string &, std::string &
%{  $input = as($input, "character"); %}

%typemap(scoerceout) enum SWIGTYPE 
  %{  $result = enumFromInteger($result, "$R_class"); %}

%typemap(scoerceout) enum SWIGTYPE &
  %{  $result = enumFromInteger($result, "$*R_class"); %}

%typemap(scoerceout) enum SWIGTYPE &&
  %{  $result = enumFromInteger($result, "$R_class"); %}

%typemap(scoerceout) enum SWIGTYPE *
  %{  $result = enumToInteger($result, "$R_class"); %}

%typemap(scoerceout) enum SWIGTYPE *const
  %{  $result = enumToInteger($result, "$R_class"); %}

%typemap(scoerceout) SEXP %{ %}

%typemap(scoerceout) SWIGTYPE
  %{ $result <- if (is.null($result)) $result
  else new("$&R_class", ref=$result); %}
 
%typemap(scoerceout) SWIGTYPE &
  %{ $result <- if (is.null($result)) $result
  else new("$R_class", ref=$result); %}

 
%typemap(scoerceout) SWIGTYPE &&
  %{ $result <- if (is.null($result)) $result
  else new("$R_class", ref=$result); %}
 
%typemap(scoerceout) SWIGTYPE *
  %{ $result <- if (is.null($result)) $result
  else new("$R_class", ref=$result); %}

 
%typemap(scoerceout) SWIGTYPE *const
  %{ $result <- if (is.null($result)) $result
  else new("$R_class", ref=$result); %}


/* Override the SWIGTYPE * above. */
%typemap(scoerceout) char,
		     char *,
		     char &,
                     float,
		     double,
		     float*,
		     double*,
		     float &,
		     double &,
                     int,
		     int &,
		     long,
		     long &,
                     bool,
		     bool &,
		     string,
		     std::string,
		     string &,
		     std::string &, 
		     void,
		     signed int,
		     signed int &,		     
		     unsigned int,
		     unsigned int &,		     
		     short,
		     short &,		     
		     unsigned short,
		     unsigned short &,		     
		     long long,
		     signed long long,
		     signed long long &,		     
		     unsigned long long,
		     unsigned long long &,		     
		     signed long,
		     signed long &,		     
		     unsigned long,
		     unsigned long &,
		     signed char,
		     signed char &,
		     unsigned char,
		     unsigned char &
 %{    %}

%apply int {size_t,
std::size_t,
ptrdiff_t,
std::ptrdiff_t,
signed int,
unsigned int,
short,
unsigned short,
signed char,
unsigned char}

%apply int* {size_t[],
std::size_t[],
ptrdiff_t[],
std::ptrdiff_t[],
signed int[],
unsigned int[],
short[],
unsigned short[],
signed char[],
unsigned char[]}

%apply int* {size_t[ANY],
std::size_t[ANY],
ptrdiff_t[ANY],
std::ptrdiff_t[ANY],
signed int[ANY],
unsigned int[ANY],
short[ANY],
unsigned short[ANY],
signed char[ANY],
unsigned char[ANY]}

%apply int* {size_t*,
std::size_t*,
ptrdiff_t*,
std::ptrdiff_t*,
signed int*,
unsigned int*,
short*,
unsigned short*,
signed char*,
unsigned char*}

%apply long {
       long long,
       signed long long,
       unsigned long long,
       signed long,
       unsigned long}

%apply long* {
       long long*,
       signed long long*,
       unsigned long long*,
       signed long*,
       unsigned long*,
       long long[],
       signed long long[],
       unsigned long long[],
       signed long[],
       unsigned long[],
       long long[ANY],
       signed long long[ANY],
       unsigned long long[ANY],
       signed long[ANY],
       unsigned long[ANY]}

%apply float* {
       float[],
       float[ANY]
}
%apply double * {
       double[],
       double[ANY]
}

%apply bool* {
       bool[],
       bool[ANY]
}

#if 0
 Just examining the values for a SWIGTYPE.

%typemap(scoerceout) SWIGTYPE  %{

  name = $1_name
  type = $1_type
  ltype = $1_ltype

  mangle = $1_mangle
  descriptor = $1_descriptor

  pointer type = $*1_type
  pointer ltype = $*1_ltype

  pointer descriptor = $*1_descriptor
  basetype = $*_basetype

%}
#endif


