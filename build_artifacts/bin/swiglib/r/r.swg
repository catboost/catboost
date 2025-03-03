/* */


%insert("header") "swiglabels.swg"

%insert("init") "swiginit.swg"
%insert("runtime") "swigrun.swg"
%insert("runtime") "swigerrors.swg"
%insert("runtime") "rrun.swg"

%init %{
SWIGEXPORT void SWIG_init(void) {
%}

%include <rkw.swg>

#define %Rruntime %insert("s")

#define SWIG_Object SEXP
#define VOID_Object R_NilValue

#define %append_output(obj) SET_VECTOR_ELT($result, $n, obj)

%define %set_constant(name, obj) %begin_block
   SEXP _obj = obj;
   assign(name, _obj);
%end_block %enddef

%define %raise(obj,type,desc) 
return R_NilValue;
%enddef

%insert("sinit") "srun.swg"

%insert("sinitroutine") %{
SWIG_init();
SWIG_InitializeModule(0);
%}

%include <typemaps/swigmacros.swg>
%typemap(in) (double *x, int len) %{
   $1 = REAL(x);
   $2 = Rf_length(x);
%}

/* XXX
   Need to worry about inheritance, e.g. if B extends A 
   and we are looking for an A[], then B elements are okay.
*/
%typemap(scheck) SWIGTYPE[ANY]  
  %{ 
#      assert(length($input) > $1_dim0)
      assert(all(sapply($input, class) == "$R_class"));
  %}

%typemap(out) void "";

%typemap(in) int *, int[ANY],
	     signed int *, signed int[ANY],
	     unsigned int *, unsigned int[ANY],
             short *, short[ANY],
             signed short *, signed short[ANY],
             unsigned short *, unsigned short[ANY],
             long *, long[ANY],
             signed long *, signed long[ANY],
             unsigned long *, unsigned long[ANY],
             long long *, long long[ANY],
             signed long long *, signed long long[ANY],
             unsigned long long *, unsigned long long[ANY]
             
{
{ int _rswigi;
  int _rswiglen = LENGTH($input);
  $1 = %static_cast(calloc(sizeof($1_basetype), _rswiglen), $1_ltype);
  for (_rswigi=0; _rswigi< _rswiglen; _rswigi++) {
     $1[_rswigi] = INTEGER($input)[_rswigi];
  }
}
} 

%typemap(in) float *, float[ANY],
             double *, double[ANY]
             
{
{  int _rswigi;
  int _rswiglen = LENGTH($input);
  $1 = %static_cast(calloc(sizeof($1_basetype), _rswiglen), $1_ltype);
  for (_rswigi=0; _rswigi<_rswiglen; _rswigi++) {
     $1[_rswigi] = REAL($input)[_rswigi];
  }
}
}

%typemap(freearg,noblock=1) int *, int[ANY], 
	     signed int *, signed int[ANY],
	     unsigned int *, unsigned int[ANY],
             short *, short[ANY],
             signed short *, signed short[ANY],
             unsigned short *, unsigned short[ANY],
             long *, long[ANY],
             signed long *, signed long[ANY],
             unsigned long *, unsigned long[ANY],
             long long *, long long[ANY],
             signed long long *, signed long long[ANY],
             unsigned long long *, unsigned long long[ANY],
             float *, float[ANY],
             double *, double[ANY]
%{             
  free($1);
%}

%typemap(freearg, noblock=1) int *OUTPUT,
signed int *OUTPUT,
unsigned int *OUTPUT,
short *OUTPUT,
signed short *OUTPUT,
unsigned short *OUTPUT,
long *OUTPUT,
signed long *OUTPUT,
unsigned long *OUTPUT,
long long *OUTPUT,
signed long long *OUTPUT,
unsigned long long *OUTPUT,
float *OUTPUT,
double *OUTPUT,
char *OUTPUT,
signed char *OUTPUT,
unsigned char *OUTPUT
{}



/* Should we recycle to make the length correct.
   And warn if length() > the dimension. 
*/
%typemap(scheck) SWIGTYPE [ANY] %{
#  assert(length($input) >= $1_dim0)
%}

/* Handling vector case to avoid warnings,
   although we just use the first one. */
%typemap(scheck) unsigned int %{
  assert(length($input) == 1 && $input >= 0, "All values must be non-negative");
%}


%typemap(scheck) int, long %{
  if(length($input) > 1) {
     warning("using only the first element of $input");
  };
%}

%include <typemaps/fragments.swg>
%include <rfragments.swg>
%include <ropers.swg>
%include <typemaps/swigtypemaps.swg>
%include <rtype.swg>

%typemap(in,noblock=1) enum SWIGTYPE[ANY] {
   $1 = %reinterpret_cast(INTEGER($input), $1_ltype);
}

%typemap(in,noblock=1,fragment="SWIG_strdup") char * {
   $1 = %reinterpret_cast(SWIG_strdup(CHAR(STRING_ELT($input, 0))), $1_ltype);
}

%typemap(freearg,noblock=1) char * {
   free($1);
}

%typemap(in,noblock=1,fragment="SWIG_strdup") char *[ANY]  {
   $1 = %reinterpret_cast(SWIG_strdup(CHAR(STRING_ELT($input, 0))), $1_ltype);
}

%typemap(freearg,noblock=1) char *[ANY]  {
   free($1);
}

%typemap(in,noblock=1,fragment="SWIG_strdup") char[ANY] {
    $1 = SWIG_strdup(CHAR(STRING_ELT($input, 0)));
}

%typemap(freearg,noblock=1) char[ANY] {
    free($1);
}

%typemap(in,noblock=1,fragment="SWIG_strdup") char[] {
    $1 = SWIG_strdup(CHAR(STRING_ELT($input, 0)));
}

%typemap(freearg,noblock=1) char[] {
    free($1);
}

%typemap(in) const enum SWIGTYPE & ($*1_ltype temp)
%{ temp = ($*1_ltype)INTEGER($input)[0];
   $1 = &temp; %}

%typemap(out) const enum SWIGTYPE & %{ $result = Rf_ScalarInteger((int)*$1); %}

%typemap(memberin) char[] %{
if ($input) strcpy($1, $input);
else
strcpy($1, "");
%}

%typemap(globalin) char[] %{
if ($input) strcpy($1, $input);
else
strcpy($1, "");
%}

%typemap(out,noblock=1) char * 
 {  $result = $1 ? Rf_mkString(%reinterpret_cast($1,char *)) : R_NilValue; }

%typemap(in,noblock=1) char {
$1 = %static_cast(CHAR(STRING_ELT($input, 0))[0],$1_ltype);
}

%typemap(out) char 
 { 
    char tmp[2] = "x";
    tmp[0] = $1;    
    $result = Rf_mkString(tmp); 
 }


%typemap(in,noblock=1) int, long
{
  $1 = %static_cast(INTEGER($input)[0], $1_ltype);
}

%typemap(out,noblock=1) int, long
  "$result = Rf_ScalarInteger($1);";


%typemap(in,noblock=1) bool 
  "$1 = LOGICAL($input)[0] ? true : false;";


%typemap(out,noblock=1) bool 
  "$result = Rf_ScalarLogical($1);";

%typemap(in,noblock=1) 
             float, 
             double
{
  $1 = %static_cast(REAL($input)[0], $1_ltype); 
}

/* Why is this here ? */
/* %typemap(out,noblock=1) unsigned int *
  "$result = ScalarReal(*($1));"; */

%Rruntime %{
setMethod('[', "ExternalReference",
function(x,i,j, ..., drop=TRUE) 
if (!is.null(x$"__getitem__")) 
sapply(i, function(n) x$"__getitem__"(i=as.integer(n-1))))

setMethod('[<-' , "ExternalReference",
function(x,i,j, ..., value) 
if (!is.null(x$"__setitem__")) {
sapply(1:length(i), function(n) 
x$"__setitem__"(i=as.integer(i[n]-1), x=value[n]))
x
})

setAs('ExternalReference', 'character',
function(from) {if (!is.null(from$"__str__")) from$"__str__"()})

suppressMessages(suppressWarnings(setMethod('print', 'ExternalReference',
function(x) {print(as(x, "character"))})))
%}



