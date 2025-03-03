/* -----------------------------------------------------------------------------
 * typemaps.i.
 *
 * SWIG Typemap library for PHP.
 *
 * This library provides standard typemaps for modifying SWIG's behavior.
 * With enough entries in this file, I hope that very few people actually
 * ever need to write a typemap.
 *
 * Define macros to define the following typemaps:
 *
 * TYPE *INPUT.   Argument is passed in as native variable by value.
 * TYPE *OUTPUT.  Argument is returned as an array from the function call.
 * TYPE *INOUT.   Argument is passed in by value, and out as part of returned list
 * TYPE *REFERENCE.  Argument is passed in as native variable with value
 *                   semantics.  Variable value is changed with result.
 *                   Use like this:
 *                   int foo(int *REFERENCE);
 *
 *                   $a = 0;
 *                   $rc = foo($a);
 *
 *                   Even though $a looks like it's passed by value,
 *                   its value can be changed by foo().
 * ----------------------------------------------------------------------------- */

%define BOOL_TYPEMAP(TYPE)
%typemap(in) TYPE *INPUT(TYPE temp), TYPE &INPUT(TYPE temp)
%{
  convert_to_boolean(&$input);
  temp = (Z_TYPE($input) == IS_TRUE);
  $1 = &temp;
%}
%typemap(argout) TYPE *INPUT, TYPE &INPUT "";
%typemap(in,numinputs=0) TYPE *OUTPUT(TYPE temp), TYPE &OUTPUT(TYPE temp) "$1 = &temp;";
%typemap(argout,fragment="t_output_helper") TYPE *OUTPUT, TYPE &OUTPUT
{
  zval o;
  ZVAL_BOOL(&o, temp$argnum);
  t_output_helper($result, &o);
}
%typemap(in) TYPE *REFERENCE (TYPE lvalue), TYPE &REFERENCE (TYPE lvalue)
%{
  convert_to_boolean($input);
  lvalue = (Z_TYPE_P($input) == IS_TRUE);
  $1 = &lvalue;
%}
%typemap(argout) TYPE *REFERENCE, TYPE &REFERENCE
%{
  ZVAL_BOOL(&$arg, lvalue$argnum ? true : false);
%}
%enddef

%define DOUBLE_TYPEMAP(TYPE)
%typemap(in) TYPE *INPUT(TYPE temp), TYPE &INPUT(TYPE temp)
%{
  temp = (TYPE) zval_get_double(&$input);
  $1 = &temp;
%}
%typemap(argout) TYPE *INPUT, TYPE &INPUT "";
%typemap(in,numinputs=0) TYPE *OUTPUT(TYPE temp), TYPE &OUTPUT(TYPE temp) "$1 = &temp;";
%typemap(argout,fragment="t_output_helper") TYPE *OUTPUT, TYPE &OUTPUT
{
  zval o;
  ZVAL_DOUBLE(&o, temp$argnum);
  t_output_helper($result, &o);
}
%typemap(in) TYPE *REFERENCE (TYPE dvalue), TYPE &REFERENCE (TYPE dvalue)
%{
  dvalue = (TYPE) zval_get_double(&$input);
  $1 = &dvalue;
%}
%typemap(argout) TYPE *REFERENCE, TYPE &REFERENCE
%{
  ZVAL_DOUBLE(&$arg, (double)(lvalue$argnum));
%}
%enddef

%define INT_TYPEMAP(TYPE)
%typemap(in) TYPE *INPUT(TYPE temp), TYPE &INPUT(TYPE temp)
%{
  temp = (TYPE) zval_get_long(&$input);
  $1 = &temp;
%}
%typemap(argout) TYPE *INPUT, TYPE &INPUT "";
%typemap(in,numinputs=0) TYPE *OUTPUT(TYPE temp), TYPE &OUTPUT(TYPE temp) "$1 = &temp;";
%typemap(argout,fragment="t_output_helper") TYPE *OUTPUT, TYPE &OUTPUT
{
  zval o;
  ZVAL_LONG(&o, temp$argnum);
  t_output_helper($result, &o);
}
%typemap(in) TYPE *REFERENCE (TYPE lvalue), TYPE &REFERENCE (TYPE lvalue)
%{
  lvalue = (TYPE) zval_get_long(&$input);
  $1 = &lvalue;
%}
%typemap(argout) TYPE *REFERENCE, TYPE &REFERENCE
%{
  ZVAL_LONG(&$arg, (long)(lvalue$argnum));
%}
%enddef

BOOL_TYPEMAP(bool);

DOUBLE_TYPEMAP(float);
DOUBLE_TYPEMAP(double);

INT_TYPEMAP(int);
INT_TYPEMAP(short);
INT_TYPEMAP(long);
INT_TYPEMAP(unsigned int);
INT_TYPEMAP(unsigned short);
INT_TYPEMAP(unsigned long);
INT_TYPEMAP(unsigned char);
INT_TYPEMAP(signed char);

INT_TYPEMAP(long long);
%typemap(argout,fragment="t_output_helper") long long *OUTPUT
{
  zval o;
  if ((long long)LONG_MIN <= temp$argnum && temp$argnum <= (long long)LONG_MAX) {
    ZVAL_LONG(&o, (long)temp$argnum);
  } else {
    char temp[256];
    sprintf(temp, "%lld", (long long)temp$argnum);
    ZVAL_STRING(&o, temp);
  }
  t_output_helper($result, &o);
}
%typemap(in) TYPE *REFERENCE (long long lvalue)
%{
  CONVERT_LONG_LONG_IN(lvalue, long long, $input)
  $1 = &lvalue;
%}
%typemap(argout) long long *REFERENCE
%{
  if ((long long)LONG_MIN <= lvalue$argnum && lvalue$argnum <= (long long)LONG_MAX) {
    ZVAL_LONG(&$arg, (long)temp$argnum);
  } else {
    char temp[256];
    sprintf(temp, "%lld", (long long)lvalue$argnum);
    ZVAL_STRING(&$arg, temp);
  }
%}
%typemap(argout) long long &OUTPUT
%{
  if ((long long)LONG_MIN <= *arg$argnum && *arg$argnum <= (long long)LONG_MAX) {
    ZVAL_LONG($result, (long)(*arg$argnum));
  } else {
    char temp[256];
    sprintf(temp, "%lld", (long long)(*arg$argnum));
    ZVAL_STRING($result, temp);
  }
%}
INT_TYPEMAP(unsigned long long);
%typemap(argout,fragment="t_output_helper") unsigned long long *OUTPUT
{
  zval o;
  if (temp$argnum <= (unsigned long long)LONG_MAX) {
    ZVAL_LONG(&o, temp$argnum);
  } else {
    char temp[256];
    sprintf(temp, "%llu", (unsigned long long)temp$argnum);
    ZVAL_STRING(&o, temp);
  }
  t_output_helper($result, &o);
}
%typemap(in) TYPE *REFERENCE (unsigned long long lvalue)
%{
  CONVERT_UNSIGNED_LONG_LONG_IN(lvalue, unsigned long long, $input)
  $1 = &lvalue;
%}
%typemap(argout) unsigned long long *REFERENCE
%{
  if (lvalue$argnum <= (unsigned long long)LONG_MAX) {
    ZVAL_LONG($arg, (long)(lvalue$argnum));
  } else {
    char temp[256];
    sprintf(temp, "%llu", (unsigned long long)lvalue$argnum);
    ZVAL_STRING((*$arg), temp);
  }
%}
%typemap(argout) unsigned long long &OUTPUT
%{
  if (*arg$argnum <= (unsigned long long)LONG_MAX) {
    ZVAL_LONG($result, (long)(*arg$argnum));
  } else {
    char temp[256];
    sprintf(temp, "%llu", (unsigned long long)(*arg$argnum));
    ZVAL_STRING($result, temp);
  }
%}

%typemap(in) bool *INOUT = bool *INPUT;
%typemap(in) float *INOUT = float *INPUT;
%typemap(in) double *INOUT = double *INPUT;

%typemap(in) int *INOUT = int *INPUT;
%typemap(in) short *INOUT = short *INPUT;
%typemap(in) long *INOUT = long *INPUT;
%typemap(in) long long *INOUT = long long *INPUT;
%typemap(in) unsigned *INOUT = unsigned *INPUT;
%typemap(in) unsigned short *INOUT = unsigned short *INPUT;
%typemap(in) unsigned long *INOUT = unsigned long *INPUT;
%typemap(in) unsigned char *INOUT = unsigned char *INPUT;
%typemap(in) unsigned long long *INOUT = unsigned long long *INPUT;
%typemap(in) signed char *INOUT = signed char *INPUT;

%typemap(in) bool &INOUT = bool *INPUT;
%typemap(in) float &INOUT = float *INPUT;
%typemap(in) double &INOUT = double *INPUT;

%typemap(in) int &INOUT = int *INPUT;
%typemap(in) short &INOUT = short *INPUT;
%typemap(in) long &INOUT = long *INPUT;
%typemap(in) long long &INOUT = long long *INPUT;
%typemap(in) long long &INPUT = long long *INPUT;
%typemap(in) unsigned &INOUT = unsigned *INPUT;
%typemap(in) unsigned short &INOUT = unsigned short *INPUT;
%typemap(in) unsigned long &INOUT = unsigned long *INPUT;
%typemap(in) unsigned char &INOUT = unsigned char *INPUT;
%typemap(in) unsigned long long &INOUT = unsigned long long *INPUT;
%typemap(in) unsigned long long &INPUT = unsigned long long *INPUT;
%typemap(in) signed char &INOUT = signed char *INPUT;

%typemap(argout) bool *INOUT = bool *OUTPUT;
%typemap(argout) float *INOUT = float *OUTPUT;
%typemap(argout) double *INOUT= double *OUTPUT;

%typemap(argout) int *INOUT = int *OUTPUT;
%typemap(argout) short *INOUT = short *OUTPUT;
%typemap(argout) long *INOUT= long *OUTPUT;
%typemap(argout) long long *INOUT= long long *OUTPUT;
%typemap(argout) unsigned short *INOUT= unsigned short *OUTPUT;
%typemap(argout) unsigned long *INOUT = unsigned long *OUTPUT;
%typemap(argout) unsigned char *INOUT = unsigned char *OUTPUT;
%typemap(argout) unsigned long long *INOUT = unsigned long long *OUTPUT;
%typemap(argout) signed char *INOUT = signed char *OUTPUT;

%typemap(argout) bool &INOUT = bool *OUTPUT;
%typemap(argout) float &INOUT = float *OUTPUT;
%typemap(argout) double &INOUT= double *OUTPUT;

%typemap(argout) int &INOUT = int *OUTPUT;
%typemap(argout) short &INOUT = short *OUTPUT;
%typemap(argout) long &INOUT= long *OUTPUT;
%typemap(argout) long long &INOUT= long long *OUTPUT;
%typemap(argout) unsigned short &INOUT= unsigned short *OUTPUT;
%typemap(argout) unsigned long &INOUT = unsigned long *OUTPUT;
%typemap(argout) unsigned char &INOUT = unsigned char *OUTPUT;
%typemap(argout) unsigned long long &INOUT = unsigned long long *OUTPUT;
%typemap(argout) signed char &INOUT = signed char *OUTPUT;

%typemap(in) char INPUT[ANY] ( char temp[$1_dim0] )
%{
  convert_to_string(&$input);
  strncpy(temp, Z_STRVAL($input), $1_dim0);
  $1 = temp;
%}
%typemap(in,numinputs=0) char OUTPUT[ANY] ( char temp[$1_dim0] )
  "$1 = temp;";
%typemap(argout,fragment="t_output_helper") char OUTPUT[ANY]
{
  zval o;
  ZVAL_STRINGL(&o, temp$argnum, $1_dim0);
  t_output_helper($result, &o);
}

%typemap(in,numinputs=0) void **OUTPUT (int force),
                         void *&OUTPUT (int force)
%{
  /* If they pass NULL by reference, make it into a void*
     This bit should go in arginit if arginit support init-ing scripting args */
  if (SWIG_ConvertPtr(&$input, (void **) &$1, $1_descriptor, 0) < 0) {
    /* So... we didn't get a ref or ptr, but we'll accept NULL by reference */
    if (!(Z_ISREF($input) && Z_ISNULL_P(Z_REFVAL($input)))) {
      /* wasn't a pre/ref/thing, OR anything like an int thing */
      SWIG_PHP_Error(E_ERROR, "Type error in argument $arg of $symname.");
    }
  }
  force=0;
  if (arg1==NULL) {
#ifdef __cplusplus
    ptr=new $*1_ltype();
#else
    ptr=($*1_ltype) calloc(1,sizeof($*1_ltype));
#endif
    $1=&ptr;
    /* have to passback arg$arg too */
    force=1;
  }
%}

%typemap(argout) void **OUTPUT,
                 void *&OUTPUT
%{
  if (force$argnum) {  /* pass back arg$argnum through params ($arg) if we can */
    if (!Z_ISREF($arg)) {
      SWIG_PHP_Error(E_WARNING, "Parameter $argnum of $symname wasn't passed by reference");
    } else {
      SWIG_SetPointerZval(*$arg, (void *) ptr$argnum, $*1_descriptor, 1);
    }
  }
%}
