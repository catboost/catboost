/* -----------------------------------------------------------------------------
 * globalvar.i
 *
 * Global variables - add the variable to PHP
 * ----------------------------------------------------------------------------- */

%typemap(varinit) char *
{
  zval z_var;
  if ($1) {
    ZVAL_STRING(&z_var, $1);
  } else {
    ZVAL_STR(&z_var, ZSTR_EMPTY_ALLOC());
  }
  zend_hash_str_add(&EG(symbol_table), "$1", sizeof("$1") - 1, &z_var);
}

%typemap(varinit) char []
{
  zval z_var;
  ZVAL_STRING(&z_var, $1);
  zend_hash_str_add(&EG(symbol_table), "$1", sizeof("$1") - 1, &z_var);
}

%typemap(varinit) int,
	          unsigned int,
                  unsigned short,
                  short,
                  unsigned short,
                  long,
                  unsigned long,
                  signed char,
                  unsigned char,
                  enum SWIGTYPE
{
  zval z_var;
  ZVAL_LONG(&z_var, (long)$1);
  zend_hash_str_add(&EG(symbol_table), "$1", sizeof("$1") - 1, &z_var);
}

%typemap(varinit) bool
{
  zval z_var;
  ZVAL_BOOL(&z_var, ($1)?1:0);
  zend_hash_str_add(&EG(symbol_table), "$1", sizeof("$1") - 1, &z_var);
}

%typemap(varinit) float, double
{
  zval z_var;
  ZVAL_DOUBLE(&z_var, (double)$1);
  zend_hash_str_add(&EG(symbol_table), "$1", sizeof("$1") - 1, &z_var);
}

%typemap(varinit) char
{
  zval z_var;
  char c = $1;
  ZVAL_STRINGL(&z_var, &c, 1);
  zend_hash_str_add(&EG(symbol_table), "$1", sizeof("$1") - 1, &z_var);
}

%typemap(varinit) SWIGTYPE *, SWIGTYPE []
{
  zval z_var;
  SWIG_SetPointerZval(&z_var, (void*)$1, $1_descriptor, 0);
  zend_hash_str_add(&EG(symbol_table), "$1", sizeof("$1") - 1, &z_var);
}

%typemap(varinit) SWIGTYPE, SWIGTYPE &, SWIGTYPE &&
{
  zval z_var;
  SWIG_SetPointerZval(&z_var, (void*)&$1, $&1_descriptor, 0);
  zend_hash_str_add(&EG(symbol_table), "$1", sizeof("$1") - 1, &z_var);
}

%typemap(varinit) char [ANY]
{
  zval z_var;
  /* varinit char [ANY] */
  ZVAL_STRINGL(&z_var, $1, $1_dim0);
  zend_hash_str_add(&EG(symbol_table), "$1", sizeof("$1") - 1, &z_var);
}

%typemap(varinit, fragment="swig_php_init_member_ptr") SWIGTYPE (CLASS::*)
{
  zval resource;
  void * p = emalloc(sizeof($1));
  memcpy(p, &$1, sizeof($1));
  ZVAL_RES(&resource, zend_register_resource(p, swig_member_ptr));
  zend_hash_str_add(&EG(symbol_table), "$1", sizeof("$1") - 1, &resource);
}

%typemap(varin) int, unsigned int, short, unsigned short, long, unsigned long, signed char, unsigned char,  enum SWIGTYPE
{
  zval *z_var = zend_hash_str_find(&EG(symbol_table), "$1", sizeof("$1") - 1);
  $1 = zval_get_long(z_var);
}

%typemap(varin) bool
{
  zval *z_var = zend_hash_str_find(&EG(symbol_table), "$1", sizeof("$1") - 1);
  convert_to_boolean(z_var);
  $1 = (Z_TYPE_P(z_var) == IS_TRUE);
}

%typemap(varin) double,float
{
  zval *z_var = zend_hash_str_find(&EG(symbol_table), "$1", sizeof("$1") - 1);
  $1 = zval_get_double(z_var);
}

%typemap(varin) char
{
  zval *z_var = zend_hash_str_find(&EG(symbol_table), "$1", sizeof("$1") - 1);
  convert_to_string(z_var);
  if ($1 != Z_STRVAL_P(z_var)[0]) {
    $1 = Z_STRVAL_P(z_var)[0];
  }
}

%typemap(varin) char *
{
  zval *z_var = zend_hash_str_find(&EG(symbol_table), "$1", sizeof("$1") - 1);
  char *s1;
  convert_to_string(z_var);
  s1 = Z_STRVAL_P(z_var);
  if ((s1 == NULL) || ($1 == NULL) || strcmp(s1, $1)) {
    if (s1)
      $1 = estrdup(s1);
    else
      $1 = NULL;
  }
}


%typemap(varin) SWIGTYPE []
{
  if ($1) {
    zval *z_var = zend_hash_str_find(&EG(symbol_table), "$1", sizeof("$1") - 1);
    SWIG_SetPointerZval(z_var, (void*)$1, $1_descriptor, $owner);
  }
}

%typemap(varin) char [ANY]
{
  zval **z_var;
  char *s1;

  zend_hash_str_find(&EG(symbol_table), "$1", sizeof("$1") - 1, (void**)&z_var);
  s1 = Z_STRVAL_P(z_var);
  if ((s1 == NULL) || ($1 == NULL) || strcmp(s1, $1)) {
    if (s1)
      strncpy($1, s1, $1_dim0);
  }
}

%typemap(varin) SWIGTYPE
{
  zval *z_var;
  $&1_ltype _temp;

  z_var = zend_hash_str_find(&EG(symbol_table), "$1", sizeof("$1") - 1);
  if (SWIG_ConvertPtr(z_var, (void**)&_temp, $&1_descriptor, 0) < 0) {
    SWIG_PHP_Error(E_ERROR,"Type error in value of $symname. Expected $&1_descriptor");
  }

  $1 = *($&1_ltype)_temp;
}

%typemap(varin) SWIGTYPE *, SWIGTYPE &, SWIGTYPE &&
{
  zval *z_var;
  $1_ltype _temp;

  z_var = zend_hash_str_find(&EG(symbol_table), "$1", sizeof("$1") - 1);
  if (SWIG_ConvertPtr(z_var, (void **)&_temp, $1_descriptor, 0) < 0) {
    SWIG_PHP_Error(E_ERROR,"Type error in value of $symname. Expected $&1_descriptor");
  }

  $1 = ($1_ltype)_temp;
}

%typemap(varin, fragment="swig_php_init_member_ptr") SWIGTYPE (CLASS::*)
{
  zval *z_var = zend_hash_str_find(&EG(symbol_table), "$1", sizeof("$1") - 1);
  void * p = (void*)zend_fetch_resource_ex(z_var, SWIG_MEMBER_PTR, swig_member_ptr);
  memcpy(&$1, p, sizeof($1));
}

%typemap(varout) int,
                 unsigned int,
                 unsigned short,
                 short,
                 long,
                 unsigned long,
                 signed char,
                 unsigned char,
                 enum SWIGTYPE
{
  zval *z_var = zend_hash_str_find(&EG(symbol_table), "$1", sizeof("$1") - 1);
  if ($1 != ($1_ltype)Z_LVAL_P(z_var)) {
    z_var->value.lval = (long)$1;
  }
}

//SAMFIX need to cast zval->type, what if zend-hash_find fails? etc?
%typemap(varout) bool
{
  zval *z_var = zend_hash_str_find(&EG(symbol_table), "$1", sizeof("$1") - 1);
  if ($1 != ($1_ltype)Z_LVAL_P(z_var)) {
    z_var->value.lval = (long)$1;
  }
}

%typemap(varout) double, float
{
  zval *z_var = zend_hash_str_find(&EG(symbol_table), "$1", sizeof("$1") - 1);
  if ($1 != ($1_ltype)Z_DVAL_P(z_var)) {
    z_var->value.dval = (double)$1;
  }
}

%typemap(varout) char
{
  zval *z_var = zend_hash_str_find(&EG(symbol_table), "$1", sizeof("$1") - 1);
  char c = $1;
  if ($1 != Z_STRVAL_P(z_val)[0]) {
    ZVAL_STRING(z_var, &c);
  }
}

%typemap(varout) char *
{
  zval *z_var = zend_hash_str_find(&EG(symbol_table), "$1", sizeof("$1") - 1);
  const char *s1 = Z_STRVAL_P(z_var);
  if ((s1 == NULL) || ($1 == NULL) || strcmp(s1, $1)) {
    if (s1)
      efree(s1);
    if ($1) {
      (z_var)->value.str.val = estrdup($1);
      (z_var)->value.str.len = strlen($1) + 1;
    } else {
      (z_var)->value.str.val = 0;
      (z_var)->value.str.len = 0;
    }
  }
}

%typemap(varout) SWIGTYPE
{
  zval *z_var = zend_hash_str_find(&EG(symbol_table), "$1", sizeof("$1") - 1);
  SWIG_SetPointerZval(z_var, (void*)&$1, $&1_descriptor, 0);
}

%typemap(varout) SWIGTYPE []
{
  if($1) {
    zval *z_var = zend_hash_str_find(&EG(symbol_table), "$1", sizeof("$1") - 1);
    SWIG_SetPointerZval(z_var, (void*)$1, $1_descriptor, 0);
  }
}

%typemap(varout) char [ANY]
{
  zval *z_var = zend_hash_str_find(&EG(symbol_table), "$1", sizeof("$1") - 1);
  const char *s1 = Z_STRVAL_P(z_var);
deliberate error cos this code looks bogus to me
  if ((s1 == NULL) || strcmp(s1, $1)) {
    if ($1) {
      (z_var)->value.str.val = estrdup($1);
      (z_var)->value.str.len = strlen($1) + 1;
    } else {
      (z_var)->value.str.val = 0;
      (z_var)->value.str.len = 0;
    }
  }
}

%typemap(varout) SWIGTYPE *, SWIGTYPE &, SWIGTYPE &&
{
  zval *z_var = zend_hash_str_find(&EG(symbol_table), "$1", sizeof("$1") - 1);
  SWIG_SetPointerZval(z_var, (void*)$1, $1_descriptor, 0);
}

%typemap(varout, fragment="swig_php_init_member_ptr") SWIGTYPE (CLASS::*)
{
  zval resource;
  void * p = emalloc(sizeof($1));
  memcpy(p, &$1, sizeof($1));
  ZVAL_RES(&resource, zend_register_resource(p, swig_member_ptr));
  zend_hash_str_add(&EG(symbol_table), "$1", sizeof("$1") - 1, &resource);
}
