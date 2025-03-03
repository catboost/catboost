/* -----------------------------------------------------------------------------
 * const.i
 *
 * Typemaps for constants
 * ----------------------------------------------------------------------------- */

%typemap(consttab) int,
                   unsigned int,
                   short,
                   unsigned short,
                   long,
                   unsigned long,
                   unsigned char,
                   signed char,
                   enum SWIGTYPE
  "SWIG_LONG_CONSTANT($symname, ($1_type)$value);";

%typemap(consttab) bool
  "SWIG_BOOL_CONSTANT($symname, ($1_type)$value);";

%typemap(consttab) float,
                   double
  "SWIG_DOUBLE_CONSTANT($symname, $value);";

%typemap(consttab) char
  "SWIG_CHAR_CONSTANT($symname, $value);";

%typemap(consttab) char *,
                   const char *,
                   char [],
                   const char []
  "SWIG_STRING_CONSTANT($symname, $value);";

%typemap(consttab) SWIGTYPE *,
                   SWIGTYPE &,
                   SWIGTYPE &&,
                   SWIGTYPE [] {
  zend_constant c;
  SWIG_SetPointerZval(&c.value, (void*)$value, $1_descriptor, 0);
  zval_copy_ctor(&c.value);
  c.name = zend_string_init("$symname", sizeof("$symname") - 1, 0);
  SWIG_ZEND_CONSTANT_SET_FLAGS(&c, CONST_CS, module_number);
  zend_register_constant(&c);
}

/* Handled as a global variable. */
%typemap(consttab) SWIGTYPE (CLASS::*) "";
