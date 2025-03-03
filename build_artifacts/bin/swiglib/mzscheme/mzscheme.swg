/* -----------------------------------------------------------------------------
 * mzscheme.swg
 *
 * SWIG Configuration File for MzScheme.
 * This file is parsed by SWIG before reading any other interface file.
 * ----------------------------------------------------------------------------- */

/* Include headers */
%runtime "swigrun.swg"     // Common C API type-checking code
%runtime "swigerrors.swg"  // SWIG errors
%runtime "mzrun.swg"

%define SWIG_APPEND_VALUE(value)
   values[lenv++] = value
%enddef

/* Definitions */
#define SWIG_malloc(size) swig_malloc(size, FUNC_NAME)
#define SWIG_free(mem) free(mem)

#define SWIG_convert_short(o)					\
  SWIG_convert_integer(o, - (1 << (8 * sizeof(short) - 1)),	\
		       (1 << (8 * sizeof(short) - 1)) - 1,	\
		       FUNC_NAME, $argnum-1, argc, argv)
#define SWIG_convert_int(o)					\
  SWIG_convert_integer(o, INT_MIN, INT_MAX,			\
		       FUNC_NAME, $argnum-1, argc, argv)
#define SWIG_convert_long(o)					\
  SWIG_convert_integer(o, LONG_MIN, LONG_MAX,			\
		       FUNC_NAME, $argnum-1, argc, argv)
#define SWIG_convert_unsigned_short(o)				\
  SWIG_convert_unsigned_integer(o, 0,				\
		       (1 << (8 * sizeof(short))) - 1,		\
		       FUNC_NAME, $argnum-1, argc, argv)
#define SWIG_convert_unsigned_int(o)				\
  SWIG_convert_unsigned_integer(o, 0, UINT_MAX,			\
		       FUNC_NAME, $argnum-1, argc, argv)
#define SWIG_convert_unsigned_long(o)				\
  SWIG_convert_unsigned_integer(o, 0, ULONG_MAX,		\
		       FUNC_NAME, $argnum-1, argc, argv)

/* Guile compatibility kludges */
#define SCM_VALIDATE_VECTOR(argnum, value) (void)0
#define SCM_VALIDATE_LIST(argnum, value) (void)0

/* Read in standard typemaps. */
%include <typemaps.i>

%insert(init) "swiginit.swg"

%init %{
Scheme_Object *scheme_reload(Scheme_Env *env) {
  Scheme_Env *menv = SWIG_MZSCHEME_CREATE_MENV(env);

  SWIG_InitializeModule((void *) env);
%}
