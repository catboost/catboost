%insert(runtime) %{
#if defined(__GNUC__) && defined(_WIN32) && !defined(SWIG_PYTHON_NO_HYPOT_WORKAROUND)
/* Workaround for '::hypot' has not been declared', see https://bugs.python.org/issue11566 */
# include <math.h>
#endif

#if !defined(PY_SSIZE_T_CLEAN) && !defined(SWIG_NO_PY_SSIZE_T_CLEAN)
#define PY_SSIZE_T_CLEAN
#endif

#if __GNUC__ >= 7
#pragma GCC diagnostic push
#if defined(__cplusplus) && __cplusplus >=201703L
#pragma GCC diagnostic ignored "-Wregister" /* For python-2.7 headers that use register */
#endif
#endif

#if defined(_DEBUG) && defined(SWIG_PYTHON_INTERPRETER_NO_DEBUG)
/* Use debug wrappers with the Python release dll */

#if defined(_MSC_VER) && _MSC_VER >= 1929
/* Workaround compilation errors when redefining _DEBUG in MSVC 2019 version 16.10 and later
 * See https://github.com/swig/swig/issues/2090 */
# include <corecrt.h>
#endif

# undef _DEBUG
# include <Python.h>
# define _DEBUG 1
#else
# include <Python.h>
#endif

#if __GNUC__ >= 7
#pragma GCC diagnostic pop
#endif
%}

%insert(runtime) "swigrun.swg";         /* SWIG API */
%insert(runtime) "swigerrors.swg";      /* SWIG errors */   
%insert(runtime) "pyhead.swg";          /* Python includes and fixes */
%insert(runtime) "pyerrors.swg";        /* Python errors */
%insert(runtime) "pythreads.swg";       /* Python thread code */
%insert(runtime) "pyapi.swg";           /* Python API */
%insert(runtime) "pyrun.swg";           /* Python run-time code */

#if defined(SWIGPYTHON_BUILTIN)
%insert(runtime) "builtin.swg";		/* Specialization for classes with single inheritance */
#endif
