/* -----------------------------------------------------------------------------
 * javahead.swg
 *
 * Java support code
 * ----------------------------------------------------------------------------- */


/* JNI function calls require different calling conventions for C and C++. These JCALL macros are used so
 * that the same typemaps can be used for generating code for both C and C++. The SWIG preprocessor can expand
 * the macros thereby generating the correct calling convention. It is thus essential that all typemaps that
 * use the macros are not within %{ %} brackets as they won't be run through the SWIG preprocessor. */
#ifdef __cplusplus
#   define JCALL0(func, jenv) jenv->func()
#   define JCALL1(func, jenv, ar1) jenv->func(ar1)
#   define JCALL2(func, jenv, ar1, ar2) jenv->func(ar1, ar2)
#   define JCALL3(func, jenv, ar1, ar2, ar3) jenv->func(ar1, ar2, ar3)
#   define JCALL4(func, jenv, ar1, ar2, ar3, ar4) jenv->func(ar1, ar2, ar3, ar4)
#   define JCALL5(func, jenv, ar1, ar2, ar3, ar4, ar5) jenv->func(ar1, ar2, ar3, ar4, ar5)
#   define JCALL6(func, jenv, ar1, ar2, ar3, ar4, ar5, ar6) jenv->func(ar1, ar2, ar3, ar4, ar5, ar6)
#   define JCALL7(func, jenv, ar1, ar2, ar3, ar4, ar5, ar6, ar7) jenv->func(ar1, ar2, ar3, ar4, ar5, ar6, ar7)
#else
#   define JCALL0(func, jenv) (*jenv)->func(jenv)
#   define JCALL1(func, jenv, ar1) (*jenv)->func(jenv, ar1)
#   define JCALL2(func, jenv, ar1, ar2) (*jenv)->func(jenv, ar1, ar2)
#   define JCALL3(func, jenv, ar1, ar2, ar3) (*jenv)->func(jenv, ar1, ar2, ar3)
#   define JCALL4(func, jenv, ar1, ar2, ar3, ar4) (*jenv)->func(jenv, ar1, ar2, ar3, ar4)
#   define JCALL5(func, jenv, ar1, ar2, ar3, ar4, ar5) (*jenv)->func(jenv, ar1, ar2, ar3, ar4, ar5)
#   define JCALL6(func, jenv, ar1, ar2, ar3, ar4, ar5, ar6) (*jenv)->func(jenv, ar1, ar2, ar3, ar4, ar5, ar6)
#   define JCALL7(func, jenv, ar1, ar2, ar3, ar4, ar5, ar6, ar7) (*jenv)->func(jenv, ar1, ar2, ar3, ar4, ar5, ar6, ar7)
#endif

%insert(runtime) %{
#include <jni.h>
#include <stdlib.h>
#include <string.h>
%}

%insert(runtime) %{
/* Support for throwing Java exceptions */
typedef enum {
  SWIG_JavaOutOfMemoryError = 1,
  SWIG_JavaIOException,
  SWIG_JavaRuntimeException,
  SWIG_JavaIndexOutOfBoundsException,
  SWIG_JavaArithmeticException,
  SWIG_JavaIllegalArgumentException,
  SWIG_JavaNullPointerException,
  SWIG_JavaDirectorPureVirtual,
  SWIG_JavaUnknownError,
  SWIG_JavaIllegalStateException,
} SWIG_JavaExceptionCodes;

typedef struct {
  SWIG_JavaExceptionCodes code;
  const char *java_exception;
} SWIG_JavaExceptions_t;
%}

%insert(runtime) {
static void SWIGUNUSED SWIG_JavaThrowException(JNIEnv *jenv, SWIG_JavaExceptionCodes code, const char *msg) {
  jclass excep;
  static const SWIG_JavaExceptions_t java_exceptions[] = {
    { SWIG_JavaOutOfMemoryError, "java/lang/OutOfMemoryError" },
    { SWIG_JavaIOException, "java/io/IOException" },
    { SWIG_JavaRuntimeException, "java/lang/RuntimeException" },
    { SWIG_JavaIndexOutOfBoundsException, "java/lang/IndexOutOfBoundsException" },
    { SWIG_JavaArithmeticException, "java/lang/ArithmeticException" },
    { SWIG_JavaIllegalArgumentException, "java/lang/IllegalArgumentException" },
    { SWIG_JavaNullPointerException, "java/lang/NullPointerException" },
    { SWIG_JavaDirectorPureVirtual, "java/lang/RuntimeException" },
    { SWIG_JavaUnknownError,  "java/lang/UnknownError" },
    { SWIG_JavaIllegalStateException, "java/lang/IllegalStateException" },
    { (SWIG_JavaExceptionCodes)0,  "java/lang/UnknownError" }
  };
  const SWIG_JavaExceptions_t *except_ptr = java_exceptions;

  while (except_ptr->code != code && except_ptr->code)
    except_ptr++;

  JCALL0(ExceptionClear, jenv);
  excep = JCALL1(FindClass, jenv, except_ptr->java_exception);
  if (excep)
    JCALL2(ThrowNew, jenv, excep, msg);
}
}

%insert(runtime) %{
/* Contract support */

#define SWIG_contract_assert(nullreturn, expr, msg) do { if (!(expr)) {SWIG_JavaThrowException(jenv, SWIG_JavaIllegalArgumentException, msg); return nullreturn; } } while (0)
%}
