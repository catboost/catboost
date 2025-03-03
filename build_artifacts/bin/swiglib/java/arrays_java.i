/* -----------------------------------------------------------------------------
 * arrays_java.i
 *
 * These typemaps give more natural support for arrays. The typemaps are not efficient
 * as there is a lot of copying of the array values whenever the array is passed to C/C++ 
 * from Java and vice versa. The Java array is expected to be the same size as the C array.
 * An exception is thrown if they are not.
 *
 * Example usage:
 * Wrapping:
 *
 *   %include <arrays_java.i>
 *   %inline %{
 *       short FiddleSticks[3];
 *   %}
 *
 * Use from Java like this:
 *
 *   short[] fs = new short[] {10, 11, 12};
 *   example.setFiddleSticks(fs);
 *   fs = example.getFiddleSticks();
 * ----------------------------------------------------------------------------- */

/* Primitive array support is a combination of SWIG macros and functions in order to reduce 
 * code bloat and aid maintainability. The SWIG preprocessor expands the macros into functions 
 * for inclusion in the generated code. */

/* Array support functions declarations macro */
%define JAVA_ARRAYS_DECL(CTYPE, JNITYPE, JAVATYPE, JFUNCNAME)
%{
static int SWIG_JavaArrayIn##JFUNCNAME (JNIEnv *jenv, JNITYPE **jarr, CTYPE **carr, JNITYPE##Array input);
static void SWIG_JavaArrayArgout##JFUNCNAME (JNIEnv *jenv, JNITYPE *jarr, CTYPE *carr, JNITYPE##Array input);
static JNITYPE##Array SWIG_JavaArrayOut##JFUNCNAME (JNIEnv *jenv, CTYPE *result, jsize sz);
%}
%enddef

/* Array support functions macro */
%define JAVA_ARRAYS_IMPL(CTYPE, JNITYPE, JAVATYPE, JFUNCNAME)
%{
/* CTYPE[] support */
static int SWIG_JavaArrayIn##JFUNCNAME (JNIEnv *jenv, JNITYPE **jarr, CTYPE **carr, JNITYPE##Array input) {
  int i;
  jsize sz;
  if (!input) {
    SWIG_JavaThrowException(jenv, SWIG_JavaNullPointerException, "null array");
    return 0;
  }
  sz = JCALL1(GetArrayLength, jenv, input);
  *jarr = JCALL2(Get##JAVATYPE##ArrayElements, jenv, input, 0);
  if (!*jarr)
    return 0; %}
#ifdef __cplusplus
%{  *carr = new CTYPE[sz]; %}
#else
%{  *carr = (CTYPE*) malloc(sz * sizeof(CTYPE)); %}
#endif
%{  if (!*carr) {
    SWIG_JavaThrowException(jenv, SWIG_JavaOutOfMemoryError, "array memory allocation failed");
    return 0;
  }
  for (i=0; i<sz; i++)
    JAVA_TYPEMAP_ARRAY_ELEMENT_ASSIGN(CTYPE)
  return 1;
}

static void SWIG_JavaArrayArgout##JFUNCNAME (JNIEnv *jenv, JNITYPE *jarr, CTYPE *carr, JNITYPE##Array input) {
  int i;
  jsize sz = JCALL1(GetArrayLength, jenv, input);
  for (i=0; i<sz; i++)
    jarr[i] = (JNITYPE)carr[i];
  JCALL3(Release##JAVATYPE##ArrayElements, jenv, input, jarr, 0);
}

static JNITYPE##Array SWIG_JavaArrayOut##JFUNCNAME (JNIEnv *jenv, CTYPE *result, jsize sz) {
  JNITYPE *arr;
  int i;
  JNITYPE##Array jresult = JCALL1(New##JAVATYPE##Array, jenv, sz);
  if (!jresult)
    return NULL;
  arr = JCALL2(Get##JAVATYPE##ArrayElements, jenv, jresult, 0);
  if (!arr)
    return NULL;
  for (i=0; i<sz; i++)
    arr[i] = (JNITYPE)result[i];
  JCALL3(Release##JAVATYPE##ArrayElements, jenv, jresult, arr, 0);
  return jresult;
}
%}
%enddef

%{
#if defined(SWIG_NOINCLUDE) || defined(SWIG_NOARRAYS)
%}

#ifdef __cplusplus
JAVA_ARRAYS_DECL(bool, jboolean, Boolean, Bool)       /* bool[] */
#endif

JAVA_ARRAYS_DECL(signed char, jbyte, Byte, Schar)     /* signed char[] */
JAVA_ARRAYS_DECL(unsigned char, jshort, Short, Uchar) /* unsigned char[] */
JAVA_ARRAYS_DECL(short, jshort, Short, Short)         /* short[] */
JAVA_ARRAYS_DECL(unsigned short, jint, Int, Ushort)   /* unsigned short[] */
JAVA_ARRAYS_DECL(int, jint, Int, Int)                 /* int[] */
JAVA_ARRAYS_DECL(unsigned int, jlong, Long, Uint)     /* unsigned int[] */
JAVA_ARRAYS_DECL(long, jint, Int, Long)               /* long[] */
JAVA_ARRAYS_DECL(unsigned long, jlong, Long, Ulong)   /* unsigned long[] */
JAVA_ARRAYS_DECL(jlong, jlong, Long, Longlong)        /* long long[] */
JAVA_ARRAYS_DECL(float, jfloat, Float, Float)         /* float[] */
JAVA_ARRAYS_DECL(double, jdouble, Double, Double)     /* double[] */

%{
#else
%}

#ifdef __cplusplus
/* Bool array element assignment different to other types to keep Visual C++ quiet */
#define JAVA_TYPEMAP_ARRAY_ELEMENT_ASSIGN(CTYPE) (*carr)[i] = ((*jarr)[i] != 0);
JAVA_ARRAYS_IMPL(bool, jboolean, Boolean, Bool)       /* bool[] */
#undef JAVA_TYPEMAP_ARRAY_ELEMENT_ASSIGN
#endif

#define JAVA_TYPEMAP_ARRAY_ELEMENT_ASSIGN(CTYPE) (*carr)[i] = (CTYPE)(*jarr)[i];
JAVA_ARRAYS_IMPL(signed char, jbyte, Byte, Schar)     /* signed char[] */
JAVA_ARRAYS_IMPL(unsigned char, jshort, Short, Uchar) /* unsigned char[] */
JAVA_ARRAYS_IMPL(short, jshort, Short, Short)         /* short[] */
JAVA_ARRAYS_IMPL(unsigned short, jint, Int, Ushort)   /* unsigned short[] */
JAVA_ARRAYS_IMPL(int, jint, Int, Int)                 /* int[] */
JAVA_ARRAYS_IMPL(unsigned int, jlong, Long, Uint)     /* unsigned int[] */
JAVA_ARRAYS_IMPL(long, jint, Int, Long)               /* long[] */
JAVA_ARRAYS_IMPL(unsigned long, jlong, Long, Ulong)   /* unsigned long[] */
JAVA_ARRAYS_IMPL(jlong, jlong, Long, Longlong)        /* long long[] */
JAVA_ARRAYS_IMPL(float, jfloat, Float, Float)         /* float[] */
JAVA_ARRAYS_IMPL(double, jdouble, Double, Double)     /* double[] */

%{
#endif
%}


/* The rest of this file has the array typemaps */

/* Arrays of primitive types use the following macro. The array typemaps use support functions. */
%define JAVA_ARRAYS_TYPEMAPS(CTYPE, JTYPE, JNITYPE, JFUNCNAME, JNIDESC)

%typemap(jni) CTYPE[ANY], CTYPE[]               %{JNITYPE##Array%}
%typemap(jtype) CTYPE[ANY], CTYPE[]             %{JTYPE[]%}
%typemap(jstype) CTYPE[ANY], CTYPE[]            %{JTYPE[]%}

%typemap(in) CTYPE[] (JNITYPE *jarr)
%{  if (!SWIG_JavaArrayIn##JFUNCNAME(jenv, &jarr, (CTYPE **)&$1, $input)) return $null; %}
%typemap(in) CTYPE[ANY] (JNITYPE *jarr)
%{  if ($input && JCALL1(GetArrayLength, jenv, $input) != $1_size) {
    SWIG_JavaThrowException(jenv, SWIG_JavaIndexOutOfBoundsException, "incorrect array size");
    return $null;
  }
  if (!SWIG_JavaArrayIn##JFUNCNAME(jenv, &jarr, (CTYPE **)&$1, $input)) return $null; %}
%typemap(argout) CTYPE[ANY], CTYPE[] 
%{ SWIG_JavaArrayArgout##JFUNCNAME(jenv, jarr$argnum, (CTYPE *)$1, $input); %}
%typemap(out) CTYPE[ANY]
%{$result = SWIG_JavaArrayOut##JFUNCNAME(jenv, (CTYPE *)$1, $1_dim0); %}
%typemap(out) CTYPE[] 
%{$result = SWIG_JavaArrayOut##JFUNCNAME(jenv, (CTYPE *)$1, FillMeInAsSizeCannotBeDeterminedAutomatically); %}
%typemap(freearg) CTYPE[ANY], CTYPE[] 
#ifdef __cplusplus
%{ delete [] $1; %}
#else
%{ free($1); %}
#endif

%typemap(javain) CTYPE[ANY], CTYPE[] "$javainput"
%typemap(javaout) CTYPE[ANY], CTYPE[] {
    return $jnicall;
  }

%typemap(memberin) CTYPE[ANY], CTYPE[];
%typemap(globalin) CTYPE[ANY], CTYPE[];
%enddef

JAVA_ARRAYS_TYPEMAPS(bool, boolean, jboolean, Bool, "[Z")       /* bool[ANY] */
JAVA_ARRAYS_TYPEMAPS(signed char, byte, jbyte, Schar, "[B")     /* signed char[ANY] */
JAVA_ARRAYS_TYPEMAPS(unsigned char, short, jshort, Uchar, "[S") /* unsigned char[ANY] */
JAVA_ARRAYS_TYPEMAPS(short, short, jshort, Short, "[S")         /* short[ANY] */
JAVA_ARRAYS_TYPEMAPS(unsigned short, int, jint, Ushort, "[I")   /* unsigned short[ANY] */
JAVA_ARRAYS_TYPEMAPS(int, int, jint, Int, "[I")                 /* int[ANY] */
JAVA_ARRAYS_TYPEMAPS(unsigned int, long, jlong, Uint, "[J")     /* unsigned int[ANY] */
JAVA_ARRAYS_TYPEMAPS(long, int, jint, Long, "[I")               /* long[ANY] */
JAVA_ARRAYS_TYPEMAPS(unsigned long, long, jlong, Ulong, "[J")   /* unsigned long[ANY] */
JAVA_ARRAYS_TYPEMAPS(long long, long, jlong, Longlong, "[J")    /* long long[ANY] */
JAVA_ARRAYS_TYPEMAPS(float, float, jfloat, Float, "[F")         /* float[ANY] */
JAVA_ARRAYS_TYPEMAPS(double, double, jdouble, Double, "[D")     /* double[ANY] */


%typecheck(SWIG_TYPECHECK_BOOL_ARRAY) /* Java boolean[] */
    bool[ANY], bool[]
    ""

%typecheck(SWIG_TYPECHECK_INT8_ARRAY) /* Java byte[] */
    signed char[ANY], signed char[]
    ""

%typecheck(SWIG_TYPECHECK_INT16_ARRAY) /* Java short[] */
    unsigned char[ANY], unsigned char[],
    short[ANY], short[]
    ""

%typecheck(SWIG_TYPECHECK_INT32_ARRAY) /* Java int[] */
    unsigned short[ANY], unsigned short[],
    int[ANY], int[],
    long[ANY], long[]
    ""

%typecheck(SWIG_TYPECHECK_INT64_ARRAY) /* Java long[] */
    unsigned int[ANY], unsigned int[],
    unsigned long[ANY], unsigned long[],
    long long[ANY], long long[]
    ""

%typecheck(SWIG_TYPECHECK_INT128_ARRAY) /* Java BigInteger[] */
    unsigned long long[ANY], unsigned long long[]
    ""

%typecheck(SWIG_TYPECHECK_FLOAT_ARRAY) /* Java float[] */
    float[ANY], float[]
    ""

%typecheck(SWIG_TYPECHECK_DOUBLE_ARRAY) /* Java double[] */
    double[ANY], double[]
    ""


/* Arrays of proxy classes. The typemaps in this macro make it possible to treat an array of 
 * class/struct/unions as an array of Java classes. 
 * Use the following macro to use these typemaps for an array of class/struct/unions called name:
 * JAVA_ARRAYSOFCLASSES(name) 
 * Note that multiple copies of the class/struct is made when using the array as a parameter input. */
%define JAVA_ARRAYSOFCLASSES(ARRAYSOFCLASSES)

%typemap(jni) ARRAYSOFCLASSES[ANY], ARRAYSOFCLASSES[] "jlongArray"
%typemap(jtype) ARRAYSOFCLASSES[ANY], ARRAYSOFCLASSES[] "long[]"
%typemap(jstype) ARRAYSOFCLASSES[ANY], ARRAYSOFCLASSES[] "$javaclassname[]"

%typemap(javain) ARRAYSOFCLASSES[ANY], ARRAYSOFCLASSES[] "$javaclassname.cArrayUnwrap($javainput)"
%typemap(javaout) ARRAYSOFCLASSES[ANY], ARRAYSOFCLASSES[] {
    return $javaclassname.cArrayWrap($jnicall, $owner);
  }

%typemap(in) ARRAYSOFCLASSES[] (jlong *jarr, jsize sz)
{
  int i;
  if (!$input) {
    SWIG_JavaThrowException(jenv, SWIG_JavaNullPointerException, "null array");
    return $null;
  }
  sz = JCALL1(GetArrayLength, jenv, $input);
  jarr = JCALL2(GetLongArrayElements, jenv, $input, 0);
  if (!jarr) {
    return $null;
  }
#ifdef __cplusplus
  $1 = new $*1_ltype[sz];
#else
  $1 = ($1_ltype) malloc(sz * sizeof($*1_ltype));
#endif
  if (!$1) {
    SWIG_JavaThrowException(jenv, SWIG_JavaOutOfMemoryError, "array memory allocation failed");
    return $null;
  }
  for (i=0; i<sz; i++) {
    $1[i] = **($&1_ltype)&jarr[i];
  }
}

%typemap(in) ARRAYSOFCLASSES[ANY] (jlong *jarr, jsize sz)
{
  int i;
  if (!$input) {
    SWIG_JavaThrowException(jenv, SWIG_JavaNullPointerException, "null array");
    return $null;
  }
  sz = JCALL1(GetArrayLength, jenv, $input);
  if (sz != $1_size) {
    SWIG_JavaThrowException(jenv, SWIG_JavaIndexOutOfBoundsException, "incorrect array size");
    return $null;
  }
  jarr = JCALL2(GetLongArrayElements, jenv, $input, 0);
  if (!jarr) {
    return $null;
  }
#ifdef __cplusplus
  $1 = new $*1_ltype[sz];
#else
  $1 = ($1_ltype) malloc(sz * sizeof($*1_ltype));
#endif
  if (!$1) {
    SWIG_JavaThrowException(jenv, SWIG_JavaOutOfMemoryError, "array memory allocation failed");
    return $null;
  }
  for (i=0; i<sz; i++) {
    $1[i] = **($&1_ltype)&jarr[i];
  }
}

%typemap(argout) ARRAYSOFCLASSES[ANY], ARRAYSOFCLASSES[]
{
  int i;
  for (i=0; i<sz$argnum; i++) {
    **($&1_ltype)&jarr$argnum[i] = $1[i];
  }
  JCALL3(ReleaseLongArrayElements, jenv, $input, jarr$argnum, 0);
}

%typemap(out) ARRAYSOFCLASSES[ANY]
{
  jlong *arr;
  int i;
  $result = JCALL1(NewLongArray, jenv, $1_dim0);
  if (!$result) {
    return $null;
  }
  arr = JCALL2(GetLongArrayElements, jenv, $result, 0);
  if (!arr) {
    return $null;
  }
  for (i=0; i<$1_dim0; i++) {
    arr[i] = 0;
    *($&1_ltype)&arr[i] = &$1[i];
  }
  JCALL3(ReleaseLongArrayElements, jenv, $result, arr, 0);
}

%typemap(freearg) ARRAYSOFCLASSES[ANY], ARRAYSOFCLASSES[]
#ifdef __cplusplus
%{ delete [] $1; %}
#else
%{ free($1); %}
#endif

/* Add some code to the proxy class of the array type for converting between type used in 
 * JNI class (long[]) and type used in proxy class ( ARRAYSOFCLASSES[] ) */
%extend ARRAYSOFCLASSES {
%proxycode %{
  protected static long[] cArrayUnwrap($javaclassname[] arrayWrapper) {
      long[] cArray = new long[arrayWrapper.length];
      for (int i=0; i<arrayWrapper.length; i++)
        cArray[i] = $javaclassname.getCPtr(arrayWrapper[i]);
      return cArray;
  }

  protected static $javaclassname[] cArrayWrap(long[] cArray, boolean cMemoryOwn) {
    $javaclassname[] arrayWrapper = new $javaclassname[cArray.length];
    for (int i=0; i<cArray.length; i++)
      arrayWrapper[i] = new $javaclassname(cArray[i], cMemoryOwn);
    return arrayWrapper;
  }
%}
}

%enddef /* JAVA_ARRAYSOFCLASSES */


/* Arrays of enums. 
 * Use the following to use these typemaps for an array of enums called name:
 * %apply ARRAYSOFENUMS[ANY] { name[ANY] }; */
%typemap(jni) ARRAYSOFENUMS[ANY], ARRAYSOFENUMS[] "jintArray"
%typemap(jtype) ARRAYSOFENUMS[ANY], ARRAYSOFENUMS[] "int[]"
%typemap(jstype) ARRAYSOFENUMS[ANY], ARRAYSOFENUMS[] "int[]"

%typemap(javain) ARRAYSOFENUMS[ANY], ARRAYSOFENUMS[] "$javainput"
%typemap(javaout) ARRAYSOFENUMS[ANY], ARRAYSOFENUMS[] {
    return $jnicall;
  }

%typemap(in) ARRAYSOFENUMS[] (jint *jarr)
%{  if (!SWIG_JavaArrayInInt(jenv, &jarr, (int **)&$1, $input)) return $null; %}
%typemap(in) ARRAYSOFENUMS[ANY] (jint *jarr) {
  if ($input && JCALL1(GetArrayLength, jenv, $input) != $1_size) {
    SWIG_JavaThrowException(jenv, SWIG_JavaIndexOutOfBoundsException, "incorrect array size");
    return $null;
  }
  if (!SWIG_JavaArrayInInt(jenv, &jarr, (int **)&$1, $input)) return $null;
}
%typemap(argout) ARRAYSOFENUMS[ANY], ARRAYSOFENUMS[] 
%{ SWIG_JavaArrayArgoutInt(jenv, jarr$argnum, (int *)$1, $input); %}
%typemap(out) ARRAYSOFENUMS[ANY] 
%{$result = SWIG_JavaArrayOutInt(jenv, (int *)$1, $1_dim0); %}
%typemap(freearg) ARRAYSOFENUMS[ANY], ARRAYSOFENUMS[] 
#ifdef __cplusplus
%{ delete [] $1; %}
#else
%{ free($1); %}
#endif

