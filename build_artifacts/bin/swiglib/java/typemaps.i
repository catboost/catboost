/* -----------------------------------------------------------------------------
 * typemaps.i
 *
 * Pointer and reference handling typemap library
 *
 * These mappings provide support for input/output arguments and common
 * uses for C/C++ pointers and C++ references.
 * ----------------------------------------------------------------------------- */

/*
INPUT typemaps
--------------

These typemaps remap a C pointer or C++ reference to be an "INPUT" value which is
passed by value instead of reference.

The following typemaps can be applied to turn a pointer or reference into a simple
input value.  That is, instead of passing a pointer or reference to an object,
you would use a real value instead.

        bool               *INPUT, bool               &INPUT
        signed char        *INPUT, signed char        &INPUT
        unsigned char      *INPUT, unsigned char      &INPUT
        short              *INPUT, short              &INPUT
        unsigned short     *INPUT, unsigned short     &INPUT
        int                *INPUT, int                &INPUT
        unsigned int       *INPUT, unsigned int       &INPUT
        long               *INPUT, long               &INPUT
        unsigned long      *INPUT, unsigned long      &INPUT
        long long          *INPUT, long long          &INPUT
        unsigned long long *INPUT, unsigned long long &INPUT
        float              *INPUT, float              &INPUT
        double             *INPUT, double             &INPUT
         
To use these, suppose you had a C function like this :

        double fadd(double *a, double *b) {
               return *a+*b;
        }

You could wrap it with SWIG as follows :
        
        %include <typemaps.i>
        double fadd(double *INPUT, double *INPUT);

or you can use the %apply directive :

        %include <typemaps.i>
        %apply double *INPUT { double *a, double *b };
        double fadd(double *a, double *b);

In Java you could then use it like this:
        double answer = modulename.fadd(10.0, 20.0);

There are no char *INPUT typemaps, however you can apply the signed char * typemaps instead:
        %include <typemaps.i>
        %apply signed char *INPUT {char *input};
        void f(char *input);
*/

%define INPUT_TYPEMAP(TYPE, JNITYPE, JTYPE, JNIDESC)
%typemap(jni) TYPE *INPUT, TYPE &INPUT "JNITYPE"
%typemap(jtype) TYPE *INPUT, TYPE &INPUT "JTYPE"
%typemap(jstype) TYPE *INPUT, TYPE &INPUT "JTYPE"
%typemap(javain) TYPE *INPUT, TYPE &INPUT "$javainput"

%typemap(in) TYPE *INPUT, TYPE &INPUT
%{ $1 = ($1_ltype)&$input; %}

%typemap(freearg) TYPE *INPUT, TYPE &INPUT ""

%typemap(typecheck) TYPE *INPUT = TYPE;
%typemap(typecheck) TYPE &INPUT = TYPE;
%enddef

INPUT_TYPEMAP(bool, jboolean, boolean, "Z");
INPUT_TYPEMAP(signed char, jbyte, byte, "B");
INPUT_TYPEMAP(unsigned char, jshort, short, "S");
INPUT_TYPEMAP(short, jshort, short, "S");
INPUT_TYPEMAP(unsigned short, jint, int, "I");
INPUT_TYPEMAP(int, jint, int, "I");
INPUT_TYPEMAP(unsigned int, jlong, long, "J");
INPUT_TYPEMAP(long, jint, int, "I");
INPUT_TYPEMAP(unsigned long, jlong, long, "J");
INPUT_TYPEMAP(long long, jlong, long, "J");
INPUT_TYPEMAP(unsigned long long, jobject, java.math.BigInteger, "Ljava/math/BigInteger;");
INPUT_TYPEMAP(float, jfloat, float, "F");
INPUT_TYPEMAP(double, jdouble, double, "D");

#undef INPUT_TYPEMAP

/* Convert from BigInteger using the toByteArray member function */
/* Overrides the typemap in the INPUT_TYPEMAP macro */
%typemap(in) unsigned long long *INPUT($*1_ltype temp), unsigned long long &INPUT($*1_ltype temp) {
  jclass clazz;
  jmethodID mid;
  jbyteArray ba;
  jbyte* bae;
  jsize sz;
  int i;

  if (!$input) {
    SWIG_JavaThrowException(jenv, SWIG_JavaNullPointerException, "BigInteger null");
    return $null;
  }
  clazz = JCALL1(GetObjectClass, jenv, $input);
  mid = JCALL3(GetMethodID, jenv, clazz, "toByteArray", "()[B");
  ba = (jbyteArray)JCALL2(CallObjectMethod, jenv, $input, mid);
  bae = JCALL2(GetByteArrayElements, jenv, ba, 0);
  sz = JCALL1(GetArrayLength, jenv, ba);
  temp = 0;
  if (sz > 0) {
    temp = ($*1_ltype)(signed char)bae[0];
    for(i=1; i<sz; i++) {
      temp = (temp << 8) | ($*1_ltype)(unsigned char)bae[i];
    }
  }
  JCALL3(ReleaseByteArrayElements, jenv, ba, bae, 0);
  $1 = &temp;
}

// OUTPUT typemaps.   These typemaps are used for parameters that
// are output only.   An array replaces the c pointer or reference parameter. 
// The output value is returned in this array passed in. 

/*
OUTPUT typemaps
---------------

The following typemaps can be applied to turn a pointer or reference into an "output"
value.  When calling a function, no input value would be given for
a parameter, but an output value would be returned.  This works by a 
Java array being passed as a parameter where a c pointer or reference is required. 
As with any Java function, the array is passed by reference so that 
any modifications to the array will be picked up in the calling function.
Note that the array passed in MUST have at least one element, but as the 
c function does not require any input, the value can be set to anything.

        bool               *OUTPUT, bool               &OUTPUT
        signed char        *OUTPUT, signed char        &OUTPUT
        unsigned char      *OUTPUT, unsigned char      &OUTPUT
        short              *OUTPUT, short              &OUTPUT
        unsigned short     *OUTPUT, unsigned short     &OUTPUT
        int                *OUTPUT, int                &OUTPUT
        unsigned int       *OUTPUT, unsigned int       &OUTPUT
        long               *OUTPUT, long               &OUTPUT
        unsigned long      *OUTPUT, unsigned long      &OUTPUT
        long long          *OUTPUT, long long          &OUTPUT
        unsigned long long *OUTPUT, unsigned long long &OUTPUT
        float              *OUTPUT, float              &OUTPUT
        double             *OUTPUT, double             &OUTPUT
         
For example, suppose you were trying to wrap the modf() function in the
C math library which splits x into integral and fractional parts (and
returns the integer part in one of its parameters):

        double modf(double x, double *ip);

You could wrap it with SWIG as follows :

        %include <typemaps.i>
        double modf(double x, double *OUTPUT);

or you can use the %apply directive :

        %include <typemaps.i>
        %apply double *OUTPUT { double *ip };
        double modf(double x, double *ip);

The Java output of the function would be the function return value and the 
value in the single element array. In Java you would use it like this:

    double[] ptr = {0.0};
    double fraction = modulename.modf(5.0,ptr);

There are no char *OUTPUT typemaps, however you can apply the signed char * typemaps instead:
        %include <typemaps.i>
        %apply signed char *OUTPUT {char *output};
        void f(char *output);
*/

/* Java BigInteger[] */
%typecheck(SWIG_TYPECHECK_INT128_ARRAY) SWIGBIGINTEGERARRAY ""

%define OUTPUT_TYPEMAP(TYPE, JNITYPE, JTYPE, JAVATYPE, JNIDESC, TYPECHECKTYPE)
%typemap(jni) TYPE *OUTPUT, TYPE &OUTPUT %{JNITYPE##Array%}
%typemap(jtype) TYPE *OUTPUT, TYPE &OUTPUT "JTYPE[]"
%typemap(jstype) TYPE *OUTPUT, TYPE &OUTPUT "JTYPE[]"
%typemap(javain) TYPE *OUTPUT, TYPE &OUTPUT "$javainput"
%typemap(javadirectorin) TYPE *OUTPUT, TYPE &OUTPUT "$jniinput"
%typemap(javadirectorout) TYPE *OUTPUT, TYPE &OUTPUT "$javacall"

%typemap(in) TYPE *OUTPUT($*1_ltype temp), TYPE &OUTPUT($*1_ltype temp)
{
  if (!$input) {
    SWIG_JavaThrowException(jenv, SWIG_JavaNullPointerException, "array null");
    return $null;
  }
  if (JCALL1(GetArrayLength, jenv, $input) == 0) {
    SWIG_JavaThrowException(jenv, SWIG_JavaIndexOutOfBoundsException, "Array must contain at least 1 element");
    return $null;
  }
  temp = ($*1_ltype)0;
  $1 = &temp; 
}

%typemap(freearg) TYPE *OUTPUT, TYPE &OUTPUT ""

%typemap(argout) TYPE *OUTPUT, TYPE &OUTPUT
{
  JNITYPE jvalue = (JNITYPE)temp$argnum;
  JCALL4(Set##JAVATYPE##ArrayRegion, jenv, $input, 0, 1, &jvalue);
}

%typemap(directorin,descriptor=JNIDESC) TYPE &OUTPUT %{
  $input = JCALL1(New##JAVATYPE##Array, jenv, 1);
  if (!$input) return $null;
  Swig::LocalRefGuard $1_refguard(jenv, $input); %}

%typemap(directorin,descriptor=JNIDESC) TYPE *OUTPUT %{
  if ($1) {
    $input = JCALL1(New##JAVATYPE##Array, jenv, 1);
    if (!$input) return $null;
  }
  Swig::LocalRefGuard $1_refguard(jenv, $input); %}

%typemap(directorargout, noblock=1) TYPE &OUTPUT
{
  JNITYPE $1_jvalue;
  JCALL4(Get##JAVATYPE##ArrayRegion, jenv, $input, 0, 1, &$1_jvalue);
  $result = ($*1_ltype)$1_jvalue;
}

%typemap(directorargout, noblock=1) TYPE *OUTPUT
{
  if ($result) {
    JNITYPE $1_jvalue;
    JCALL4(Get##JAVATYPE##ArrayRegion, jenv, $input, 0, 1, &$1_jvalue);
    *$result = ($*1_ltype)$1_jvalue;
  }
}

%typemap(typecheck) TYPE *OUTPUT = TYPECHECKTYPE;
%typemap(typecheck) TYPE &OUTPUT = TYPECHECKTYPE;
%enddef

OUTPUT_TYPEMAP(bool, jboolean, boolean, Boolean, "[Z", jbooleanArray);
OUTPUT_TYPEMAP(signed char, jbyte, byte, Byte, "[B", jbyteArray);
OUTPUT_TYPEMAP(unsigned char, jshort, short, Short, "[S", jshortArray);
OUTPUT_TYPEMAP(short, jshort, short, Short, "[S", jshortArray);
OUTPUT_TYPEMAP(unsigned short, jint, int, Int, "[I", jintArray);
OUTPUT_TYPEMAP(int, jint, int, Int, "[I", jintArray);
OUTPUT_TYPEMAP(unsigned int, jlong, long, Long, "[J", jlongArray);
OUTPUT_TYPEMAP(long, jint, int, Int, "[I", jintArray);
OUTPUT_TYPEMAP(unsigned long, jlong, long, Long, "[J", jlongArray);
OUTPUT_TYPEMAP(long long, jlong, long, Long, "[J", jlongArray);
OUTPUT_TYPEMAP(unsigned long long, jobject, java.math.BigInteger, Object, "[Ljava/math/BigInteger;", jobjectArray);
OUTPUT_TYPEMAP(float, jfloat, float, Float, "[F", jfloatArray);
OUTPUT_TYPEMAP(double, jdouble, double, Double, "[D", jdoubleArray);

#undef OUTPUT_TYPEMAP

%typemap(in) bool *OUTPUT($*1_ltype temp), bool &OUTPUT($*1_ltype temp)
{
  if (!$input) {
    SWIG_JavaThrowException(jenv, SWIG_JavaNullPointerException, "array null");
    return $null;
  }
  if (JCALL1(GetArrayLength, jenv, $input) == 0) {
    SWIG_JavaThrowException(jenv, SWIG_JavaIndexOutOfBoundsException, "Array must contain at least 1 element");
    return $null;
  }
  temp = false;
  $1 = &temp; 
}

%typemap(directorargout, noblock=1) bool &OUTPUT
{
  jboolean $1_jvalue;
  JCALL4(GetBooleanArrayRegion, jenv, $input, 0, 1, &$1_jvalue);
  $result = $1_jvalue ? true : false;
}

%typemap(directorargout, noblock=1) bool *OUTPUT
{
  if ($result) {
    jboolean $1_jvalue;
    JCALL4(GetBooleanArrayRegion, jenv, $input, 0, 1, &$1_jvalue);
    *$result = $1_jvalue ? true : false;
  }
}


/* Convert to BigInteger - byte array holds number in 2's complement big endian format */
/* Use first element in BigInteger array for output */
/* Overrides the typemap in the OUTPUT_TYPEMAP macro */
%typemap(argout) unsigned long long *OUTPUT, unsigned long long &OUTPUT { 
  jbyteArray ba = JCALL1(NewByteArray, jenv, 9);
  jbyte* bae = JCALL2(GetByteArrayElements, jenv, ba, 0);
  jclass clazz = JCALL1(FindClass, jenv, "java/math/BigInteger");
  jmethodID mid = JCALL3(GetMethodID, jenv, clazz, "<init>", "([B)V");
  jobject bigint;
  int i;

  bae[0] = 0;
  for(i=1; i<9; i++ ) {
    bae[i] = (jbyte)(temp$argnum>>8*(8-i));
  }

  JCALL3(ReleaseByteArrayElements, jenv, ba, bae, 0);
  bigint = JCALL3(NewObject, jenv, clazz, mid, ba);
  JCALL1(DeleteLocalRef, jenv, ba);
  JCALL3(SetObjectArrayElement, jenv, $input, 0, bigint);
}

/*
INOUT typemaps
--------------

Mappings for a parameter that is both an input and an output parameter

The following typemaps can be applied to make a function parameter both
an input and output value.  This combines the behavior of both the
"INPUT" and "OUTPUT" typemaps described earlier.  Output values are
returned as an element in a Java array.

        bool               *INOUT, bool               &INOUT
        signed char        *INOUT, signed char        &INOUT
        unsigned char      *INOUT, unsigned char      &INOUT
        short              *INOUT, short              &INOUT
        unsigned short     *INOUT, unsigned short     &INOUT
        int                *INOUT, int                &INOUT
        unsigned int       *INOUT, unsigned int       &INOUT
        long               *INOUT, long               &INOUT
        unsigned long      *INOUT, unsigned long      &INOUT
        long long          *INOUT, long long          &INOUT
        unsigned long long *INOUT, unsigned long long &INOUT
        float              *INOUT, float              &INOUT
        double             *INOUT, double             &INOUT
         
For example, suppose you were trying to wrap the following function :

        void neg(double *x) {
             *x = -(*x);
        }

You could wrap it with SWIG as follows :

        %include <typemaps.i>
        void neg(double *INOUT);

or you can use the %apply directive :

        %include <typemaps.i>
        %apply double *INOUT { double *x };
        void neg(double *x);

This works similarly to C in that the mapping directly modifies the
input value - the input must be an array with a minimum of one element. 
The element in the array is the input and the output is the element in 
the array.

       double x[] = {5.0};
       neg(x);

The implementation of the OUTPUT and INOUT typemaps is different to other 
languages in that other languages will return the output value as part 
of the function return value. This difference is due to Java being a typed language.

There are no char *INOUT typemaps, however you can apply the signed char * typemaps instead:
        %include <typemaps.i>
        %apply signed char *INOUT {char *inout};
        void f(char *inout);
*/

%define INOUT_TYPEMAP(TYPE, JNITYPE, JTYPE, JAVATYPE, JNIDESC, TYPECHECKTYPE)
%typemap(jni) TYPE *INOUT, TYPE &INOUT %{JNITYPE##Array%}
%typemap(jtype) TYPE *INOUT, TYPE &INOUT "JTYPE[]"
%typemap(jstype) TYPE *INOUT, TYPE &INOUT "JTYPE[]"
%typemap(javain) TYPE *INOUT, TYPE &INOUT "$javainput"
%typemap(javadirectorin) TYPE *INOUT, TYPE &INOUT "$jniinput"
%typemap(javadirectorout) TYPE *INOUT, TYPE &INOUT "$javacall"

%typemap(in) TYPE *INOUT, TYPE &INOUT {
  if (!$input) {
    SWIG_JavaThrowException(jenv, SWIG_JavaNullPointerException, "array null");
    return $null;
  }
  if (JCALL1(GetArrayLength, jenv, $input) == 0) {
    SWIG_JavaThrowException(jenv, SWIG_JavaIndexOutOfBoundsException, "Array must contain at least 1 element");
    return $null;
  }
  $1 = ($1_ltype) JCALL2(Get##JAVATYPE##ArrayElements, jenv, $input, 0); 
}

%typemap(freearg) TYPE *INOUT, TYPE &INOUT ""

%typemap(argout) TYPE *INOUT, TYPE &INOUT
{ JCALL3(Release##JAVATYPE##ArrayElements, jenv, $input, (JNITYPE *)$1, 0); }

%typemap(directorin,descriptor=JNIDESC) TYPE &INOUT %{
  $input = JCALL1(New##JAVATYPE##Array, jenv, 1);
  if (!$input) return $null;
  JNITYPE $1_jvalue = (JNITYPE)$1;
  JCALL4(Set##JAVATYPE##ArrayRegion, jenv, $input, 0, 1, &$1_jvalue);
  Swig::LocalRefGuard $1_refguard(jenv, $input); %}

%typemap(directorin,descriptor=JNIDESC) TYPE *INOUT %{
  if ($1) {
    $input = JCALL1(New##JAVATYPE##Array, jenv, 1);
    if (!$input) return $null;
    JNITYPE $1_jvalue = (JNITYPE)*$1;
    JCALL4(Set##JAVATYPE##ArrayRegion, jenv, $input, 0, 1, &$1_jvalue);
  }
  Swig::LocalRefGuard $1_refguard(jenv, $input); %}

%typemap(directorargout, noblock=1) TYPE &INOUT
{
  JCALL4(Get##JAVATYPE##ArrayRegion, jenv, $input, 0, 1, &$1_jvalue);
  $result = ($*1_ltype)$1_jvalue;
}

%typemap(directorargout, noblock=1) TYPE *INOUT
{
  if ($result) {
    JNITYPE $1_jvalue;
    JCALL4(Get##JAVATYPE##ArrayRegion, jenv, $input, 0, 1, &$1_jvalue);
    *$result = ($*1_ltype)$1_jvalue;
  }
}

%typemap(typecheck) TYPE *INOUT = TYPECHECKTYPE;
%typemap(typecheck) TYPE &INOUT = TYPECHECKTYPE;
%enddef

INOUT_TYPEMAP(bool, jboolean, boolean, Boolean, "[Z", jbooleanArray);
INOUT_TYPEMAP(signed char, jbyte, byte, Byte, "[B", jbyteArray);
INOUT_TYPEMAP(unsigned char, jshort, short, Short, "[S", jshortArray);
INOUT_TYPEMAP(short, jshort, short, Short, "[S", jshortArray);
INOUT_TYPEMAP(unsigned short, jint, int, Int, "[I", jintArray);
INOUT_TYPEMAP(int, jint, int, Int, "[I", jintArray);
INOUT_TYPEMAP(unsigned int, jlong, long, Long, "[J", jlongArray);
INOUT_TYPEMAP(long, jint, int, Int, "[I", jintArray);
INOUT_TYPEMAP(unsigned long, jlong, long, Long, "[J", jlongArray);
INOUT_TYPEMAP(long long, jlong, long, Long, "[J", jlongArray);
INOUT_TYPEMAP(unsigned long long, jobject, java.math.BigInteger, Object, "[java/math/BigInteger;", jobjectArray);
INOUT_TYPEMAP(float, jfloat, float, Float, "[F", jfloatArray);
INOUT_TYPEMAP(double, jdouble, double, Double, "[D", jdoubleArray);

#undef INOUT_TYPEMAP

/* Override typemaps in the INOUT_TYPEMAP macro for booleans to fix casts
   as a jboolean isn't always the same size as a bool */
%typemap(in) bool *INOUT (bool btemp, jboolean *jbtemp), bool &INOUT (bool btemp, jboolean *jbtemp) {
  if (!$input) {
    SWIG_JavaThrowException(jenv, SWIG_JavaNullPointerException, "array null");
    return $null;
  }
  if (JCALL1(GetArrayLength, jenv, $input) == 0) {
    SWIG_JavaThrowException(jenv, SWIG_JavaIndexOutOfBoundsException, "Array must contain at least 1 element");
    return $null;
  }
  jbtemp = JCALL2(GetBooleanArrayElements, jenv, $input, 0);
  btemp = (*jbtemp) ? true : false;
  $1 = &btemp;
}

%typemap(argout) bool *INOUT, bool &INOUT {
  *jbtemp$argnum = btemp$argnum ? (jboolean)1 : (jboolean)0;
  JCALL3(ReleaseBooleanArrayElements, jenv, $input , (jboolean *)jbtemp$argnum, 0);
}

%typemap(directorargout, noblock=1) bool &INOUT
{
  JCALL4(GetBooleanArrayRegion, jenv, $input, 0, 1, &$1_jvalue);
  $result = $1_jvalue ? true : false;
}

%typemap(directorargout, noblock=1) bool *INOUT
{
  if ($result) {
    jboolean $1_jvalue;
    JCALL4(GetBooleanArrayRegion, jenv, $input, 0, 1, &$1_jvalue);
    *$result = $1_jvalue ? true : false;
  }
}


/* Override the typemap in the INOUT_TYPEMAP macro for unsigned long long */
%typemap(in) unsigned long long *INOUT ($*1_ltype temp), unsigned long long &INOUT ($*1_ltype temp) { 
  jobject bigint;
  jclass clazz;
  jmethodID mid;
  jbyteArray ba;
  jbyte* bae;
  jsize sz;
  int i;

  if (!$input) {
    SWIG_JavaThrowException(jenv, SWIG_JavaNullPointerException, "array null");
    return $null;
  }
  if (JCALL1(GetArrayLength, jenv, $input) == 0) {
    SWIG_JavaThrowException(jenv, SWIG_JavaIndexOutOfBoundsException, "Array must contain at least 1 element");
    return $null;
  }
  bigint = JCALL2(GetObjectArrayElement, jenv, $input, 0);
  if (!bigint) {
    SWIG_JavaThrowException(jenv, SWIG_JavaNullPointerException, "array element null");
    return $null;
  }
  clazz = JCALL1(GetObjectClass, jenv, bigint);
  mid = JCALL3(GetMethodID, jenv, clazz, "toByteArray", "()[B");
  ba = (jbyteArray)JCALL2(CallObjectMethod, jenv, bigint, mid);
  bae = JCALL2(GetByteArrayElements, jenv, ba, 0);
  sz = JCALL1(GetArrayLength, jenv, ba);
  temp = 0;
  if (sz > 0) {
    temp = ($*1_ltype)(signed char)bae[0];
    for(i=1; i<sz; i++) {
      temp = (temp << 8) | ($*1_ltype)(unsigned char)bae[i];
    }
  }
  JCALL3(ReleaseByteArrayElements, jenv, ba, bae, 0);
  $1 = &temp;
}

%typemap(argout) unsigned long long *INOUT = unsigned long long *OUTPUT;
%typemap(argout) unsigned long long &INOUT = unsigned long long &OUTPUT;
