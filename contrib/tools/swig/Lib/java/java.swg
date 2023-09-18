/* -----------------------------------------------------------------------------
 * java.swg
 *
 * Java typemaps
 * ----------------------------------------------------------------------------- */

%include <javahead.swg>

/* The jni, jtype and jstype typemaps work together and so there should be one of each. 
 * The jni typemap contains the JNI type used in the JNI (C/C++) code. 
 * The jtype typemap contains the Java type used in the JNI intermediary class. 
 * The jstype typemap contains the Java type used in the Java proxy classes, type wrapper classes and module class. */

/* Fragments */
%fragment("SWIG_PackData", "header") {
/* Pack binary data into a string */
SWIGINTERN char * SWIG_PackData(char *c, void *ptr, size_t sz) {
  static const char hex[17] = "0123456789abcdef";
  const unsigned char *u = (unsigned char *) ptr;
  const unsigned char *eu =  u + sz;
  for (; u != eu; ++u) {
    unsigned char uu = *u;
    *(c++) = hex[(uu & 0xf0) >> 4];
    *(c++) = hex[uu & 0xf];
  }
  return c;
}
}

%fragment("SWIG_UnPackData", "header") {
/* Unpack binary data from a string */
SWIGINTERN const char * SWIG_UnpackData(const char *c, void *ptr, size_t sz) {
  unsigned char *u = (unsigned char *) ptr;
  const unsigned char *eu = u + sz;
  for (; u != eu; ++u) {
    char d = *(c++);
    unsigned char uu;
    if ((d >= '0') && (d <= '9'))
      uu = ((d - '0') << 4);
    else if ((d >= 'a') && (d <= 'f'))
      uu = ((d - ('a'-10)) << 4);
    else 
      return (char *) 0;
    d = *(c++);
    if ((d >= '0') && (d <= '9'))
      uu |= (d - '0');
    else if ((d >= 'a') && (d <= 'f'))
      uu |= (d - ('a'-10));
    else 
      return (char *) 0;
    *u = uu;
  }
  return c;
}
}

%fragment("SWIG_JavaIntFromSize_t", "header") {
/* Check for overflow converting to Java int (always signed 32-bit) from (unsigned variable-bit) size_t */
SWIGINTERN jint SWIG_JavaIntFromSize_t(size_t size) {
  static const jint JINT_MAX = 0x7FFFFFFF;
  return (size > (size_t)JINT_MAX) ? -1 : (jint)size;
}
}

/* Primitive types */
%typemap(jni) bool,               const bool &               "jboolean"
%typemap(jni) char,               const char &               "jchar"
%typemap(jni) signed char,        const signed char &        "jbyte"
%typemap(jni) unsigned char,      const unsigned char &      "jshort"
%typemap(jni) short,              const short &              "jshort"
%typemap(jni) unsigned short,     const unsigned short &     "jint"
%typemap(jni) int,                const int &                "jint"
%typemap(jni) unsigned int,       const unsigned int &       "jlong"
%typemap(jni) long,               const long &               "jint"
%typemap(jni) unsigned long,      const unsigned long &      "jlong"
%typemap(jni) long long,          const long long &          "jlong"
%typemap(jni) unsigned long long, const unsigned long long & "jobject"
%typemap(jni) float,              const float &              "jfloat"
%typemap(jni) double,             const double &             "jdouble"
%typemap(jni) void                                           "void"

%typemap(jtype) bool,               const bool &               "boolean"
%typemap(jtype) char,               const char &               "char"
%typemap(jtype) signed char,        const signed char &        "byte"
%typemap(jtype) unsigned char,      const unsigned char &      "short"
%typemap(jtype) short,              const short &              "short"
%typemap(jtype) unsigned short,     const unsigned short &     "int"
%typemap(jtype) int,                const int &                "int"
%typemap(jtype) unsigned int,       const unsigned int &       "long"
%typemap(jtype) long,               const long &               "int"
%typemap(jtype) unsigned long,      const unsigned long &      "long"
%typemap(jtype) long long,          const long long &          "long"
%typemap(jtype) unsigned long long, const unsigned long long & "java.math.BigInteger"
%typemap(jtype) float,              const float &              "float"
%typemap(jtype) double,             const double &             "double"
%typemap(jtype) void                                           "void"

%typemap(jstype) bool,               const bool &               "boolean"
%typemap(jstype) char,               const char &               "char"
%typemap(jstype) signed char,        const signed char &        "byte"
%typemap(jstype) unsigned char,      const unsigned char &      "short"
%typemap(jstype) short,              const short &              "short"
%typemap(jstype) unsigned short,     const unsigned short &     "int"
%typemap(jstype) int,                const int &                "int"
%typemap(jstype) unsigned int,       const unsigned int &       "long"
%typemap(jstype) long,               const long &               "int"
%typemap(jstype) unsigned long,      const unsigned long &      "long"
%typemap(jstype) long long,          const long long &          "long"
%typemap(jstype) unsigned long long, const unsigned long long & "java.math.BigInteger"
%typemap(jstype) float,              const float &              "float"
%typemap(jstype) double,             const double &             "double"
%typemap(jstype) void                                           "void"

%typemap(jboxtype) bool,               const bool &               "Boolean"
%typemap(jboxtype) char,               const char &               "Character"
%typemap(jboxtype) signed char,        const signed char &        "Byte"
%typemap(jboxtype) unsigned char,      const unsigned char &      "Short"
%typemap(jboxtype) short,              const short &              "Short"
%typemap(jboxtype) unsigned short,     const unsigned short &     "Integer"
%typemap(jboxtype) int,                const int &                "Integer"
%typemap(jboxtype) unsigned int,       const unsigned int &       "Long"
%typemap(jboxtype) long,               const long &               "Integer"
%typemap(jboxtype) unsigned long,      const unsigned long &      "Long"
%typemap(jboxtype) long long,          const long long &          "Long"
%typemap(jboxtype) unsigned long long, const unsigned long long & "java.math.BigInteger"
%typemap(jboxtype) float,              const float &              "Float"
%typemap(jboxtype) double,             const double &             "Double"

%typemap(jni) char *, char *&, char[ANY], char[]               "jstring"
%typemap(jtype) char *, char *&, char[ANY], char[]               "String"
%typemap(jstype) char *, char *&, char[ANY], char[]               "String"

/* JNI types */
%typemap(jni) jboolean      "jboolean"
%typemap(jni) jchar         "jchar"
%typemap(jni) jbyte         "jbyte"
%typemap(jni) jshort        "jshort"
%typemap(jni) jint          "jint"
%typemap(jni) jlong         "jlong"
%typemap(jni) jfloat        "jfloat"
%typemap(jni) jdouble       "jdouble"
%typemap(jni) jstring       "jstring"
%typemap(jni) jobject       "jobject"
%typemap(jni) jbooleanArray "jbooleanArray"
%typemap(jni) jcharArray    "jcharArray"
%typemap(jni) jbyteArray    "jbyteArray"
%typemap(jni) jshortArray   "jshortArray"
%typemap(jni) jintArray     "jintArray"
%typemap(jni) jlongArray    "jlongArray"
%typemap(jni) jfloatArray   "jfloatArray"
%typemap(jni) jdoubleArray  "jdoubleArray"
%typemap(jni) jobjectArray  "jobjectArray"

%typemap(jtype) jboolean      "boolean"
%typemap(jtype) jchar         "char"
%typemap(jtype) jbyte         "byte"
%typemap(jtype) jshort        "short"
%typemap(jtype) jint          "int"
%typemap(jtype) jlong         "long"
%typemap(jtype) jfloat        "float"
%typemap(jtype) jdouble       "double"
%typemap(jtype) jstring       "String"
%typemap(jtype) jobject       "java.lang.Object"
%typemap(jtype) jbooleanArray "boolean[]"
%typemap(jtype) jcharArray    "char[]"
%typemap(jtype) jbyteArray    "byte[]"
%typemap(jtype) jshortArray   "short[]"
%typemap(jtype) jintArray     "int[]"
%typemap(jtype) jlongArray    "long[]"
%typemap(jtype) jfloatArray   "float[]"
%typemap(jtype) jdoubleArray  "double[]"
%typemap(jtype) jobjectArray  "java.lang.Object[]"

%typemap(jstype) jboolean      "boolean"
%typemap(jstype) jchar         "char"
%typemap(jstype) jbyte         "byte"
%typemap(jstype) jshort        "short"
%typemap(jstype) jint          "int"
%typemap(jstype) jlong         "long"
%typemap(jstype) jfloat        "float"
%typemap(jstype) jdouble       "double"
%typemap(jstype) jstring       "String"
%typemap(jstype) jobject       "java.lang.Object"
%typemap(jstype) jbooleanArray "boolean[]"
%typemap(jstype) jcharArray    "char[]"
%typemap(jstype) jbyteArray    "byte[]"
%typemap(jstype) jshortArray   "short[]"
%typemap(jstype) jintArray     "int[]"
%typemap(jstype) jlongArray    "long[]"
%typemap(jstype) jfloatArray   "float[]"
%typemap(jstype) jdoubleArray  "double[]"
%typemap(jstype) jobjectArray  "java.lang.Object[]"

/* Non primitive types */
%typemap(jni) SWIGTYPE "jlong"
%typemap(jtype) SWIGTYPE "long"
%typemap(jstype) SWIGTYPE "$&javaclassname"
%typemap(jboxtype) SWIGTYPE "$typemap(jstype, $1_type)"

%typemap(jni) SWIGTYPE [] "jlong"
%typemap(jtype) SWIGTYPE [] "long"
%typemap(jstype) SWIGTYPE [] "$javaclassname"

%typemap(jni) SWIGTYPE * "jlong"
%typemap(jtype) SWIGTYPE * "long"
%typemap(jstype) SWIGTYPE * "$javaclassname"

%typemap(jni) SWIGTYPE & "jlong"
%typemap(jtype) SWIGTYPE & "long"
%typemap(jstype) SWIGTYPE & "$javaclassname"

%typemap(jni) SWIGTYPE && "jlong"
%typemap(jtype) SWIGTYPE && "long"
%typemap(jstype) SWIGTYPE && "$javaclassname"

/* pointer to a class member */
%typemap(jni) SWIGTYPE (CLASS::*) "jstring"
%typemap(jtype) SWIGTYPE (CLASS::*) "String"
%typemap(jstype) SWIGTYPE (CLASS::*) "$javaclassname"

/* The following are the in, out, freearg, argout typemaps. These are the JNI code generating typemaps for converting from Java to C and visa versa. */

/* primitive types */
%typemap(in) bool
%{ $1 = $input ? true : false; %}

%typemap(directorout) bool
%{ $result = $input ? true : false; %}

%typemap(javadirectorin) bool "$jniinput"
%typemap(javadirectorout) bool "$javacall"

%typemap(in) char, 
             signed char, 
             unsigned char, 
             short, 
             unsigned short, 
             int, 
             unsigned int, 
             long, 
             unsigned long, 
             long long, 
             float, 
             double
%{ $1 = ($1_ltype)$input; %}

%typemap(directorout) char, 
             signed char, 
             unsigned char, 
             short, 
             unsigned short, 
             int, 
             unsigned int, 
             long, 
             unsigned long, 
             long long, 
             float, 
             double
%{ $result = ($1_ltype)$input; %}

%typemap(directorin, descriptor="Z") bool             "$input = (jboolean) $1;"
%typemap(directorin, descriptor="C") char             "$input = (jint) $1;"
%typemap(directorin, descriptor="B") signed char      "$input = (jbyte) $1;"
%typemap(directorin, descriptor="S") unsigned char    "$input = (jshort) $1;"
%typemap(directorin, descriptor="S") short            "$input = (jshort) $1;"
%typemap(directorin, descriptor="I") unsigned short   "$input = (jint) $1;"
%typemap(directorin, descriptor="I") int              "$input = (jint) $1;"
%typemap(directorin, descriptor="J") unsigned int     "$input = (jlong) $1;"
%typemap(directorin, descriptor="I") long             "$input = (jint) $1;"
%typemap(directorin, descriptor="J") unsigned long    "$input = (jlong) $1;"
%typemap(directorin, descriptor="J") long long        "$input = (jlong) $1;"
%typemap(directorin, descriptor="F") float            "$input = (jfloat) $1;"
%typemap(directorin, descriptor="D") double           "$input = (jdouble) $1;"

%typemap(javadirectorin) char, 
                         signed char, 
                         unsigned char, 
                         short, 
                         unsigned short, 
                         int, 
                         unsigned int, 
                         long, 
                         unsigned long, 
                         long long, 
                         float, 
                         double
  "$jniinput"

%typemap(javadirectorout) char, 
                          signed char, 
                          unsigned char, 
                          short, 
                          unsigned short, 
                          int, 
                          unsigned int, 
                          long, 
                          unsigned long, 
                          long long, 
                          float, 
                          double
  "$javacall"

%typemap(out) bool           %{ $result = (jboolean)$1; %}
%typemap(out) char           %{ $result = (jchar)$1; %}
%typemap(out) signed char    %{ $result = (jbyte)$1; %}
%typemap(out) unsigned char  %{ $result = (jshort)$1; %}
%typemap(out) short          %{ $result = (jshort)$1; %}
%typemap(out) unsigned short %{ $result = (jint)$1; %}
%typemap(out) int            %{ $result = (jint)$1; %}
%typemap(out) unsigned int   %{ $result = (jlong)$1; %}
%typemap(out) long           %{ $result = (jint)$1; %}
%typemap(out) unsigned long  %{ $result = (jlong)$1; %}
%typemap(out) long long      %{ $result = (jlong)$1; %}
%typemap(out) float          %{ $result = (jfloat)$1; %}
%typemap(out) double         %{ $result = (jdouble)$1; %}

/* unsigned long long */
/* Convert from BigInteger using the toByteArray member function */
%typemap(in) unsigned long long { 
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
  $1 = 0;
  if (sz > 0) {
    $1 = ($1_type)(signed char)bae[0];
    for(i=1; i<sz; i++) {
      $1 = ($1 << 8) | ($1_type)(unsigned char)bae[i];
    }
  }
  JCALL3(ReleaseByteArrayElements, jenv, ba, bae, 0);
}

%typemap(directorout) unsigned long long { 
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
  $result = 0;
  if (sz > 0) {
    $result = ($1_type)(signed char)bae[0];
    for(i=1; i<sz; i++) {
      $result = ($result << 8) | ($1_type)(unsigned char)bae[i];
    }
  }
  JCALL3(ReleaseByteArrayElements, jenv, ba, bae, 0);
}


/* Convert to BigInteger - byte array holds number in 2's complement big endian format */
%typemap(out) unsigned long long { 
  jbyteArray ba = JCALL1(NewByteArray, jenv, 9);
  jbyte* bae = JCALL2(GetByteArrayElements, jenv, ba, 0);
  jclass clazz = JCALL1(FindClass, jenv, "java/math/BigInteger");
  jmethodID mid = JCALL3(GetMethodID, jenv, clazz, "<init>", "([B)V");
  jobject bigint;
  int i;

  bae[0] = 0;
  for(i=1; i<9; i++ ) {
    bae[i] = (jbyte)($1>>8*(8-i));
  }

  JCALL3(ReleaseByteArrayElements, jenv, ba, bae, 0);
  bigint = JCALL3(NewObject, jenv, clazz, mid, ba);
  JCALL1(DeleteLocalRef, jenv, ba);
  $result = bigint;
}

/* Convert to BigInteger (see out typemap) */
%typemap(directorin, descriptor="Ljava/math/BigInteger;", noblock=1) unsigned long long, const unsigned long long & {
{
  jbyteArray ba = JCALL1(NewByteArray, jenv, 9);
  jbyte* bae = JCALL2(GetByteArrayElements, jenv, ba, 0);
  jclass clazz = JCALL1(FindClass, jenv, "java/math/BigInteger");
  jmethodID mid = JCALL3(GetMethodID, jenv, clazz, "<init>", "([B)V");
  jobject bigint;
  int swig_i;

  bae[0] = 0;
  for(swig_i=1; swig_i<9; swig_i++ ) {
    bae[swig_i] = (jbyte)($1>>8*(8-swig_i));
  }

  JCALL3(ReleaseByteArrayElements, jenv, ba, bae, 0);
  bigint = JCALL3(NewObject, jenv, clazz, mid, ba);
  JCALL1(DeleteLocalRef, jenv, ba);
  $input = bigint;
}
Swig::LocalRefGuard $1_refguard(jenv, $input); }

%typemap(javadirectorin) unsigned long long "$jniinput"
%typemap(javadirectorout) unsigned long long "$javacall"

/* char * - treat as String */
%typemap(in, noblock=1) char * {
 $1 = 0;
  if ($input) {
    $1 = ($1_ltype)JCALL2(GetStringUTFChars, jenv, $input, 0);
    if (!$1) return $null;
  }
}

%typemap(directorout, noblock=1, warning=SWIGWARN_TYPEMAP_DIRECTOROUT_PTR_MSG) char * {
  $1 = 0;
  if ($input) {
    $result = ($1_ltype)JCALL2(GetStringUTFChars, jenv, $input, 0);
    if (!$result) return $null;
  }
}

%typemap(directorin, descriptor="Ljava/lang/String;", noblock=1) char * {
  $input = 0;
  if ($1) {
    $input = JCALL1(NewStringUTF, jenv, (const char *)$1);
    if (!$input) return $null;
  }
  Swig::LocalRefGuard $1_refguard(jenv, $input);
}

%typemap(freearg, noblock=1) char * { if ($1) JCALL2(ReleaseStringUTFChars, jenv, $input, (const char *)$1); }
%typemap(out, noblock=1) char * { if ($1) $result = JCALL1(NewStringUTF, jenv, (const char *)$1); }
%typemap(javadirectorin) char * "$jniinput"
%typemap(javadirectorout) char * "$javacall"

/* char *& - treat as String */
%typemap(in, noblock=1) char *& ($*1_ltype temp = 0) {
 $1 = 0;
  if ($input) {
    temp = ($*1_ltype)JCALL2(GetStringUTFChars, jenv, $input, 0);
    if (!temp) return $null;
  }
  $1 = &temp;
}
%typemap(freearg, noblock=1) char *& { if ($1 && *$1) JCALL2(ReleaseStringUTFChars, jenv, $input, (const char *)*$1); }
%typemap(out, noblock=1) char *& { if (*$1) $result = JCALL1(NewStringUTF, jenv, (const char *)*$1); }

%typemap(out) void ""
%typemap(javadirectorin) void "$jniinput"
%typemap(javadirectorout) void "$javacall"
%typemap(directorin, descriptor="V") void ""

/* primitive types by reference */
%typemap(in) const bool & ($*1_ltype temp)
%{ temp = $input ? true : false; 
   $1 = &temp; %}

%typemap(directorout,warning=SWIGWARN_TYPEMAP_THREAD_UNSAFE_MSG) const bool &
%{ static $*1_ltype temp;
   temp = $input ? true : false; 
   $result = &temp; %}

%typemap(javadirectorin) const bool & "$jniinput"
%typemap(javadirectorout) const bool & "$javacall"

%typemap(in) const char & ($*1_ltype temp), 
             const signed char & ($*1_ltype temp), 
             const unsigned char & ($*1_ltype temp), 
             const short & ($*1_ltype temp), 
             const unsigned short & ($*1_ltype temp), 
             const int & ($*1_ltype temp), 
             const unsigned int & ($*1_ltype temp), 
             const long & ($*1_ltype temp), 
             const unsigned long & ($*1_ltype temp), 
             const long long & ($*1_ltype temp), 
             const float & ($*1_ltype temp), 
             const double & ($*1_ltype temp)
%{ temp = ($*1_ltype)$input; 
   $1 = &temp; %}

%typemap(directorout,warning=SWIGWARN_TYPEMAP_THREAD_UNSAFE_MSG) const char &,
             const signed char &,
             const unsigned char &,
             const short &,
             const unsigned short &,
             const int &,
             const unsigned int &,
             const long &,
             const unsigned long &,
             const long long &,
             const float &,
             const double &
%{ static $*1_ltype temp;
   temp = ($*1_ltype)$input; 
   $result = &temp; %}

%typemap(directorin, descriptor="Z") const bool &           "$input = (jboolean)$1;"
%typemap(directorin, descriptor="C") const char &           "$input = (jchar)$1;"
%typemap(directorin, descriptor="B") const signed char &    "$input = (jbyte)$1;"
%typemap(directorin, descriptor="S") const unsigned char &  "$input = (jshort)$1;"
%typemap(directorin, descriptor="S") const short &          "$input = (jshort)$1;"
%typemap(directorin, descriptor="I") const unsigned short & "$input = (jint)$1;"
%typemap(directorin, descriptor="I") const int &            "$input = (jint)$1;"
%typemap(directorin, descriptor="J") const unsigned int &   "$input = (jlong)$1;"
%typemap(directorin, descriptor="I") const long &           "$input = (jint)$1;"
%typemap(directorin, descriptor="J") const unsigned long &  "$input = (jlong)$1;"
%typemap(directorin, descriptor="J") const long long &      "$input = (jlong)$1;"
%typemap(directorin, descriptor="F") const float &          "$input = (jfloat)$1;"
%typemap(directorin, descriptor="D") const double &         "$input = (jdouble)$1;"

%typemap(javadirectorin) const char & ($*1_ltype temp), 
                         const signed char & ($*1_ltype temp), 
                         const unsigned char & ($*1_ltype temp), 
                         const short & ($*1_ltype temp), 
                         const unsigned short & ($*1_ltype temp), 
                         const int & ($*1_ltype temp), 
                         const unsigned int & ($*1_ltype temp), 
                         const long & ($*1_ltype temp), 
                         const unsigned long & ($*1_ltype temp), 
                         const long long & ($*1_ltype temp), 
                         const float & ($*1_ltype temp), 
                         const double & ($*1_ltype temp)
  "$jniinput"

%typemap(javadirectorout) const char & ($*1_ltype temp), 
                          const signed char & ($*1_ltype temp), 
                          const unsigned char & ($*1_ltype temp), 
                          const short & ($*1_ltype temp), 
                          const unsigned short & ($*1_ltype temp), 
                          const int & ($*1_ltype temp), 
                          const unsigned int & ($*1_ltype temp), 
                          const long & ($*1_ltype temp), 
                          const unsigned long & ($*1_ltype temp), 
                          const long long & ($*1_ltype temp), 
                          const float & ($*1_ltype temp), 
                          const double & ($*1_ltype temp)
  "$javacall"


%typemap(out) const bool &           %{ $result = (jboolean)*$1; %}
%typemap(out) const char &           %{ $result = (jchar)*$1; %}
%typemap(out) const signed char &    %{ $result = (jbyte)*$1; %}
%typemap(out) const unsigned char &  %{ $result = (jshort)*$1; %}
%typemap(out) const short &          %{ $result = (jshort)*$1; %}
%typemap(out) const unsigned short & %{ $result = (jint)*$1; %}
%typemap(out) const int &            %{ $result = (jint)*$1; %}
%typemap(out) const unsigned int &   %{ $result = (jlong)*$1; %}
%typemap(out) const long &           %{ $result = (jint)*$1; %}
%typemap(out) const unsigned long &  %{ $result = (jlong)*$1; %}
%typemap(out) const long long &      %{ $result = (jlong)*$1; %}
%typemap(out) const float &          %{ $result = (jfloat)*$1; %}
%typemap(out) const double &         %{ $result = (jdouble)*$1; %}

/* const unsigned long long & */
/* Similar to unsigned long long */
%typemap(in) const unsigned long long & ($*1_ltype temp) { 
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
  $1 = &temp;
  temp = 0;
  if (sz > 0) {
    temp = ($*1_ltype)(signed char)bae[0];
    for(i=1; i<sz; i++) {
      temp = (temp << 8) | ($*1_ltype)(unsigned char)bae[i];
    }
  }
  JCALL3(ReleaseByteArrayElements, jenv, ba, bae, 0);
}

%typemap(directorout,warning=SWIGWARN_TYPEMAP_THREAD_UNSAFE_MSG) const unsigned long long & { 
  static $*1_ltype temp;
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
  $result = &temp;
  temp = 0;
  if (sz > 0) {
    temp = ($*1_ltype)(signed char)bae[0];
    for(i=1; i<sz; i++) {
      temp = (temp << 8) | ($*1_ltype)(unsigned char)bae[i];
    }
  }
  JCALL3(ReleaseByteArrayElements, jenv, ba, bae, 0);
}

%typemap(out) const unsigned long long & { 
  jbyteArray ba = JCALL1(NewByteArray, jenv, 9);
  jbyte* bae = JCALL2(GetByteArrayElements, jenv, ba, 0);
  jclass clazz = JCALL1(FindClass, jenv, "java/math/BigInteger");
  jmethodID mid = JCALL3(GetMethodID, jenv, clazz, "<init>", "([B)V");
  jobject bigint;
  int i;

  bae[0] = 0;
  for(i=1; i<9; i++ ) {
    bae[i] = (jbyte)(*$1>>8*(8-i));
  }

  JCALL3(ReleaseByteArrayElements, jenv, ba, bae, 0);
  bigint = JCALL3(NewObject, jenv, clazz, mid, ba);
  JCALL1(DeleteLocalRef, jenv, ba);
  $result = bigint;
}

%typemap(javadirectorin) const unsigned long long & "$jniinput"
%typemap(javadirectorout) const unsigned long long & "$javacall"

/* Default handling. Object passed by value. Convert to a pointer */
%typemap(in) SWIGTYPE ($&1_type argp)
%{ argp = *($&1_ltype*)&$input; 
   if (!argp) {
     SWIG_JavaThrowException(jenv, SWIG_JavaNullPointerException, "Attempt to dereference null $1_type");
     return $null;
   }
   $1 = *argp; %}

%typemap(directorout) SWIGTYPE ($&1_type argp)
%{ argp = *($&1_ltype*)&$input; 
   if (!argp) {
     SWIG_JavaThrowException(jenv, SWIG_JavaNullPointerException, "Unexpected null return for type $1_type");
     return $null;
   }
   $result = *argp; %}

%typemap(out) SWIGTYPE 
#ifdef __cplusplus
%{ *($&1_ltype*)&$result = new $1_ltype($1); %}
#else
{
  $&1_ltype $1ptr = ($&1_ltype) malloc(sizeof($1_ltype));
  memmove($1ptr, &$1, sizeof($1_type));
  *($&1_ltype*)&$result = $1ptr;
}
#endif

%typemap(directorin,descriptor="L$packagepath/$&javaclassname;") SWIGTYPE 
%{ $input = 0;
   *(($&1_ltype*)&$input) = new $1_ltype(SWIG_STD_MOVE($1)); %}
%typemap(javadirectorin) SWIGTYPE "new $&javaclassname($jniinput, true)"
%typemap(javadirectorout) SWIGTYPE "$&javaclassname.getCPtr($javacall)"

/* Generic pointers and references */
%typemap(in) SWIGTYPE * %{ $1 = *($&1_ltype)&$input; %}
%typemap(in, fragment="SWIG_UnPackData") SWIGTYPE (CLASS::*) { 
  const char *temp = 0;
  if ($input) {
    temp = JCALL2(GetStringUTFChars, jenv, $input, 0);
    if (!temp) return $null;
  }
  SWIG_UnpackData(temp, (void *)&$1, sizeof($1));
}
%typemap(in) SWIGTYPE & %{ $1 = *($&1_ltype)&$input;
  if (!$1) {
    SWIG_JavaThrowException(jenv, SWIG_JavaNullPointerException, "$1_type is null");
    return $null;
  } %}
%typemap(in, fragment="<memory>") SWIGTYPE && (std::unique_ptr<$*1_ltype> rvrdeleter) %{ $1 = *($&1_ltype)&$input;
  if (!$1) {
    SWIG_JavaThrowException(jenv, SWIG_JavaNullPointerException, "$1_type is null");
    return $null;
  }
  rvrdeleter.reset($1); %}
%typemap(out) SWIGTYPE *
%{ *($&1_ltype)&$result = $1; %} 
%typemap(out, fragment="SWIG_PackData", noblock=1) SWIGTYPE (CLASS::*) {
  char buf[128];
  char *data = SWIG_PackData(buf, (void *)&$1, sizeof($1));
  *data = '\0';
  $result = JCALL1(NewStringUTF, jenv, buf);
}
%typemap(out) SWIGTYPE &
%{ *($&1_ltype)&$result = $1; %} 
%typemap(out) SWIGTYPE &&
%{ *($&1_ltype)&$result = $1; %} 

%typemap(directorout, warning=SWIGWARN_TYPEMAP_DIRECTOROUT_PTR_MSG) SWIGTYPE *
%{ $result = *($&1_ltype)&$input; %}
%typemap(directorout, warning=SWIGWARN_TYPEMAP_DIRECTOROUT_PTR_MSG) SWIGTYPE (CLASS::*)
%{ $result = *($&1_ltype)&$input; %}

%typemap(directorin,descriptor="L$packagepath/$javaclassname;") SWIGTYPE *
%{ *(($&1_ltype)&$input) = ($1_ltype) $1; %}
%typemap(directorin,descriptor="L$packagepath/$javaclassname;") SWIGTYPE (CLASS::*)
%{ *(($&1_ltype)&$input) = ($1_ltype) $1; %}

%typemap(directorout, warning=SWIGWARN_TYPEMAP_DIRECTOROUT_PTR_MSG) SWIGTYPE &
%{ if (!$input) {
     SWIG_JavaThrowException(jenv, SWIG_JavaNullPointerException, "Unexpected null return for type $1_type");
     return $null;
   }
   $result = *($&1_ltype)&$input; %}
%typemap(directorout, warning=SWIGWARN_TYPEMAP_DIRECTOROUT_PTR_MSG) SWIGTYPE &&
%{ if (!$input) {
     SWIG_JavaThrowException(jenv, SWIG_JavaNullPointerException, "Unexpected null return for type $1_type");
     return $null;
   }
   $result = *($&1_ltype)&$input; %}
%typemap(directorin,descriptor="L$packagepath/$javaclassname;") SWIGTYPE &
%{ *($&1_ltype)&$input = ($1_ltype) &$1; %}
%typemap(directorin,descriptor="L$packagepath/$javaclassname;") SWIGTYPE &&
%{ *($&1_ltype)&$input = ($1_ltype) &$1; %}

%typemap(javadirectorin) SWIGTYPE *, SWIGTYPE (CLASS::*) "($jniinput == 0) ? null : new $javaclassname($jniinput, false)"
%typemap(javadirectorin) SWIGTYPE & "new $javaclassname($jniinput, false)"
%typemap(javadirectorin) SWIGTYPE && "new $javaclassname($jniinput, false)"
%typemap(javadirectorout) SWIGTYPE *, SWIGTYPE (CLASS::*), SWIGTYPE &, SWIGTYPE && "$javaclassname.getCPtr($javacall)"

/* Default array handling */
%typemap(in) SWIGTYPE [] %{ $1 = *($&1_ltype)&$input; %}
%typemap(out) SWIGTYPE [] %{ *($&1_ltype)&$result = $1; %} 
%typemap(freearg) SWIGTYPE [ANY], SWIGTYPE [] ""

/* char arrays - treat as String */
%typemap(in, noblock=1) char[ANY], char[] {
  $1 = 0;
  if ($input) {
    $1 = ($1_ltype)JCALL2(GetStringUTFChars, jenv, $input, 0);
    if (!$1) return $null;
  }
}

%typemap(directorout, noblock=1) char[ANY], char[] {
  $1 = 0;
  if ($input) {
    $result = ($1_ltype)JCALL2(GetStringUTFChars, jenv, $input, 0);
    if (!$result) return $null;
  }
}

%typemap(directorin, descriptor="Ljava/lang/String;", noblock=1) char[ANY], char[] {
  $input = 0;
  if ($1) {
    $input = JCALL1(NewStringUTF, jenv, (const char *)$1);
    if (!$input) return $null;
  }
  Swig::LocalRefGuard $1_refguard(jenv, $input);
}

%typemap(argout) char[ANY], char[] ""
%typemap(freearg, noblock=1) char[ANY], char[] { if ($1) JCALL2(ReleaseStringUTFChars, jenv, $input, (const char *)$1); }
%typemap(out, noblock=1) char[ANY], char[] { if ($1) $result = JCALL1(NewStringUTF, jenv, (const char *)$1); }
%typemap(javadirectorin) char[ANY], char[] "$jniinput"
%typemap(javadirectorout) char[ANY], char[] "$javacall"

/* JNI types */
%typemap(in) jboolean,
             jchar,
             jbyte,
             jshort,
             jint,
             jlong,
             jfloat,
             jdouble,
             jstring,
             jobject,
             jbooleanArray,
             jcharArray,
             jbyteArray,
             jshortArray,
             jintArray,
             jlongArray,
             jfloatArray,
             jdoubleArray,
             jobjectArray
%{ $1 = $input; %}

%typemap(directorout) jboolean,
             jchar,
             jbyte,
             jshort,
             jint,
             jlong,
             jfloat,
             jdouble,
             jstring,
             jobject,
             jbooleanArray,
             jcharArray,
             jbyteArray,
             jshortArray,
             jintArray,
             jlongArray,
             jfloatArray,
             jdoubleArray,
             jobjectArray
%{ $result = $input; %}

%typemap(out) jboolean,
              jchar,
              jbyte,
              jshort,
              jint,
              jlong,
              jfloat,
              jdouble,
              jstring,
              jobject,
              jbooleanArray,
              jcharArray,
              jbyteArray,
              jshortArray,
              jintArray,
              jlongArray,
              jfloatArray,
              jdoubleArray,
              jobjectArray
%{ $result = $1; %}

%typemap(directorin,descriptor="Z")  jboolean       "$input = $1;"
%typemap(directorin,descriptor="C")  jchar          "$input = $1;"
%typemap(directorin,descriptor="B")  jbyte          "$input = $1;"
%typemap(directorin,descriptor="S")  jshort         "$input = $1;"
%typemap(directorin,descriptor="I")  jint           "$input = $1;"
%typemap(directorin,descriptor="J")  jlong          "$input = $1;"
%typemap(directorin,descriptor="F")  jfloat         "$input = $1;"
%typemap(directorin,descriptor="D")  jdouble        "$input = $1;"
%typemap(directorin,descriptor="Ljava/lang/String;")            jstring        "$input = $1;"
%typemap(directorin,descriptor="Ljava/lang/Object;",nouse="1")  jobject        "$input = $1;"
%typemap(directorin,descriptor="[Z")  jbooleanArray "$input = $1;"
%typemap(directorin,descriptor="[C")  jcharArray    "$input = $1;"
%typemap(directorin,descriptor="[B")  jbyteArray    "$input = $1;"
%typemap(directorin,descriptor="[S")  jshortArray   "$input = $1;"
%typemap(directorin,descriptor="[I")  jintArray     "$input = $1;"
%typemap(directorin,descriptor="[J")  jlongArray    "$input = $1;"
%typemap(directorin,descriptor="[F")  jfloatArray   "$input = $1;"
%typemap(directorin,descriptor="[D")  jdoubleArray  "$input = $1;"
%typemap(directorin,descriptor="[Ljava/lang/Object;",nouse="1") jobjectArray   "$input = $1;"

%typemap(javadirectorin) jboolean,
                         jchar,
                         jbyte,
                         jshort,
                         jint,
                         jlong,
                         jfloat,
                         jdouble,
                         jstring,
                         jobject,
                         jbooleanArray,
                         jcharArray,
                         jbyteArray,
                         jshortArray,
                         jintArray,
                         jlongArray,
                         jfloatArray,
                         jdoubleArray,
                         jobjectArray
  "$jniinput"

%typemap(javadirectorout) jboolean,
                          jchar,
                          jbyte,
                          jshort,
                          jint,
                          jlong,
                          jfloat,
                          jdouble,
                          jstring,
                          jobject,
                          jbooleanArray,
                          jcharArray,
                          jbyteArray,
                          jshortArray,
                          jintArray,
                          jlongArray,
                          jfloatArray,
                          jdoubleArray,
                          jobjectArray
  "$javacall"

/* Typecheck typemaps - The purpose of these is merely to issue a warning for overloaded C++ functions 
 * that cannot be overloaded in Java as more than one C++ type maps to a single Java type */

%typecheck(SWIG_TYPECHECK_BOOL) /* Java boolean */
    jboolean,
    bool,
    const bool &
    ""

%typecheck(SWIG_TYPECHECK_CHAR) /* Java char */
    jchar,
    char, 
    const char &
    ""

%typecheck(SWIG_TYPECHECK_INT8) /* Java byte */
    jbyte,
    signed char,
    const signed char &
    ""

%typecheck(SWIG_TYPECHECK_INT16) /* Java short */
    jshort,
    unsigned char, 
    short, 
    const unsigned char &, 
    const short &
    ""

%typecheck(SWIG_TYPECHECK_INT32) /* Java int */
    jint,
    unsigned short, 
    int, 
    long, 
    const unsigned short &, 
    const int &, 
    const long &
    ""

%typecheck(SWIG_TYPECHECK_INT64) /* Java long */
    jlong,
    unsigned int, 
    unsigned long, 
    long long, 
    const unsigned int &, 
    const unsigned long &, 
    const long long &
    ""

%typecheck(SWIG_TYPECHECK_INT128) /* Java BigInteger */
    unsigned long long,
    const unsigned long long &
    ""

%typecheck(SWIG_TYPECHECK_FLOAT) /* Java float */
    jfloat,
    float,
    const float &
    ""

%typecheck(SWIG_TYPECHECK_DOUBLE) /* Java double */
    jdouble,
    double,
    const double &
    ""

%typecheck(SWIG_TYPECHECK_STRING) /* Java String */
    jstring,
    char *,
    char *&,
    char[ANY],
    char []
    ""

%typecheck(SWIG_TYPECHECK_BOOL_ARRAY) /* Java boolean[] */
    jbooleanArray
    ""

%typecheck(SWIG_TYPECHECK_CHAR_ARRAY) /* Java char[] */
    jcharArray
    ""

%typecheck(SWIG_TYPECHECK_INT8_ARRAY) /* Java byte[] */
    jbyteArray
    ""

%typecheck(SWIG_TYPECHECK_INT16_ARRAY) /* Java short[] */
    jshortArray
    ""

%typecheck(SWIG_TYPECHECK_INT32_ARRAY) /* Java int[] */
    jintArray
    ""

%typecheck(SWIG_TYPECHECK_INT64_ARRAY) /* Java long[] */
    jlongArray
    ""

%typecheck(SWIG_TYPECHECK_FLOAT_ARRAY) /* Java float[] */
    jfloatArray
    ""

%typecheck(SWIG_TYPECHECK_DOUBLE_ARRAY) /* Java double[] */
    jdoubleArray
    ""

%typecheck(SWIG_TYPECHECK_OBJECT_ARRAY) /* Java jobject[] */
    jobjectArray
    ""

%typecheck(SWIG_TYPECHECK_POINTER) /* Default */
    SWIGTYPE, 
    SWIGTYPE *, 
    SWIGTYPE &, 
    SWIGTYPE &&, 
    SWIGTYPE *const&, 
    SWIGTYPE [],
    SWIGTYPE (CLASS::*)
    ""


/* Exception handling */

%typemap(throws) int, 
                 long, 
                 short, 
                 unsigned int, 
                 unsigned long, 
                 unsigned short
%{ char error_msg[256];
   sprintf(error_msg, "C++ $1_type exception thrown, value: %d", $1);
   SWIG_JavaThrowException(jenv, SWIG_JavaRuntimeException, error_msg);
   return $null; %}

%typemap(throws) SWIGTYPE, SWIGTYPE &, SWIGTYPE &&, SWIGTYPE *, SWIGTYPE [], SWIGTYPE [ANY]
%{ (void)$1;
   SWIG_JavaThrowException(jenv, SWIG_JavaRuntimeException, "C++ $1_type exception thrown");
   return $null; %}

%typemap(throws) char *
%{ SWIG_JavaThrowException(jenv, SWIG_JavaRuntimeException, $1);
   return $null; %}

/* For methods to raise/throw the original Java exception thrown in a director method */
%typemap(throws) Swig::DirectorException
%{  $1.throwException(jenv);
    return $null; %}

/* Java to C++ DirectorException should already be handled. Suppress warning and do nothing in the
   event a user specifies a global: %catches(Swig::DirectorException); */
%typemap(directorthrows) Swig::DirectorException ""

/* Typemaps for code generation in proxy classes and Java type wrapper classes */

/* The javain typemap is used for converting function parameter types from the type 
 * used in the proxy, module or type wrapper class to the type used in the JNI class. */
%typemap(javain) bool,               const bool &,
                 char,               const char &,
                 signed char,        const signed char &,
                 unsigned char,      const unsigned char &,
                 short,              const short &,
                 unsigned short,     const unsigned short &,
                 int,                const int &,
                 unsigned int,       const unsigned int &,
                 long,               const long &,
                 unsigned long,      const unsigned long &,
                 long long,          const long long &,
                 unsigned long long, const unsigned long long &,
                 float,              const float &,
                 double,             const double &
    "$javainput"
%typemap(javain) char *, char *&, char[ANY], char[] "$javainput"
%typemap(javain) jboolean,
                 jchar,
                 jbyte,
                 jshort,
                 jint,
                 jlong,
                 jfloat,
                 jdouble,
                 jstring,
                 jobject,
                 jbooleanArray,
                 jcharArray,
                 jbyteArray,
                 jshortArray,
                 jintArray,
                 jlongArray,
                 jfloatArray,
                 jdoubleArray,
                 jobjectArray
    "$javainput"
%typemap(javain) SWIGTYPE "$&javaclassname.getCPtr($javainput)"
%typemap(javain) SWIGTYPE *, SWIGTYPE &, SWIGTYPE [] "$javaclassname.getCPtr($javainput)"
%typemap(javain) SWIGTYPE && "$javaclassname.swigRelease($javainput)"
%typemap(javain) SWIGTYPE (CLASS::*) "$javaclassname.getCMemberPtr($javainput)"

/* The javaout typemap is used for converting function return types from the return type
 * used in the JNI class to the type returned by the proxy, module or type wrapper class. */
%typemap(javaout) bool,               const bool &,
                  char,               const char &,
                  signed char,        const signed char &,
                  unsigned char,      const unsigned char &,
                  short,              const short &,
                  unsigned short,     const unsigned short &,
                  int,                const int &,
                  unsigned int,       const unsigned int &,
                  long,               const long &,
                  unsigned long,      const unsigned long &,
                  long long,          const long long &,
                  unsigned long long, const unsigned long long &,
                  float,              const float &,
                  double,             const double & {
    return $jnicall;
  }
%typemap(javaout) char *, char *&, char[ANY], char[] {
    return $jnicall;
  }
%typemap(javaout) jboolean,
                  jchar,
                  jbyte,
                  jshort,
                  jint,
                  jlong,
                  jfloat,
                  jdouble,
                  jstring,
                  jobject,
                  jbooleanArray,
                  jcharArray,
                  jbyteArray,
                  jshortArray,
                  jintArray,
                  jlongArray,
                  jfloatArray,
                  jdoubleArray,
                  jobjectArray {
    return $jnicall;
  }
%typemap(javaout) void {
    $jnicall;
  }
%typemap(javaout) SWIGTYPE {
    return new $&javaclassname($jnicall, true);
  }
%typemap(javaout) SWIGTYPE & {
    return new $javaclassname($jnicall, $owner);
  }
%typemap(javaout) SWIGTYPE && {
    return new $javaclassname($jnicall, $owner);
  }
%typemap(javaout) SWIGTYPE *, SWIGTYPE [] {
    long cPtr = $jnicall;
    return (cPtr == 0) ? null : new $javaclassname(cPtr, $owner);
  }
%typemap(javaout) SWIGTYPE (CLASS::*) {
    String cMemberPtr = $jnicall;
    return (cMemberPtr == null) ? null : new $javaclassname(cMemberPtr, $owner);
  }

/* Pointer reference typemaps */
%typemap(jni) SWIGTYPE *const& "jlong"
%typemap(jtype) SWIGTYPE *const& "long"
%typemap(jstype) SWIGTYPE *const& "$*javaclassname"
%typemap(javain) SWIGTYPE *const& "$*javaclassname.getCPtr($javainput)"
%typemap(javaout) SWIGTYPE *const& {
    long cPtr = $jnicall;
    return (cPtr == 0) ? null : new $*javaclassname(cPtr, $owner);
  }
%typemap(in) SWIGTYPE *const& ($*1_ltype temp = 0)
%{ temp = *($1_ltype)&$input;
   $1 = ($1_ltype)&temp; %}
%typemap(out) SWIGTYPE *const&
%{ *($1_ltype)&$result = *$1; %} 
%typemap(directorin,descriptor="L$packagepath/$*javaclassname;") SWIGTYPE *const&
%{ *(($1_ltype)&$input) = ($*1_ltype) $1; %}
%typemap(directorout, warning=SWIGWARN_TYPEMAP_THREAD_UNSAFE_MSG) SWIGTYPE *const&
%{ static $*1_ltype swig_temp;
   swig_temp = *($1_ltype)&$input;
   $result = &swig_temp; %}
%typemap(javadirectorin) SWIGTYPE *const& "($jniinput == 0) ? null : new $*javaclassname($jniinput, false)"
%typemap(javadirectorout) SWIGTYPE *const& "$*javaclassname.getCPtr($javacall)"

/* Typemaps used for the generation of proxy and type wrapper class code */
%typemap(javabase)             SWIGTYPE, SWIGTYPE *, SWIGTYPE &, SWIGTYPE &&, SWIGTYPE [], SWIGTYPE (CLASS::*) ""
%typemap(javaclassmodifiers)   SWIGTYPE, SWIGTYPE *, SWIGTYPE &, SWIGTYPE &&, SWIGTYPE [], SWIGTYPE (CLASS::*) "public class"
%typemap(javacode)             SWIGTYPE, SWIGTYPE *, SWIGTYPE &, SWIGTYPE &&, SWIGTYPE [], SWIGTYPE (CLASS::*) ""
%typemap(javaimports)          SWIGTYPE, SWIGTYPE *, SWIGTYPE &, SWIGTYPE &&, SWIGTYPE [], SWIGTYPE (CLASS::*) ""
%typemap(javainterfaces)       SWIGTYPE, SWIGTYPE *, SWIGTYPE &, SWIGTYPE &&, SWIGTYPE [], SWIGTYPE (CLASS::*) ""
%typemap(javainterfacemodifiers) SWIGTYPE, SWIGTYPE *, SWIGTYPE &, SWIGTYPE &&, SWIGTYPE [], SWIGTYPE (CLASS::*) "public interface"

/* javabody typemaps */

%define SWIG_JAVABODY_METHODS(PTRCTOR_VISIBILITY, CPTR_VISIBILITY, TYPE...) SWIG_JAVABODY_PROXY(PTRCTOR_VISIBILITY, CPTR_VISIBILITY, TYPE) %enddef // legacy name

%define SWIG_JAVABODY_PROXY(PTRCTOR_VISIBILITY, CPTR_VISIBILITY, TYPE...)
// Base proxy classes
%typemap(javabody) TYPE %{
  private transient long swigCPtr;
  protected transient boolean swigCMemOwn;

  PTRCTOR_VISIBILITY $javaclassname(long cPtr, boolean cMemoryOwn) {
    swigCMemOwn = cMemoryOwn;
    swigCPtr = cPtr;
  }

  CPTR_VISIBILITY static long getCPtr($javaclassname obj) {
    return (obj == null) ? 0 : obj.swigCPtr;
  }

  CPTR_VISIBILITY static long swigRelease($javaclassname obj) {
    long ptr = 0;
    if (obj != null) {
      if (!obj.swigCMemOwn)
        throw new RuntimeException("Cannot release ownership as memory is not owned");
      ptr = obj.swigCPtr;
      obj.swigCMemOwn = false;
      obj.delete();
    }
    return ptr;
  }
%}

// Derived proxy classes
%typemap(javabody_derived) TYPE %{
  private transient long swigCPtr;

  PTRCTOR_VISIBILITY $javaclassname(long cPtr, boolean cMemoryOwn) {
    super($imclassname.$javaclazznameSWIGUpcast(cPtr), cMemoryOwn);
    swigCPtr = cPtr;
  }

  CPTR_VISIBILITY static long getCPtr($javaclassname obj) {
    return (obj == null) ? 0 : obj.swigCPtr;
  }

  CPTR_VISIBILITY static long swigRelease($javaclassname obj) {
    long ptr = 0;
    if (obj != null) {
      if (!obj.swigCMemOwn)
        throw new RuntimeException("Cannot release ownership as memory is not owned");
      ptr = obj.swigCPtr;
      obj.swigCMemOwn = false;
      obj.delete();
    }
    return ptr;
  }
%}
%enddef

%define SWIG_JAVABODY_TYPEWRAPPER(PTRCTOR_VISIBILITY, DEFAULTCTOR_VISIBILITY, CPTR_VISIBILITY, TYPE...)
// Typewrapper classes
%typemap(javabody) TYPE *, TYPE &, TYPE &&, TYPE [] %{
  private transient long swigCPtr;

  PTRCTOR_VISIBILITY $javaclassname(long cPtr, @SuppressWarnings("unused") boolean futureUse) {
    swigCPtr = cPtr;
  }

  DEFAULTCTOR_VISIBILITY $javaclassname() {
    swigCPtr = 0;
  }

  CPTR_VISIBILITY static long getCPtr($javaclassname obj) {
    return (obj == null) ? 0 : obj.swigCPtr;
  }

  CPTR_VISIBILITY static long swigRelease($javaclassname obj) {
    return (obj == null) ? 0 : obj.swigCPtr;
  }
%}

%typemap(javabody) TYPE (CLASS::*) %{
  private transient String swigCMemberPtr;

  PTRCTOR_VISIBILITY $javaclassname(String cMemberPtr, @SuppressWarnings("unused") boolean futureUse) {
    swigCMemberPtr = cMemberPtr;
  }

  DEFAULTCTOR_VISIBILITY $javaclassname() {
    swigCMemberPtr = null;
  }

  CPTR_VISIBILITY static String getCMemberPtr($javaclassname obj) {
    return obj.swigCMemberPtr;
  }
%}
%enddef

/* Set the default javabody typemaps to use protected visibility.
   Use the macros to change to public if using multiple modules. */
SWIG_JAVABODY_PROXY(protected, protected, SWIGTYPE)
SWIG_JAVABODY_TYPEWRAPPER(protected, protected, protected, SWIGTYPE)

%typemap(javafinalize) SWIGTYPE %{
  @SuppressWarnings("deprecation")
  protected void finalize() {
    delete();
  }
%}

/*
 * Java constructor typemaps:
 *
 * The javaconstruct typemap is inserted when a proxy class's constructor is generated.
 * This typemap allows control over what code is executed in the constructor as
 * well as specifying who owns the underlying C/C++ object. Normally, Java has
 * ownership and the underlying C/C++ object is deallocated when the Java object
 * is finalized (swigCMemOwn is true.) If swigCMemOwn is false, C/C++ is
 * ultimately responsible for deallocating the underlying object's memory.
 *
 * The SWIG_PROXY_CONSTRUCTOR macro defines the javaconstruct typemap for a proxy
 * class for a particular TYPENAME. OWNERSHIP is passed as the value of
 * swigCMemOwn to the pointer constructor method.  WEAKREF determines which kind
 * of Java object reference will be used by the C++ director class (WeakGlobalRef
 * vs. GlobalRef.)
 *
 * The SWIG_DIRECTOR_OWNED macro sets the ownership of director-based proxy
 * classes and the weak reference flag to false, meaning that the underlying C++
 * object will be reclaimed by C++.
 */

%define SWIG_PROXY_CONSTRUCTOR(OWNERSHIP, WEAKREF, TYPENAME...)
%typemap(javaconstruct,directorconnect="\n    $imclassname.$javaclazznamedirector_connect(this, swigCPtr, OWNERSHIP, WEAKREF);") TYPENAME {
    this($imcall, OWNERSHIP);$directorconnect
  }
%enddef

%define SWIG_DIRECTOR_OWNED(TYPENAME...)
SWIG_PROXY_CONSTRUCTOR(true, false, TYPENAME)
%enddef

// Set the default for SWIGTYPE: Java owns the C/C++ object.
SWIG_PROXY_CONSTRUCTOR(true, true, SWIGTYPE)

%typemap(javadestruct, methodname="delete", methodmodifiers="public synchronized", parameters="") SWIGTYPE {
    if (swigCPtr != 0) {
      if (swigCMemOwn) {
        swigCMemOwn = false;
        $jnicall;
      }
      swigCPtr = 0;
    }
  }

%typemap(javadestruct_derived, methodname="delete", methodmodifiers="public synchronized", parameters="") SWIGTYPE {
    if (swigCPtr != 0) {
      if (swigCMemOwn) {
        swigCMemOwn = false;
        $jnicall;
      }
      swigCPtr = 0;
    }
    super.delete();
  }

%typemap(directordisconnect, methodname="swigDirectorDisconnect") SWIGTYPE %{
  protected void $methodname() {
    swigCMemOwn = false;
    $jnicall;
  }
%}

%typemap(directorowner_release, methodname="swigReleaseOwnership") SWIGTYPE %{
  public void $methodname() {
    swigCMemOwn = false;
    $jnicall;
  }
%}

%typemap(directorowner_take, methodname="swigTakeOwnership") SWIGTYPE %{
  public void $methodname() {
    swigCMemOwn = true;
    $jnicall;
  }
%}

/* Java specific directives */
#define %javaconst(flag)            %feature("java:const","flag")
#define %javaconstvalue(value)      %feature("java:constvalue",value)
#define %javaenum(wrapapproach)     %feature("java:enum","wrapapproach")
#define %javamethodmodifiers        %feature("java:methodmodifiers")
#define %javaexception(exceptionclasses) %feature("except",throws=exceptionclasses)
#define %nojavaexception            %feature("except","0",throws="")
#define %clearjavaexception         %feature("except","",throws="")
#define %proxycode                  %insert("proxycode")

%pragma(java) jniclassclassmodifiers="public class"
%pragma(java) moduleclassmodifiers="public class"

/* Some ANSI C typemaps */

%apply unsigned long { size_t };
%apply const unsigned long & { const size_t & };

/* Array reference typemaps */
%apply SWIGTYPE & { SWIGTYPE ((&)[ANY]) }
%apply SWIGTYPE && { SWIGTYPE ((&&)[ANY]) }

/* const pointers */
%apply SWIGTYPE * { SWIGTYPE *const }
%apply SWIGTYPE (CLASS::*) { SWIGTYPE (CLASS::*const) }
%apply SWIGTYPE & { SWIGTYPE (CLASS::*const&) }

/* String & length */
%typemap(jni)     (const char *STRING, size_t LENGTH) "jbyteArray"
%typemap(jtype)   (const char *STRING, size_t LENGTH) "byte[]"
%typemap(jstype)  (const char *STRING, size_t LENGTH) "byte[]"
%typemap(javain)  (const char *STRING, size_t LENGTH) "$javainput"
%typemap(freearg) (const char *STRING, size_t LENGTH) ""
%typemap(in)      (const char *STRING, size_t LENGTH) {
  if ($input) {
    $1 = ($1_ltype) JCALL2(GetByteArrayElements, jenv, $input, 0);
    $2 = ($2_type) JCALL1(GetArrayLength, jenv, $input);
  } else {
    $1 = 0;
    $2 = 0;
  }
}
%typemap(argout)  (const char *STRING, size_t LENGTH) {
  if ($input) JCALL3(ReleaseByteArrayElements, jenv, $input, (jbyte *)$1, JNI_ABORT);
}
%typemap(directorin, descriptor="[B", noblock=1) (const char *STRING, size_t LENGTH) {
  $input = 0;
  if ($1) {
    $input = JCALL1(NewByteArray, jenv, (jsize)$2);
    if (!$input) return $null;
    JCALL4(SetByteArrayRegion, jenv, $input, 0, (jsize)$2, (jbyte *)$1);
  }
  Swig::LocalRefGuard $1_refguard(jenv, $input);
}
%typemap(javadirectorin, descriptor="[B") (const char *STRING, size_t LENGTH) "$jniinput"
%apply (const char *STRING, size_t LENGTH) { (char *STRING, size_t LENGTH) }
/* Enable write-back for non-const version */
%typemap(argout)  (char *STRING, size_t LENGTH) {
  if ($input) JCALL3(ReleaseByteArrayElements, jenv, $input, (jbyte *)$1, 0);
}
%typemap(directorargout, noblock=1) (char *STRING, size_t LENGTH)
{ if ($input && $1) JCALL4(GetByteArrayRegion, jenv, $input, 0, (jsize)$2, (jbyte *)$1); }
%apply (char *STRING, size_t LENGTH) { (char *STRING, int LENGTH) }

/* java keywords */
%include <javakw.swg>

// Default enum handling
%include <enumtypesafe.swg>

