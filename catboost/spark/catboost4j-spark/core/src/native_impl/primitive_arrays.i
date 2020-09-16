
%{
#include <util/generic/array_ref.h>
#include <util/generic/fwd.h>
#include <stdexcept>

#include <catboost/spark/catboost4j-spark/core/src/native_impl/jni_helpers.h>
%}

%include <std_except.i>


%define PRIMITIVE_TYPE_ARRAY_IMPL(CPPTYPE, JNITYPE, JAVAPRIMITIVETYPE, JAVABOXEDTYPE)
%{
    // returns true if successful, false otherwise

    // generic because should be applicable to both TArrayRef and TConstArrayRef types
    template <class T>
    static bool SWIG_PrimitiveArrayJavaToCpp(
        JNIEnv *jenv,
        JNITYPE##Array input,
        TArrayRef<T>* output
    ) {
        if (!input) {
            SWIG_JavaThrowException(jenv, SWIG_JavaNullPointerException, "null array");
            return false;
        }
        jsize size = jenv->GetArrayLength(input);
        JNITYPE* jarr = jenv->Get##JAVABOXEDTYPE##ArrayElements(input, 0);
        if (!jarr) {
            SWIG_JavaThrowException(
                jenv,
                SWIG_JavaRuntimeException,
                "Get" ## #JAVABOXEDTYPE ## "ArrayElements failed"
            );
            return false;
        }

        *output = TArrayRef<T>((T*)jarr, (size_t)size);
        return true;
    }

    static JNITYPE##Array SWIG_PrimitiveArrayCppToJava(JNIEnv* jenv, const CPPTYPE* data, size_t size) {
        if (size > Max<jsize>()) {
            throw std::length_error("Array size is too big for JVM");
        }

        JNITYPE##Array result = jenv->New##JAVABOXEDTYPE##Array((jsize)size);
        if (!result) {
            throw std::runtime_error("Cannot construct Java array");
        }
        jenv->Set##JAVABOXEDTYPE##ArrayRegion(result, 0, (jsize)size, ToJniPtr<JNITYPE>(data));

        return result;
    }
%}

%typemap(jni) TConstArrayRef<CPPTYPE> %{JNITYPE##Array%}
%typemap(jtype) TConstArrayRef<CPPTYPE> %{JAVAPRIMITIVETYPE[]%}
%typemap(jstype) TConstArrayRef<CPPTYPE> %{JAVAPRIMITIVETYPE[]%}

%typemap(javain) TConstArrayRef<CPPTYPE> "$javainput"

%typemap(in) TConstArrayRef<CPPTYPE>
%{  if (!SWIG_PrimitiveArrayJavaToCpp(jenv, $input, (TConstArrayRef<CPPTYPE>*)&$1)) return $null; %}

/* Release memory */
%typemap(freearg) TConstArrayRef<CPPTYPE> {
    jenv->Release##JAVABOXEDTYPE##ArrayElements(
        $input,
        ToJniPtr<JNITYPE>($1.data()),
        JNI_ABORT
    );
}


%typemap(jni) TArrayRef<CPPTYPE> %{JNITYPE##Array%}
%typemap(jtype) TArrayRef<CPPTYPE> %{JAVAPRIMITIVETYPE[]%}
%typemap(jstype) TArrayRef<CPPTYPE> %{JAVAPRIMITIVETYPE[]%}

%typemap(javain) TArrayRef<CPPTYPE> "$javainput"

%typemap(in) TArrayRef<CPPTYPE>
%{  if (!SWIG_PrimitiveArrayJavaToCpp(jenv, $input, (TArrayRef<CPPTYPE>*)&$1)) return $null; %}

%typemap(freearg) TArrayRef<CPPTYPE> ""

%typemap(argout) TArrayRef<CPPTYPE> {
    jenv->Release##JAVABOXEDTYPE##ArrayElements($input, ToJniPtr<JNITYPE>($1.data()), 0);
}


%enddef

PRIMITIVE_TYPE_ARRAY_IMPL(i8, jbyte, byte, Byte);
PRIMITIVE_TYPE_ARRAY_IMPL(ui16, jchar, char, Char);
PRIMITIVE_TYPE_ARRAY_IMPL(i16, jshort, short, Short);
PRIMITIVE_TYPE_ARRAY_IMPL(i32, jint, int, Int);
PRIMITIVE_TYPE_ARRAY_IMPL(i64, jlong, long, Long);
PRIMITIVE_TYPE_ARRAY_IMPL(float, jfloat, float, Float);
PRIMITIVE_TYPE_ARRAY_IMPL(double, jdouble, double, Double);

