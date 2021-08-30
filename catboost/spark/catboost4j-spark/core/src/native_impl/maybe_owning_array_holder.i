%{
#include <catboost/libs/helpers/maybe_owning_array_holder.h>
#include <stdexcept>

#include <catboost/spark/catboost4j-spark/core/src/native_impl/jni_helpers.h>
%}

%include "jni_context.i"
%include "primitive_arrays.i"

namespace NCB {

    template <class T>
    class TMaybeOwningConstArrayHolder;

}


%define PRIMITIVE_TYPE_MAYBE_OWNING_ARRAY_HOLDER_IMPL(CPPTYPE, JNITYPE, JAVAPRIMITIVETYPE, JAVABOXEDTYPE)


%{

struct TJava##JAVABOXEDTYPE##ArrayHolder : public NCB::IResourceHolder {
    CPPTYPE* CppPtr;
    JNITYPE##Array JniArrayGlobalRef;

public:
    TJava##JAVABOXEDTYPE##ArrayHolder(JNIEnv* jEnv, CPPTYPE* cppPtr, JNITYPE##Array jniArray)
        : CppPtr(cppPtr)
        , JniArrayGlobalRef((JNITYPE##Array)jEnv->NewGlobalRef(jniArray))
    {
        if (!JniArrayGlobalRef) {
            throw std::runtime_error("JVM: NewGlobalRef failed. Out of memory");
        }
    }

    ~TJava##JAVABOXEDTYPE##ArrayHolder() {
        JNIEnv* jEnv = GetPerThreadJniEnv();
        jEnv->Release##JAVABOXEDTYPE##ArrayElements(JniArrayGlobalRef, ToJniPtr<JNITYPE>(CppPtr), JNI_ABORT);
        jEnv->DeleteGlobalRef((jobject)JniArrayGlobalRef);
    }
};

%}

%typemap(jni) NCB::TMaybeOwningConstArrayHolder<CPPTYPE>, const NCB::TMaybeOwningConstArrayHolder<CPPTYPE>& %{JNITYPE##Array%}
%typemap(jtype) NCB::TMaybeOwningConstArrayHolder<CPPTYPE>, const NCB::TMaybeOwningConstArrayHolder<CPPTYPE>& %{JAVAPRIMITIVETYPE[]%}
%typemap(jstype) NCB::TMaybeOwningConstArrayHolder<CPPTYPE>, const NCB::TMaybeOwningConstArrayHolder<CPPTYPE>& %{JAVAPRIMITIVETYPE[]%}

%typemap(javain) NCB::TMaybeOwningConstArrayHolder<CPPTYPE>, const NCB::TMaybeOwningConstArrayHolder<CPPTYPE>& "$javainput"
%typemap(javaout) NCB::TMaybeOwningConstArrayHolder<CPPTYPE>, const NCB::TMaybeOwningConstArrayHolder<CPPTYPE>& {
    return $jnicall;
}

%typemap(in) NCB::TMaybeOwningConstArrayHolder<CPPTYPE>
%{
    TConstArrayRef<CPPTYPE> data;
    if (!SWIG_PrimitiveArrayJavaToCpp(jenv, $input, &data)) {
        return $null;
    }

    $1 = NCB::TMaybeOwningConstArrayHolder<CPPTYPE>::CreateOwning(
        data,
        MakeIntrusive<TJava##JAVABOXEDTYPE##ArrayHolder>(jenv, const_cast<CPPTYPE*>(data.data()), $input)
    );
%}

%typemap(out) NCB::TMaybeOwningConstArrayHolder<CPPTYPE>
%{
    $result = SWIG_PrimitiveArrayCppToJava(jenv, $1.data(), $1.GetSize());
%}

%typemap(out) const NCB::TMaybeOwningConstArrayHolder<CPPTYPE>&
%{
    $result = SWIG_PrimitiveArrayCppToJava(jenv, $1->data(), $1->GetSize());
%}

%typemap(in) const NCB::TMaybeOwningConstArrayHolder<CPPTYPE>& (NCB::TMaybeOwningConstArrayHolder<CPPTYPE> temp)
%{
    TConstArrayRef<CPPTYPE> data;
    if (!SWIG_PrimitiveArrayJavaToCpp(jenv, $input, &data)) {
        return $null;
    }

    temp = NCB::TMaybeOwningConstArrayHolder<CPPTYPE>::CreateOwning(
        data,
        MakeIntrusive<TJava##JAVABOXEDTYPE##ArrayHolder>(jenv, const_cast<CPPTYPE*>(data.data()), $input)
    );
    $1 = &temp;
%}

%enddef

PRIMITIVE_TYPE_MAYBE_OWNING_ARRAY_HOLDER_IMPL(i8, jbyte, byte, Byte);
PRIMITIVE_TYPE_MAYBE_OWNING_ARRAY_HOLDER_IMPL(ui16, jchar, char, Char);
PRIMITIVE_TYPE_MAYBE_OWNING_ARRAY_HOLDER_IMPL(i16, jshort, short, Short);
PRIMITIVE_TYPE_MAYBE_OWNING_ARRAY_HOLDER_IMPL(i32, jint, int, Int);
//PRIMITIVE_TYPE_ARRAY_IMPL(i64, jlong, long, Long);
PRIMITIVE_TYPE_MAYBE_OWNING_ARRAY_HOLDER_IMPL(float, jfloat, float, Float);
PRIMITIVE_TYPE_MAYBE_OWNING_ARRAY_HOLDER_IMPL(double, jdouble, double, Double);
