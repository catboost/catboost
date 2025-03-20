#include "jni.h"

#include <util/string/cast.h>
#include <util/system/mutex.h>
#include <util/system/thread.h>
#include <util/system/tls.h>

#include <memory>

namespace NJni {

// TObjectRef ///////////////////////////////////////////////////////////////////////////////////

template <typename TRefPolicy, typename TObject>
TObjectRef<TRefPolicy, TObject>::TObjectRef(TObject object) {
    static_assert(std::is_base_of<std::remove_pointer_t<jobject>, std::remove_pointer_t<TObject>>::value);
    auto* env = Env()->GetJniEnv();
    Object = static_cast<TObject>(TRefPolicy::Ref(object, env));
}

// Specialization - don't unref class TSingletonRef's
template<>
TObjectRef<TIntentionallyLeakedRefPolicy, jclass>::~TObjectRef() {
    Object = nullptr;
}

template <typename TRefPolicy, typename TObject>
TObjectRef<TRefPolicy, TObject>::~TObjectRef() {
    try {
        if (!Object) {
            return;
        }
        auto* env = Env()->GetJniEnv();
        TRefPolicy::Unref(Object, env);
        Object = nullptr;
    } catch (...) {
    }
}

template <typename TRefPolicy, typename TObject>
TObjectRef<TRefPolicy, TObject>& TObjectRef<TRefPolicy, TObject>::operator= (TObjectRef&& rhs) noexcept {
    auto* env = Env()->GetJniEnv();
    TRefPolicy::Unref(Object, env);
    Object = rhs.Object;
    rhs.Object = nullptr;
    return *this;
}

template <typename TRefPolicy, typename TObject>
TObject TObjectRef<TRefPolicy, TObject>::Release() {
    auto ret = Object;
    Object = nullptr;
    return ret;
}

// Instantiate.
//
template class TObjectRef<TGlobalRefPolicy, jobject>;
template class TObjectRef<TWeakGlobalRefPolicy, jobject>;
template class TObjectRef<TLocalRefPolicy, jobject>;

template class TObjectRef<TLocalRefPolicy, jclass>;
template class TObjectRef<TGlobalRefPolicy, jclass>;
template class TObjectRef<TIntentionallyLeakedRefPolicy, jclass>;
template class TObjectRef<TLocalRefPolicy, jstring>;
template class TObjectRef<TLocalRefPolicy, jbyteArray>;

// TJniException ////////////////////////////////////////////////////////////////////////////////

TJniException::TJniException(int error) {
    *this << "code (" << error << ")";
}

void RethrowExceptionFromJavaToCpp() {
    auto* env = Env()->GetJniEnv();
    if (env->ExceptionCheck()) {
        auto exc = TLocalRef(env->ExceptionOccurred());
        env->ExceptionClear();
        auto excClass = TLocalClassRef(env->GetObjectClass(exc.Get()));
        jmethodID getMessage = env->GetMethodID(excClass.Get(), "getMessage", "()Ljava/lang/String;");
        auto message = static_cast<jstring>(env->CallObjectMethod(exc.Get(), getMessage));
        TString exceptionMsg;
        if (message) {
            char const* msg = env->GetStringUTFChars(message, nullptr);
            exceptionMsg = msg;
            env->ReleaseStringUTFChars(message, msg);
        }
        ythrow TJniException() << (exceptionMsg ? exceptionMsg : "<no message>");
    }
}

void Check() {
    RethrowExceptionFromJavaToCpp();
}

// TThreadAttacher //////////////////////////////////////////////////////////////////////////////
//
// Used to attach jni-env to thread and detach on thread exit. Should be used only with TLS.

class TThreadAttacher {
public:
    TThreadAttacher(JavaVM* jvm)
        : IsAttached(), Jvm(jvm), JniEnv()
    {
        Y_ENSURE(jvm);
        int ret = Jvm->GetEnv((void**)&JniEnv, JNI_VERSION);
        if (ret == JNI_EDETACHED) {
            JavaVMAttachArgs args;
            TString name = "Native: " + ToString<size_t>(TThread::CurrentThreadId());
            args.version = JNI_VERSION;
            args.name = (char*)name.data();
            args.group = nullptr;

#if defined(__ANDROID__)
            int err = Jvm->AttachCurrentThread(&JniEnv, &args);
#else
            int err = Jvm->AttachCurrentThread((void**)&JniEnv, &args);
#endif
            if (err != JNI_OK) {
                ythrow TJniException(err) << ": can't attach thread";
            }
            IsAttached = true;
        } else if (ret != JNI_OK) {
            ythrow TJniException(ret);
        }
    }

    ~TThreadAttacher() {
        if (IsAttached)
            Jvm->DetachCurrentThread();
    }

    JNIEnv* GetJniEnv() {
        return JniEnv;
    }

private:
    bool IsAttached;
    JavaVM* Jvm;
    JNIEnv* JniEnv;
};

// TJniEnv /////////////////////////////////////////////////////////////////////////////////

struct TJniEnv::TResources {
    JavaVM* Jvm {};
    TSingletonClassRef ClassLoader {};
    jmethodID LoadMethod {};
};

TJniEnv::TJniEnv() : Resources(std::make_unique<TResources>()) {}

TJniEnv* TJniEnv::Get() {
    static TJniEnv env;
    return &env;
}

jint TJniEnv::Init(JavaVM* jvm, EClassLoader classLoader) {
    Y_ENSURE(jvm);

    Resources->Jvm = jvm;
    JNIEnv* env = nullptr;
    if (Resources->Jvm->GetEnv((void**)(&env), JNI_VERSION) != JNI_OK)
        return JNI_ERR;

    TryToSetClassLoader(classLoader);

    return JNI_VERSION;
}

void TJniEnv::Cleanup(JavaVM*) {
    Resources.reset();
}

void TJniEnv::TryToSetClassLoader(EClassLoader classLoader) {
    try {
        auto classLoaderClass = FindClass("java/lang/ClassLoader");

        NJni::TLocalRef classLoaderRef;
        switch (classLoader) {
            case EClassLoader::CONTEXT: {
                auto threadClassRef = FindClass("java/lang/Thread");
                auto currentThreadMethod = GetMethodID(threadClassRef.Get(), "currentThread", "()Ljava/lang/Thread;", /*isStatic=*/true);
                auto currentThreadRef = CallStaticObjectMethod(threadClassRef.Get(), currentThreadMethod);
                auto getContextClassLoaderMethod = GetMethodID(threadClassRef.Get(), "getContextClassLoader", "()Ljava/lang/ClassLoader;", /*isStatic=*/false);
                classLoaderRef = CallObjectMethod(currentThreadRef.Get(), getContextClassLoaderMethod);
                break;
            }
            case EClassLoader::NORMAL: {
                auto classClass = FindClass("java/lang/Class");
                auto getClassLoaderMethod = GetMethodID(classClass.Get(), "getClassLoader", "()Ljava/lang/ClassLoader;", /*isStatic=*/false);
                classLoaderRef = CallObjectMethod(classLoaderClass.Get(), getClassLoaderMethod);
                break;
            }
            default: {
                Cerr << "Unknown class loader type: " << ToString(classLoader);
                return;
            }
        }

        Resources->LoadMethod = GetMethodID(classLoaderClass.Get(), "loadClass", "(Ljava/lang/String;)Ljava/lang/Class;", /*isStatic=*/false);
        Resources->ClassLoader = TSingletonClassRef((jclass)classLoaderRef.Get());
    } catch (...) {
        Cerr << "Can't set class loader: " << CurrentExceptionMessage();
    }
}

JNIEnv* TJniEnv::GetJniEnv() const {
    Y_STATIC_THREAD(TThreadAttacher) attacher(Resources->Jvm);
    return attacher.Get().GetJniEnv();
}

TLocalClassRef TJniEnv::FindClass(TStringBuf name) const {
    TLocalClassRef localRef;
    if (Resources->ClassLoader && Resources->LoadMethod) {
        auto jname = NewStringUTF(name);
        // To cast to TClassRef, we should firstly release TLocalRef.
        //
        auto clazz = static_cast<jclass>(CallObjectMethod(Resources->ClassLoader.Get(),
            Resources->LoadMethod, jname.Get()).Release());
        localRef = TLocalClassRef(clazz);
    } else {
        // May be system class loader, that doesn't know anything. (it depends on execution thread).
        //
        auto* env = GetJniEnv();
        localRef = TLocalClassRef(env->FindClass(name.data()));
    }
    Check();
    return localRef;
}

jmethodID TJniEnv::GetMethodID(jclass clazz, TStringBuf name, TStringBuf signature,
    bool isStatic) const
{
    auto* env = GetJniEnv();
    jmethodID methodID = nullptr;
    if (isStatic) {
        methodID = env->GetStaticMethodID(clazz, name.data(), signature.data());
    } else {
        methodID = env->GetMethodID(clazz, name.data(), signature.data());
    }
    Check();
    return methodID;
}

TLocalRef TJniEnv::CallStaticObjectMethod(jclass clazz, jmethodID methodId, ...) const {
    va_list args;
    auto* env = GetJniEnv();
    va_start(args, methodId);
    auto ret = TLocalRef(env->CallStaticObjectMethodV(clazz, methodId, args));
    va_end(args);
    Check();
    return ret;
}

TLocalRef TJniEnv::CallObjectMethod(jobject object, jmethodID methodId, ...) const {
    va_list args;
    auto* env = GetJniEnv();
    va_start(args, methodId);
    auto ret = TLocalRef(env->CallObjectMethodV(object, methodId, args));
    va_end(args);
    Check();
    return ret;
}

jint TJniEnv::CallIntMethod(jobject object, jmethodID methodId, ...) const {
    va_list args;
    auto* env = GetJniEnv();
    va_start(args, methodId);
    auto ret = jint(env->CallIntMethodV(object, methodId, args));
    va_end(args);
    Check();
    return ret;
}

jboolean TJniEnv::CallBooleanMethod(jobject object, jmethodID methodId, ...) const {
    va_list args;
    auto* env = GetJniEnv();
    va_start(args, methodId);
    auto ret = jint(env->CallBooleanMethodV(object, methodId, args));
    va_end(args);
    Check();
    return ret;
}

void TJniEnv::CallVoidMethod(jobject object, jmethodID methodId, ...) const {
    va_list args;
    auto* env = GetJniEnv();
    va_start(args, methodId);
    env->CallVoidMethodV(object, methodId, args);
    va_end(args);
    Check();
}

TLocalArrayRef TJniEnv::NewByteArray(jsize len) const {
    auto* env = GetJniEnv();
    auto ret = TLocalArrayRef(env->NewByteArray(len));
    Check();
    return ret;
}

void TJniEnv::SetByteArrayRegion(jbyteArray array, jsize start, jsize len,
    const char* buf) const
{
    auto* env = GetJniEnv();
    env->SetByteArrayRegion(array, start, len, (const jbyte*)buf);
    Check();
}

void TJniEnv::GetByteArrayRegion(jbyteArray array, jsize start, jsize len,
    char* buf) const
{
    auto* env = GetJniEnv();
    env->GetByteArrayRegion(array, start, len, (jbyte*)buf);
    Check();
}

jsize TJniEnv::GetArrayLength(jarray array) const {
    return GetJniEnv()->GetArrayLength(array);
}

TLocalStringRef TJniEnv::NewStringUTF(TStringBuf str) const {
    auto* env = GetJniEnv();
    auto ret = TLocalStringRef(env->NewStringUTF(str.data()));
    Check();
    return ret;
}

const char* TJniEnv::GetStringUTFChars(jstring str, jboolean* isCopy) const {
    return GetJniEnv()->GetStringUTFChars(str, isCopy);
}

bool TJniEnv::acquireLocalRef(const NJni::TWeakGlobalRef& weakRef, NJni::TLocalRef& output) const {
    auto* env = GetJniEnv();
    jobject localRefJObject = env->NewLocalRef(weakRef.Get());
    if (!localRefJObject) {
        return false;
    }
    output = NJni::TLocalRef(localRefJObject);
    return true;
}

TJniEnv* Env() {
    return TJniEnv::Get();
}

}  // namespace NJni
