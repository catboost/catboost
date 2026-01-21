#include "jni.h"

#include <iostream>
#include <memory>
#include <string>
#include <thread>

#if defined(__cpp_exceptions) || defined(__EXCEPTIONS__) || defined(_CPPUNWIND)
    #define EXCEPTIONS_ENABLED
#endif

#ifdef EXCEPTIONS_ENABLED
    #define THROW(arg) throw arg
    #define TRY try
    #define CATCH(arg) catch(arg)
#else
    #define THROW(arg) \
        std::cerr << "Exception: " << arg.what() << std::endl; \
        std::abort();
    #define TRY if constexpr (true)
    #define CATCH(arg) else
#endif

namespace {
    void Ensure(bool condition) {
        if (!condition) {
            THROW(std::runtime_error("ensure failed"));
        }
    }
}

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
    TRY {
        if (!Object) {
            return;
        }
        auto* env = Env()->GetJniEnv();
        TRefPolicy::Unref(Object, env);
        Object = nullptr;
    } CATCH (...) {
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

TJniException::TJniException(int error) : Error("code (" + std::to_string(error) + ")") {
}

const char* TJniException::what() const noexcept {
    return Error.data();
}

TJniException& TJniException::append(std::string_view error) {
    Error += error;
    return *this;
}

void RethrowExceptionFromJavaToCpp() {
    auto* env = Env()->GetJniEnv();
    if (env->ExceptionCheck()) {
        auto exc = TLocalRef(env->ExceptionOccurred());
        env->ExceptionClear();
        auto excClass = TLocalClassRef(env->GetObjectClass(exc.Get()));
        jmethodID getMessage = env->GetMethodID(excClass.Get(), "getMessage", "()Ljava/lang/String;");
        auto message = static_cast<jstring>(env->CallObjectMethod(exc.Get(), getMessage));
        std::string exceptionMsg = "<no message>";
        if (message) {
            char const* msg = env->GetStringUTFChars(message, nullptr);
            exceptionMsg = msg;
            env->ReleaseStringUTFChars(message, msg);
        }
        THROW(TJniException().append(exceptionMsg));
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
        Ensure(jvm);
        int ret = Jvm->GetEnv((void**)&JniEnv, JNI_VERSION);
        if (ret == JNI_EDETACHED) {
            JavaVMAttachArgs args;
            std::string nameStr;
            {
                std::stringstream name;
                name << "Native: " << std::this_thread::get_id();
                nameStr = std::move(name).str();
            }
            args.version = JNI_VERSION;
            args.name = (char*)nameStr.data();
            args.group = nullptr;

#if defined(__ANDROID__)
            int err = Jvm->AttachCurrentThread(&JniEnv, &args);
#else
            int err = Jvm->AttachCurrentThread((void**)&JniEnv, &args);
#endif
            if (err != JNI_OK) {
                THROW(TJniException(err).append(": can't attach thread"));
            }
            IsAttached = true;
        } else if (ret != JNI_OK) {
            THROW(TJniException(ret));
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
    Ensure(jvm);

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
    TRY {
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
                std::cerr << "Unknown class loader type: " << static_cast<int64_t>(classLoader);
                return;
            }
        }

        Resources->LoadMethod = GetMethodID(classLoaderClass.Get(), "loadClass", "(Ljava/lang/String;)Ljava/lang/Class;", /*isStatic=*/false);
        Resources->ClassLoader = TSingletonClassRef((jclass)classLoaderRef.Get());
    }
    #ifdef EXCEPTIONS_ENABLED
    catch (const std::exception& exc) {
        std::cerr << "Can't set class loader: " << exc.what();
    } catch (...) {
        std::cerr << "Can't set class loader: unknown error";
    }
    #endif
}

JNIEnv* TJniEnv::GetJniEnv() const {
    thread_local TThreadAttacher attacher(Resources->Jvm);
    return attacher.GetJniEnv();
}

TLocalClassRef TJniEnv::FindClass(std::string_view name) const {
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

jmethodID TJniEnv::GetMethodID(jclass clazz, std::string_view name, std::string_view signature,
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

TLocalStringRef TJniEnv::NewStringUTF(std::string_view str) const {
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
