%{
#include <util/system/compiler.h>
#include <util/system/tls.h>
#include <stdexcept>
#include "jni.h"


static JavaVM* CachedJvm = 0;

extern "C" {

    JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *jvm, void *reserved) {
        Y_UNUSED(reserved);

        CachedJvm = jvm;
        return JNI_VERSION_1_8;
    }

}

struct TPerThreadJNIEnv {
public:
    JNIEnv* JniEnv = nullptr;
    bool    Detach = false;

public:
    ~TPerThreadJNIEnv() {
        if (Detach) {
            CachedJvm->DetachCurrentThread();
        }
    }
};


static JNIEnv* GetPerThreadJniEnv() {
    Y_STATIC_THREAD(TPerThreadJNIEnv) ThreadJniEnv;
    if (!ThreadJniEnv.Get().JniEnv) {
        jint rc = CachedJvm->GetEnv((void **)&ThreadJniEnv.Get().JniEnv, JNI_VERSION_1_8);
        if (rc == JNI_EDETACHED) {
            rc = CachedJvm->AttachCurrentThread((void **)&ThreadJniEnv.Get().JniEnv, NULL);
            if (rc != JNI_OK) {
                throw std::runtime_error("JNI: AttachCurrentThread error");
            }
            ThreadJniEnv.Get().Detach = true;
        }
        if (rc == JNI_EVERSION) {
            throw std::runtime_error("jni version not supported");
        }
    }
    return ThreadJniEnv.Get().JniEnv;
}

%}
