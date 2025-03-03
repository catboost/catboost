/* -----------------------------------------------------------------------------
 * director.swg
 *
 * This file contains support for director classes so that Java proxy
 * methods can be called from C++.
 * ----------------------------------------------------------------------------- */

#if defined(DEBUG_DIRECTOR_OWNED) || defined(DEBUG_DIRECTOR_EXCEPTION) || defined(DEBUG_DIRECTOR_THREAD_NAME)
#include <iostream>
#endif

#include <exception>

#if defined(SWIG_JAVA_USE_THREAD_NAME)

#if !defined(SWIG_JAVA_GET_THREAD_NAME)
namespace Swig {
  SWIGINTERN int GetThreadName(char *name, size_t len);
}

#if defined(__linux__)

#include <sys/prctl.h>
SWIGINTERN int Swig::GetThreadName(char *name, size_t len) {
  (void)len;
#if defined(PR_GET_NAME)
  return prctl(PR_GET_NAME, (unsigned long)name, 0, 0, 0);
#else
  (void)name;
  return 1;
#endif
}

#elif defined(__unix__) || defined(__APPLE__)

#include <pthread.h>
SWIGINTERN int Swig::GetThreadName(char *name, size_t len) {
  return pthread_getname_np(pthread_self(), name, len);
}

#else

SWIGINTERN int Swig::GetThreadName(char *name, size_t len) {
  (void)len;
  (void)name;
  return 1;
}
#endif

#endif

#endif

namespace Swig {

  /* Java object wrapper */
  class JObjectWrapper {
  public:
    JObjectWrapper() : jthis_(NULL), weak_global_(true) {
    }

    ~JObjectWrapper() {
      jthis_ = NULL;
      weak_global_ = true;
    }

    bool set(JNIEnv *jenv, jobject jobj, bool mem_own, bool weak_global) {
      if (!jthis_) {
        weak_global_ = weak_global || !mem_own; // hold as weak global if explicitly requested or not owned
        if (jobj)
          jthis_ = weak_global_ ? jenv->NewWeakGlobalRef(jobj) : jenv->NewGlobalRef(jobj);
#if defined(DEBUG_DIRECTOR_OWNED)
        std::cout << "JObjectWrapper::set(" << jobj << ", " << (weak_global ? "weak_global" : "global_ref") << ") -> " << jthis_ << std::endl;
#endif
        return true;
      } else {
#if defined(DEBUG_DIRECTOR_OWNED)
        std::cout << "JObjectWrapper::set(" << jobj << ", " << (weak_global ? "weak_global" : "global_ref") << ") -> already set" << std::endl;
#endif
        return false;
      }
    }

    jobject get(JNIEnv *jenv) const {
#if defined(DEBUG_DIRECTOR_OWNED)
      std::cout << "JObjectWrapper::get(";
      if (jthis_)
        std::cout << jthis_;
      else
        std::cout << "null";
      std::cout << ") -> return new local ref" << std::endl;
#endif
      return (jthis_ ? jenv->NewLocalRef(jthis_) : jthis_);
    }

    void release(JNIEnv *jenv) {
#if defined(DEBUG_DIRECTOR_OWNED)
      std::cout << "JObjectWrapper::release(" << jthis_ << "): " << (weak_global_ ? "weak global ref" : "global ref") << std::endl;
#endif
      if (jthis_) {
        if (weak_global_) {
          if (jenv->IsSameObject(jthis_, NULL) == JNI_FALSE)
            jenv->DeleteWeakGlobalRef((jweak)jthis_);
        } else
          jenv->DeleteGlobalRef(jthis_);
      }

      jthis_ = NULL;
      weak_global_ = true;
    }

    /* Only call peek if you know what you are doing wrt to weak/global references */
    jobject peek() {
      return jthis_;
    }

    /* Java proxy releases ownership of C++ object, C++ object is now
       responsible for destruction (creates NewGlobalRef to pin Java proxy) */
    void java_change_ownership(JNIEnv *jenv, jobject jself, bool take_or_release) {
      if (take_or_release) {  /* Java takes ownership of C++ object's lifetime. */
        if (!weak_global_) {
          jenv->DeleteGlobalRef(jthis_);
          jthis_ = jenv->NewWeakGlobalRef(jself);
          weak_global_ = true;
        }
      } else {
	/* Java releases ownership of C++ object's lifetime */
        if (weak_global_) {
          jenv->DeleteWeakGlobalRef((jweak)jthis_);
          jthis_ = jenv->NewGlobalRef(jself);
          weak_global_ = false;
        }
      }
    }

  private:
    /* pointer to Java object */
    jobject jthis_;
    /* Local or global reference flag */
    bool weak_global_;
  };

  /* Local JNI reference deleter */
  class LocalRefGuard {
    JNIEnv *jenv_;
    jobject jobj_;

    // non-copyable
    LocalRefGuard(const LocalRefGuard &);
    LocalRefGuard &operator=(const LocalRefGuard &);
  public:
    LocalRefGuard(JNIEnv *jenv, jobject jobj): jenv_(jenv), jobj_(jobj) {}
    ~LocalRefGuard() {
      if (jobj_)
        jenv_->DeleteLocalRef(jobj_);
    }
  };

  /* director base class */
  class Director {
    /* pointer to Java virtual machine */
    JavaVM *swig_jvm_;

  protected:
#if defined (_MSC_VER) && (_MSC_VER<1300)
    class JNIEnvWrapper;
    friend class JNIEnvWrapper;
#endif
    /* Utility class for managing the JNI environment */
    class JNIEnvWrapper {
      const Director *director_;
      JNIEnv *jenv_;
      int env_status;
    public:
      JNIEnvWrapper(const Director *director) : director_(director), jenv_(0), env_status(0) {
#if defined(__ANDROID__)
        JNIEnv **jenv = &jenv_;
#else
        void **jenv = (void **)&jenv_;
#endif
        env_status = director_->swig_jvm_->GetEnv((void **)&jenv_, JNI_VERSION_1_2);
        JavaVMAttachArgs args;
        args.version = JNI_VERSION_1_2;
        args.group = NULL;
        args.name = NULL;
#if defined(SWIG_JAVA_USE_THREAD_NAME)
        char thread_name[64];  // MAX_TASK_COMM_LEN=16 is hard-coded in the Linux kernel and MacOS has MAXTHREADNAMESIZE=64.
        if (Swig::GetThreadName(thread_name, sizeof(thread_name)) == 0) {
          args.name = thread_name;
#if defined(DEBUG_DIRECTOR_THREAD_NAME)
          std::cout << "JNIEnvWrapper: thread name: " << thread_name << std::endl;
        } else {
          std::cout << "JNIEnvWrapper: Couldn't set Java thread name" << std::endl;
#endif
        }
#endif
#if defined(SWIG_JAVA_ATTACH_CURRENT_THREAD_AS_DAEMON)
        // Attach a daemon thread to the JVM. Useful when the JVM should not wait for
        // the thread to exit upon shutdown. Only for jdk-1.4 and later.
        director_->swig_jvm_->AttachCurrentThreadAsDaemon(jenv, &args);
#else
        director_->swig_jvm_->AttachCurrentThread(jenv, &args);
#endif
      }
      ~JNIEnvWrapper() {
#if !defined(SWIG_JAVA_NO_DETACH_CURRENT_THREAD)
        // Some JVMs, eg jdk-1.4.2 and lower on Solaris have a bug and crash with the DetachCurrentThread call.
        // However, without this call, the JVM hangs on exit when the thread was not created by the JVM and creates a memory leak.
        if (env_status == JNI_EDETACHED)
          director_->swig_jvm_->DetachCurrentThread();
#endif
      }
      JNIEnv *getJNIEnv() const {
        return jenv_;
      }
    };

    /* Java object wrapper */
    JObjectWrapper swig_self_;

    /* Disconnect director from Java object */
    void swig_disconnect_director_self(const char *disconn_method) {
      JNIEnvWrapper jnienv(this) ;
      JNIEnv *jenv = jnienv.getJNIEnv() ;
      jobject jobj = swig_self_.get(jenv);
      LocalRefGuard ref_deleter(jenv, jobj);
#if defined(DEBUG_DIRECTOR_OWNED)
      std::cout << "Swig::Director::disconnect_director_self(" << jobj << ")" << std::endl;
#endif
      if (jobj && jenv->IsSameObject(jobj, NULL) == JNI_FALSE) {
        jmethodID disconn_meth = jenv->GetMethodID(jenv->GetObjectClass(jobj), disconn_method, "()V");
        if (disconn_meth) {
#if defined(DEBUG_DIRECTOR_OWNED)
          std::cout << "Swig::Director::disconnect_director_self upcall to " << disconn_method << std::endl;
#endif
          jenv->CallVoidMethod(jobj, disconn_meth);
        }
      }
    }

  public:
    Director(JNIEnv *jenv) : swig_jvm_((JavaVM *) NULL), swig_self_() {
      /* Acquire the Java VM pointer */
      jenv->GetJavaVM(&swig_jvm_);
    }

    virtual ~Director() {
      JNIEnvWrapper jnienv(this) ;
      JNIEnv *jenv = jnienv.getJNIEnv() ;
      swig_self_.release(jenv);
    }

    bool swig_set_self(JNIEnv *jenv, jobject jself, bool mem_own, bool weak_global) {
      return swig_self_.set(jenv, jself, mem_own, weak_global);
    }

    jobject swig_get_self(JNIEnv *jenv) const {
      return swig_self_.get(jenv);
    }

    // Change C++ object's ownership, relative to Java
    void swig_java_change_ownership(JNIEnv *jenv, jobject jself, bool take_or_release) {
      swig_self_.java_change_ownership(jenv, jself, take_or_release);
    }
  };

  // Zero initialized bool array
  template<size_t N> class BoolArray {
    bool array_[N];
  public:
    BoolArray() {
      memset(array_, 0, sizeof(array_));
    }
    bool& operator[](size_t n) {
      return array_[n];
    }
    bool operator[](size_t n) const {
      return array_[n];
    }
  };

  // Utility classes and functions for exception handling.

  // Simple holder for a Java string during exception handling, providing access to a c-style string
  class JavaString {
  public:
    JavaString(JNIEnv *jenv, jstring jstr) : jenv_(jenv), jstr_(jstr), cstr_(0) {
      if (jenv_ && jstr_)
	cstr_ = (const char *) jenv_->GetStringUTFChars(jstr_, NULL);
    }

    ~JavaString() {
      if (jenv_ && jstr_ && cstr_)
	jenv_->ReleaseStringUTFChars(jstr_, cstr_);
    }

    const char *c_str(const char *null_string = "null JavaString") const {
      return cstr_ ? cstr_ : null_string;
    }

  private:
    // non-copyable
    JavaString(const JavaString &);
    JavaString &operator=(const JavaString &);

    JNIEnv *jenv_;
    jstring jstr_;
    const char *cstr_;
  };

  // Helper class to extract the exception message from a Java throwable
  class JavaExceptionMessage {
  public:
    JavaExceptionMessage(JNIEnv *jenv, jthrowable throwable) : message_(jenv, exceptionMessageFromThrowable(jenv, throwable)) {
    }

    // Return a C string of the exception message in the jthrowable passed in the constructor
    // If no message is available, null_string is return instead
    const char *message(const char *null_string = "Could not get exception message in JavaExceptionMessage") const {
      return message_.c_str(null_string);
    }

  private:
    // non-copyable
    JavaExceptionMessage(const JavaExceptionMessage &);
    JavaExceptionMessage &operator=(const JavaExceptionMessage &);

    // Get exception message by calling Java method Throwable.getMessage()
    static jstring exceptionMessageFromThrowable(JNIEnv *jenv, jthrowable throwable) {
      jstring jmsg = NULL;
      if (jenv && throwable) {
	jenv->ExceptionClear(); // Cannot invoke methods with any pending exceptions
	jclass throwclz = jenv->GetObjectClass(throwable);
	if (throwclz) {
	  // All Throwable classes have a getMessage() method, so call it to extract the exception message
	  jmethodID getMessageMethodID = jenv->GetMethodID(throwclz, "getMessage", "()Ljava/lang/String;");
	  if (getMessageMethodID)
	    jmsg = (jstring)jenv->CallObjectMethod(throwable, getMessageMethodID);
	}
	if (jmsg == NULL && jenv->ExceptionCheck())
	  jenv->ExceptionClear();
      }
      return jmsg;
    }

    JavaString message_;
  };

  // C++ Exception class for handling Java exceptions thrown during a director method Java upcall
  class DirectorException : public std::exception {
  public:

    // Construct exception from a Java throwable
    DirectorException(JNIEnv *jenv, jthrowable throwable) : jenv_(jenv), throwable_(throwable), classname_(0), msg_(0) {

      // Call Java method Object.getClass().getName() to obtain the throwable's class name (delimited by '/')
      if (jenv && throwable) {
	jenv->ExceptionClear(); // Cannot invoke methods with any pending exceptions
	jclass throwclz = jenv->GetObjectClass(throwable);
	if (throwclz) {
	  jclass clzclz = jenv->GetObjectClass(throwclz);
	  if (clzclz) {
	    jmethodID getNameMethodID = jenv->GetMethodID(clzclz, "getName", "()Ljava/lang/String;");
	    if (getNameMethodID) {
	      jstring jstr_classname = (jstring)(jenv->CallObjectMethod(throwclz, getNameMethodID));
              // Copy strings, since there is no guarantee that jenv will be active when handled
              if (jstr_classname) {
                JavaString jsclassname(jenv, jstr_classname);
                const char *classname = jsclassname.c_str(0);
                if (classname)
                  classname_ = copypath(classname);
              }
	    }
	  }
	}
      }

      JavaExceptionMessage exceptionmsg(jenv, throwable);
      msg_ = copystr(exceptionmsg.message(0));
    }

    // More general constructor for handling as a java.lang.RuntimeException
    DirectorException(const char *msg) : jenv_(0), throwable_(0), classname_(0), msg_(msg ? copystr(msg) : 0) {
    }

    ~DirectorException() throw() {
      delete[] classname_;
      delete[] msg_;
    }

    const char *what() const throw() {
      return msg_ ? msg_ : "Unspecified DirectorException message";
    }

    // Reconstruct and raise/throw the Java Exception that caused the DirectorException
    // Note that any error in the JNI exception handling results in a Java RuntimeException
    void throwException(JNIEnv *jenv) const {
      if (jenv) {
        if (jenv == jenv_ && throwable_) {
          // Throw original exception if not already pending
          jthrowable throwable = jenv->ExceptionOccurred();
          if (throwable && jenv->IsSameObject(throwable, throwable_) == JNI_FALSE) {
            jenv->ExceptionClear();
            throwable = 0;
          }
          if (!throwable)
            jenv->Throw(throwable_);
        } else {
          // Try and reconstruct original exception, but original stacktrace is not reconstructed
          jenv->ExceptionClear();

          jmethodID ctorMethodID = 0;
          jclass throwableclass = 0;
          if (classname_) {
            throwableclass = jenv->FindClass(classname_);
            if (throwableclass)
              ctorMethodID = jenv->GetMethodID(throwableclass, "<init>", "(Ljava/lang/String;)V");
          }

          if (ctorMethodID) {
            jenv->ThrowNew(throwableclass, what());
          } else {
            SWIG_JavaThrowException(jenv, SWIG_JavaRuntimeException, what());
          }
        }
      }
    }

    // Deprecated - use throwException
    void raiseJavaException(JNIEnv *jenv) const {
      throwException(jenv);
    }

    // Create and throw the DirectorException
    static void raise(JNIEnv *jenv, jthrowable throwable) {
      throw DirectorException(jenv, throwable);
    }

  private:
    static char *copypath(const char *srcmsg) {
      char *target = copystr(srcmsg);
      for (char *c=target; *c; ++c) {
        if ('.' == *c)
          *c = '/';
      }
      return target;
    }

    static char *copystr(const char *srcmsg) {
      char *target = 0;
      if (srcmsg) {
	size_t msglen = strlen(srcmsg) + 1;
	target = new char[msglen];
	strncpy(target, srcmsg, msglen);
      }
      return target;
    }

    JNIEnv *jenv_;
    jthrowable throwable_;
    const char *classname_;
    const char *msg_;
  };

  // Helper method to determine if a Java throwable matches a particular Java class type
  // Note side effect of clearing any pending exceptions
  SWIGINTERN bool ExceptionMatches(JNIEnv *jenv, jthrowable throwable, const char *classname) {
    bool matches = false;

    if (throwable && jenv && classname) {
      // Exceptions need to be cleared for correct behavior.
      // The caller of ExceptionMatches should restore pending exceptions if desired -
      // the caller already has the throwable.
      jenv->ExceptionClear();

      jclass clz = jenv->FindClass(classname);
      if (clz) {
	jclass classclz = jenv->GetObjectClass(clz);
	jmethodID isInstanceMethodID = jenv->GetMethodID(classclz, "isInstance", "(Ljava/lang/Object;)Z");
	if (isInstanceMethodID) {
	  matches = jenv->CallBooleanMethod(clz, isInstanceMethodID, throwable) != 0;
	}
      }

#if defined(DEBUG_DIRECTOR_EXCEPTION)
      if (jenv->ExceptionCheck()) {
        // Typically occurs when an invalid classname argument is passed resulting in a ClassNotFoundException
        JavaExceptionMessage exc(jenv, jenv->ExceptionOccurred());
        std::cout << "Error: ExceptionMatches: class '" << classname << "' : " << exc.message() << std::endl;
      }
#endif
    }
    return matches;
  }
}

