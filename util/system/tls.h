#pragma once

#include "defaults.h"

#include <util/generic/ptr.h>
#include <util/generic/noncopyable.h>

#include <new>

#if defined(_darwin_)
    #define Y_DISABLE_THRKEY_OPTIMIZATION
#endif

#if defined(_arm_) && defined(_linux_)
    #define Y_DISABLE_THRKEY_OPTIMIZATION
#endif

#if defined(__GNUC__) && defined(__ANDROID__) && defined(__i686__) // https://st.yandex-team.ru/DEVTOOLS-3352
    #define Y_DISABLE_THRKEY_OPTIMIZATION
#endif

/**
    @def Y_THREAD(TType)

    A thread-local wrapper for a given class. Suitable for POD and classes with a constructor with a single argument.

    The wrapper can be treated as the original class in many cases, as it has the same signature for the constructor and an implicit cast to the origianl class.

    Has methods :
        - implicit caster to TType
        - TType& Get()
        - TType* GetPtr()

    Time complexity: getting a variable takes O(number of threads where the variable has been constructed)

    Memory usage: O(number of threads where the variable has been constructed)

    Best practices:
        - storing singletons won't result in heavy memory overheads
        - storing pointers allows complex constructors as well as lazy constructions
        - storing static variables won't result in heavy memory overheads

    Possibly bad practices:
        - field in a class with numerous instances and numerous threads will result in slow working and memory overheads

    Example:
        @code
        // the field declaration in header
        Y_THREAD(TBuffer) TmpBuffer;
        // ...later somewhere in cpp...
        TmpBuffer.Clear();
        for (size_t i = 0; i < sz && TrieCursor[i].second.IsFork(); ++i) {
            TmpBuffer.Append(TrieCursor[i].second.Char);
        }
        @endcode

    Example:
        @code
        // the field decalrataion in header
        Y_THREAD(TMyWriter*) ThreadLocalWriter;
        // ...later somewhere in cpp...
        TMyWriter*& writerRef = ThreadLocalWriter.Get();
        if (writerRef == nullptr) {
            THolder<TMyWriter> threadLocalWriter( new TMyWriter(
                *Session,
                MinLogError,
                MaxRps,
                LogFraction,
                WriteCounters,
                Log));
            writerRef = threadLocalWriter.Get();
        }
        @endcode

    Example:
        @code
        // in header
        namespace TMorph {
            Y_THREAD(ELanguage) ThreadLocalMainLanguage;
        }
        // in cpp
        Y_THREAD(ELanguage) TMorph::ThreadLocalMainLanguage(LANG_RUS);
        @endcode

    Example:
        @code
        Y_THREAD(TScoreCalcer*) ScoreCalcerPtr;
        static TScoreCalcer* GetScoreCalcer(yint maxElemCount) {
            if (ScoreCalcerPtr == 0) {
                ScoreCalcerPtr = new TScoreCalcer();
                ScoreCalcerPtr->Alloc(maxElemCount);
            }
            return ScoreCalcerPtr;
        }
        @endcode

    @param TType POD or a class with a constructor taking 1 argument
**/

/**
    @def Y_STATIC_THREAD(TType)

    Equivalent to "static Y_THREAD(TType)"

    @see Y_THREAD(TType)
**/

/**
    @def Y_POD_THREAD(TType)

    Same interface as Y_THREAD(TType), but TType must be a POD.
    Implemented (based on the compiler) as Y_THREAD(TType) or as native tls.

    @see Y_THREAD(TType)
**/

/**
    @def STATIC_POD_THREAD(TType)

    Equivalent to "static Y_POD_THREAD(TType)"

    @see Y_POD_THREAD(TType)
**/

#define Y_THREAD(T) ::NTls::TValue<T>
#define Y_STATIC_THREAD(T) static Y_THREAD(T)

// gcc and msvc support automatic tls for POD types
#if defined(Y_DISABLE_THRKEY_OPTIMIZATION)
// nothing to do
#elif defined(__clang__)
    #define Y_POD_THREAD(T) thread_local T
    #define Y_POD_STATIC_THREAD(T) static thread_local T
#elif defined(__GNUC__) && !defined(_cygwin_) && !defined(_arm_) && !defined(__IOS_SIMULATOR__)
    #define Y_POD_THREAD(T) __thread T
    #define Y_POD_STATIC_THREAD(T) static __thread T
// msvc doesn't support __declspec(thread) in dlls, loaded manually (via LoadLibrary)
#elif (defined(_MSC_VER) && !defined(_WINDLL)) || defined(_arm_)
    #define Y_POD_THREAD(T) __declspec(thread) T
    #define Y_POD_STATIC_THREAD(T) __declspec(thread) static T
#endif

#if !defined(Y_POD_THREAD) || !defined(Y_POD_STATIC_THREAD)
    #define Y_POD_THREAD(T) Y_THREAD(T)
    #define Y_POD_STATIC_THREAD(T) Y_STATIC_THREAD(T)
#else
    #define Y_HAVE_FAST_POD_TLS
#endif

namespace NPrivate {
    void FillWithTrash(void* ptr, size_t len);
} // namespace NPrivate

namespace NTls {
    using TDtor = void (*)(void*);

    class TKey {
    public:
        TKey(TDtor dtor);
        TKey(TKey&&) noexcept;
        ~TKey();

        void* Get() const;
        void Set(void* ptr) const;

        static void Cleanup() noexcept;

    private:
        class TImpl;
        THolder<TImpl> Impl_;
    };

    struct TCleaner {
        inline ~TCleaner() {
            TKey::Cleanup();
        }
    };

    template <class T>
    class TValue: public TMoveOnly {
        class TConstructor {
        public:
            TConstructor() noexcept = default;

            virtual ~TConstructor() = default;

            virtual T* Construct(void* ptr) const = 0;
        };

        class TDefaultConstructor: public TConstructor {
        public:
            ~TDefaultConstructor() override = default;

            T* Construct(void* ptr) const override {
                // memset(ptr, 0, sizeof(T));
                return ::new (ptr) T();
            }
        };

        template <class T1>
        class TCopyConstructor: public TConstructor {
        public:
            inline TCopyConstructor(const T1& value)
                : Value(value)
            {
            }

            ~TCopyConstructor() override = default;

            T* Construct(void* ptr) const override {
                return ::new (ptr) T(Value);
            }

        private:
            T1 Value;
        };

    public:
        inline TValue()
            : Constructor_(new TDefaultConstructor())
            , Key_(Dtor)
        {
        }

        template <class T1>
        inline TValue(const T1& value)
            : Constructor_(new TCopyConstructor<T1>(value))
            , Key_(Dtor)
        {
        }

        template <class T1>
        inline T& operator=(const T1& val) {
            return Get() = val;
        }

        inline operator const T&() const {
            return Get();
        }

        inline operator T&() {
            return Get();
        }

        inline const T& operator->() const {
            return Get();
        }

        inline T& operator->() {
            return Get();
        }

        inline const T* operator&() const {
            return GetPtr();
        }

        inline T* operator&() {
            return GetPtr();
        }

        inline T& Get() const {
            return *GetPtr();
        }

        inline T* GetPtr() const {
            T* val = static_cast<T*>(Key_.Get());

            if (!val) {
                THolder<void> mem(::operator new(sizeof(T)));
                THolder<T> newval(Constructor_->Construct(mem.Get()));

                Y_UNUSED(mem.Release());
                Key_.Set((void*)newval.Get());
                val = newval.Release();
            }

            return val;
        }

    private:
        static void Dtor(void* ptr) {
            THolder<void> mem(ptr);

            ((T*)ptr)->~T();
            ::NPrivate::FillWithTrash(ptr, sizeof(T));
        }

    private:
        THolder<TConstructor> Constructor_;
        TKey Key_;
    };
} // namespace NTls

template <class T>
static inline T& TlsRef(NTls::TValue<T>& v) noexcept {
    return v;
}

template <class T>
static inline const T& TlsRef(const NTls::TValue<T>& v) noexcept {
    return v;
}

template <class T>
static inline T& TlsRef(T& v) noexcept {
    return v;
}
