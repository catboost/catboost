#include "mutex.h"

#include <util/generic/yexception.h>
#include <errno.h>

#if defined(_win_)
    #include "winint.h"
#else
    #include <pthread.h>
#endif

class TMutex::TImpl {
public:
    inline TImpl() {
#if defined(_win_)
        InitializeCriticalSection(&Obj);
#else
        struct T {
            pthread_mutexattr_t Attr;

            inline T() {
                int result;

                memset(&Attr, 0, sizeof(Attr));
                result = pthread_mutexattr_init(&Attr);
                if (result != 0) {
                    ythrow yexception() << "mutexattr init failed(" << LastSystemErrorText(result) << ")";
                }

                result = pthread_mutexattr_settype(&Attr, PTHREAD_MUTEX_RECURSIVE);
                if (result != 0) {
                    ythrow yexception() << "mutexattr set type failed(" << LastSystemErrorText(result) << ")";
                }
            }

            inline ~T() {
                int result = pthread_mutexattr_destroy(&Attr);
                Y_ABORT_UNLESS(result == 0, "mutexattr destroy(%s)", LastSystemErrorText(result));
            }
        } pma;

        int result = pthread_mutex_init(&Obj, &pma.Attr);
        if (result != 0) {
            ythrow yexception() << "mutex init failed(" << LastSystemErrorText(result) << ")";
        }
#endif
    }

    inline ~TImpl() {
#if defined(_win_)
        DeleteCriticalSection(&Obj);
#else
        int result = pthread_mutex_destroy(&Obj);
        Y_ABORT_UNLESS(result == 0, "mutex destroy failure (%s)", LastSystemErrorText(result));
#endif
    }

    inline void Acquire() noexcept {
#if defined(_win_)
        EnterCriticalSection(&Obj);
#else
        int result = pthread_mutex_lock(&Obj);
        Y_ABORT_UNLESS(result == 0, "mutex lock failure (%s)", LastSystemErrorText(result));
#endif
    }

#if defined(_win_)
    static bool TryEnterCriticalSectionInt(CRITICAL_SECTION* obj) {
    #if (_WIN32_WINNT < 0x0400)
        if (-1L == ::InterlockedCompareExchange(&obj->LockCount, 0, -1)) {
            obj->OwningThread = (HANDLE)(DWORD_PTR)::GetCurrentThreadId();
            obj->RecursionCount = 1;

            return true;
        }

        if (obj->OwningThread == (HANDLE)(DWORD_PTR)::GetCurrentThreadId()) {
            ::InterlockedIncrement(&obj->LockCount);
            ++obj->RecursionCount;
            return true;
        }

        return false;
    #else  // _WIN32_WINNT < 0x0400
        return TryEnterCriticalSection(obj);
    #endif // _WIN32_WINNT < 0x0400
    }
#endif // _win_

    inline bool TryAcquire() noexcept {
#if defined(_win_)
        return TryEnterCriticalSectionInt(&Obj);
#else
        int result = pthread_mutex_trylock(&Obj);
        if (result == 0 || result == EBUSY) {
            return result == 0;
        }
        Y_ABORT("mutex trylock failure (%s)", LastSystemErrorText(result));
#endif
    }

    inline void Release() noexcept {
#if defined(_win_)
        LeaveCriticalSection(&Obj);
#else
        int result = pthread_mutex_unlock(&Obj);
        Y_ABORT_UNLESS(result == 0, "mutex unlock failure (%s)", LastSystemErrorText(result));
#endif
    }

    inline void* Handle() const noexcept {
        return (void*)&Obj;
    }

private:
#ifdef _win_
    CRITICAL_SECTION Obj;
#else
    pthread_mutex_t Obj;
#endif
};

TMutex::TMutex()
    : Impl_(new TImpl())
{
}

TMutex::TMutex(TMutex&&) noexcept = default;

TMutex::~TMutex() = default;

void TMutex::Acquire() noexcept {
    Impl_->Acquire();
}

bool TMutex::TryAcquire() noexcept {
    return Impl_->TryAcquire();
}

void TMutex::Release() noexcept {
    Impl_->Release();
}

void* TMutex::Handle() const noexcept {
    return Impl_->Handle();
}
