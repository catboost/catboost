#include "sem.h"

#ifdef _win_
    #include <malloc.h>
#elif defined(_sun)
    #include <alloca.h>
#endif

#include <cstring>

#ifdef _win_
    #include "winint.h"
#else
    #include <semaphore.h>

    #if defined(_bionic_) || defined(_darwin_) && defined(_arm_)
        #include <fcntl.h>
    #else
        #define USE_SYSV_SEMAPHORES // unixoids declared the standard but not implemented it...
    #endif
#endif

#ifdef USE_SYSV_SEMAPHORES
    #include <errno.h>
    #include <sys/types.h>
    #include <sys/ipc.h>
    #include <sys/sem.h>

    #if defined(_linux_) || defined(_sun_) || defined(_cygwin_)
union semun {
    int val;
    struct semid_ds* buf;
    unsigned short* array;
} arg;
    #else
union semun arg;
    #endif
#endif

#include <util/digest/city.h>
#include <util/string/cast.h>
#include <util/random/fast.h>

#if !defined(_unix_) || defined(_darwin_)
    #include <util/random/random.h>
#endif

namespace {
    class TSemaphoreImpl {
    private:
#ifdef _win_
        using SEMHANDLE = HANDLE;
#else
    #ifdef USE_SYSV_SEMAPHORES
        using SEMHANDLE = int;
    #else
        using SEMHANDLE = sem_t*;
    #endif
#endif

        SEMHANDLE Handle;

    public:
        inline TSemaphoreImpl(const char* name, ui32 max_free_count)
            : Handle(0)
        {
#ifdef _win_
            char* key = (char*)name;
            if (name) {
                size_t len = strlen(name);
                key = (char*)alloca(len + 1);
                strcpy(key, name);
                if (len > MAX_PATH) {
                    *(key + MAX_PATH) = 0;
                }
                char* p = key;
                while (*p) {
                    if (*p == '\\') {
                        *p = '/';
                    }
                    ++p;
                }
            }
            // non-blocking on init
            Handle = ::CreateSemaphore(0, max_free_count, max_free_count, key);
#else
    #ifdef USE_SYSV_SEMAPHORES
            key_t key = TPCGMixer::Mix(CityHash64(name, strlen(name))); // 32 bit hash
            Handle = semget(key, 0, 0);                                 // try to open exist semaphore
            if (Handle == -1) {                                         // create new semaphore
                Handle = semget(key, 1, 0666 | IPC_CREAT);
                if (Handle != -1) {
                    union semun arg;
                    arg.val = max_free_count;
                    semctl(Handle, 0, SETVAL, arg);
                } else {
                    ythrow TSystemError() << "can not init sempahore";
                }
            }
    #else
            Handle = sem_open(name, O_CREAT, 0666, max_free_count);
            if (Handle == SEM_FAILED) {
                ythrow TSystemError() << "can not init sempahore";
            }
    #endif
#endif
        }

        inline ~TSemaphoreImpl() {
#ifdef _win_
            ::CloseHandle(Handle);
#else
    #ifdef USE_SYSV_SEMAPHORES
    // we DO NOT want 'semctl(Handle, 0, IPC_RMID)' for multiprocess tasks;
    // struct sembuf ops[] = {{0, 0, IPC_NOWAIT}};
    // if (semop(Handle, ops, 1) != 0) // close only if semaphore's value is zero
    //    semctl(Handle, 0, IPC_RMID);
    #else
            sem_close(Handle); // we DO NOT want sem_unlink(...)
    #endif
#endif
        }

        inline void Release() noexcept {
#ifdef _win_
            ::ReleaseSemaphore(Handle, 1, 0);
#else
    #ifdef USE_SYSV_SEMAPHORES
            struct sembuf ops[] = {{0, 1, SEM_UNDO}};
            int ret = semop(Handle, ops, 1);
    #else
            int ret = sem_post(Handle);
    #endif
            Y_ABORT_UNLESS(ret == 0, "can not release semaphore");
#endif
        }

        // The UNIX semaphore object does not support a timed "wait", and
        // hence to maintain consistancy, for win32 case we use INFINITE or 0 timeout.
        inline void Acquire() noexcept {
#ifdef _win_
            Y_ABORT_UNLESS(::WaitForSingleObject(Handle, INFINITE) == WAIT_OBJECT_0, "can not acquire semaphore");
#else
    #ifdef USE_SYSV_SEMAPHORES
            struct sembuf ops[] = {{0, -1, SEM_UNDO}};
            int ret = semop(Handle, ops, 1);
    #else
            int ret = sem_wait(Handle);
    #endif
            Y_ABORT_UNLESS(ret == 0, "can not acquire semaphore");
#endif
        }

        inline bool TryAcquire() noexcept {
#ifdef _win_
            // zero-second time-out interval
            // WAIT_OBJECT_0: current free count > 0
            // WAIT_TIMEOUT:  current free count == 0
            return ::WaitForSingleObject(Handle, 0) == WAIT_OBJECT_0;
#else
    #ifdef USE_SYSV_SEMAPHORES
            struct sembuf ops[] = {{0, -1, SEM_UNDO | IPC_NOWAIT}};
            int ret = semop(Handle, ops, 1);
    #else
            int ret = sem_trywait(Handle);
    #endif
            return ret == 0;
#endif
        }
    };

#if defined(_unix_)
    /*
    Disable errors/warnings about deprecated sem_* in Darwin
*/
    #ifdef _darwin_
    Y_PRAGMA_DIAGNOSTIC_PUSH
    Y_PRAGMA_NO_DEPRECATED
    #endif
    struct TPosixSemaphore {
        inline TPosixSemaphore(ui32 maxFreeCount) {
            if (sem_init(&S_, 0, maxFreeCount)) {
                ythrow TSystemError() << "can not init semaphore";
            }
        }

        inline ~TPosixSemaphore() {
            Y_ABORT_UNLESS(sem_destroy(&S_) == 0, "semaphore destroy failed");
        }

        inline void Acquire() noexcept {
            Y_ABORT_UNLESS(sem_wait(&S_) == 0, "semaphore acquire failed");
        }

        inline void Release() noexcept {
            Y_ABORT_UNLESS(sem_post(&S_) == 0, "semaphore release failed");
        }

        inline bool TryAcquire() noexcept {
            if (sem_trywait(&S_)) {
                Y_ABORT_UNLESS(errno == EAGAIN, "semaphore try wait failed");

                return false;
            }

            return true;
        }

        sem_t S_;
    };
    #ifdef _darwin_
    Y_PRAGMA_DIAGNOSTIC_POP
    #endif
#endif
} // namespace

class TSemaphore::TImpl: public TSemaphoreImpl {
public:
    inline TImpl(const char* name, ui32 maxFreeCount)
        : TSemaphoreImpl(name, maxFreeCount)
    {
    }
};

TSemaphore::TSemaphore(const char* name, ui32 maxFreeCount)
    : Impl_(new TImpl(name, maxFreeCount))
{
}

TSemaphore::~TSemaphore() = default;

void TSemaphore::Release() noexcept {
    Impl_->Release();
}

void TSemaphore::Acquire() noexcept {
    Impl_->Acquire();
}

bool TSemaphore::TryAcquire() noexcept {
    return Impl_->TryAcquire();
}

#if defined(_unix_) && !defined(_darwin_)
class TFastSemaphore::TImpl: public TPosixSemaphore {
public:
    inline TImpl(ui32 n)
        : TPosixSemaphore(n)
    {
    }
};
#else
class TFastSemaphore::TImpl: public TString, public TSemaphoreImpl {
public:
    inline TImpl(ui32 n)
        : TString(ToString(RandomNumber<ui64>()))
        , TSemaphoreImpl(c_str(), n)
    {
    }
};
#endif

TFastSemaphore::TFastSemaphore(ui32 maxFreeCount)
    : Impl_(new TImpl(maxFreeCount))
{
}

TFastSemaphore::~TFastSemaphore() = default;

void TFastSemaphore::Release() noexcept {
    Impl_->Release();
}

void TFastSemaphore::Acquire() noexcept {
    Impl_->Acquire();
}

bool TFastSemaphore::TryAcquire() noexcept {
    return Impl_->TryAcquire();
}
