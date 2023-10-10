#include "rwlock.h"

#include <util/generic/yexception.h>

#if defined(_unix_)
    #include <errno.h>
    #include <pthread.h>
#endif

#if defined(_win_) || defined(_darwin_)
    #include "mutex.h"
    #include "condvar.h"

//darwin rwlocks not recursive
class TRWMutex::TImpl {
public:
    TImpl();
    ~TImpl();

    void AcquireRead() noexcept;
    bool TryAcquireRead() noexcept;
    void ReleaseRead() noexcept;

    void AcquireWrite() noexcept;
    bool TryAcquireWrite() noexcept;
    void ReleaseWrite() noexcept;

    void Release() noexcept;

private:
    TMutex Lock_;
    int State_;
    TCondVar ReadCond_;
    TCondVar WriteCond_;
    int BlockedWriters_;
};

TRWMutex::TImpl::TImpl()
    : State_(0)
    , BlockedWriters_(0)
{
}

TRWMutex::TImpl::~TImpl() {
    Y_ABORT_UNLESS(State_ == 0, "failure, State_ != 0");
    Y_ABORT_UNLESS(BlockedWriters_ == 0, "failure, BlockedWriters_ != 0");
}

void TRWMutex::TImpl::AcquireRead() noexcept {
    with_lock (Lock_) {
        while (BlockedWriters_ || State_ < 0) {
            ReadCond_.Wait(Lock_);
        }

        ++State_;
    }

    ReadCond_.Signal();
}

bool TRWMutex::TImpl::TryAcquireRead() noexcept {
    with_lock (Lock_) {
        if (BlockedWriters_ || State_ < 0) {
            return false;
        }

        ++State_;
    }

    return true;
}

void TRWMutex::TImpl::ReleaseRead() noexcept {
    Lock_.Acquire();

    if (--State_ > 0) {
        Lock_.Release();
    } else if (BlockedWriters_) {
        Lock_.Release();
        WriteCond_.Signal();
    } else {
        Lock_.Release();
    }
}

void TRWMutex::TImpl::AcquireWrite() noexcept {
    with_lock (Lock_) {
        while (State_ != 0) {
            ++BlockedWriters_;
            WriteCond_.Wait(Lock_);
            --BlockedWriters_;
        }

        State_ = -1;
    }
}

bool TRWMutex::TImpl::TryAcquireWrite() noexcept {
    with_lock (Lock_) {
        if (State_ != 0) {
            return false;
        }

        State_ = -1;
    }

    return true;
}

void TRWMutex::TImpl::ReleaseWrite() noexcept {
    Lock_.Acquire();
    State_ = 0;

    if (BlockedWriters_) {
        Lock_.Release();
        WriteCond_.Signal();
    } else {
        Lock_.Release();
        ReadCond_.Signal();
    }
}

void TRWMutex::TImpl::Release() noexcept {
    Lock_.Acquire();

    if (State_ > 0) {
        if (--State_ > 0) {
            Lock_.Release();
        } else if (BlockedWriters_) {
            Lock_.Release();
            WriteCond_.Signal();
        }
    } else {
        State_ = 0;

        if (BlockedWriters_) {
            Lock_.Release();
            WriteCond_.Signal();
        } else {
            Lock_.Release();
            ReadCond_.Signal();
        }
    }
}

#else /* POSIX threads */

class TRWMutex::TImpl {
public:
    TImpl();
    ~TImpl();

    void AcquireRead() noexcept;
    bool TryAcquireRead() noexcept;
    void ReleaseRead() noexcept;

    void AcquireWrite() noexcept;
    bool TryAcquireWrite() noexcept;
    void ReleaseWrite() noexcept;

    void Release() noexcept;

private:
    pthread_rwlock_t Lock_;
};

TRWMutex::TImpl::TImpl() {
    int result = pthread_rwlock_init(&Lock_, nullptr);
    if (result != 0) {
        ythrow yexception() << "rwlock init failed (" << LastSystemErrorText(result) << ")";
    }
}

TRWMutex::TImpl::~TImpl() {
    const int result = pthread_rwlock_destroy(&Lock_);
    Y_ABORT_UNLESS(result == 0, "rwlock destroy failed (%s)", LastSystemErrorText(result));
}

void TRWMutex::TImpl::AcquireRead() noexcept {
    const int result = pthread_rwlock_rdlock(&Lock_);
    Y_ABORT_UNLESS(result == 0, "rwlock rdlock failed (%s)", LastSystemErrorText(result));
}

bool TRWMutex::TImpl::TryAcquireRead() noexcept {
    const int result = pthread_rwlock_tryrdlock(&Lock_);
    Y_ABORT_UNLESS(result == 0 || result == EBUSY, "rwlock tryrdlock failed (%s)", LastSystemErrorText(result));
    return result == 0;
}

void TRWMutex::TImpl::ReleaseRead() noexcept {
    const int result = pthread_rwlock_unlock(&Lock_);
    Y_ABORT_UNLESS(result == 0, "rwlock (read) unlock failed (%s)", LastSystemErrorText(result));
}

void TRWMutex::TImpl::AcquireWrite() noexcept {
    const int result = pthread_rwlock_wrlock(&Lock_);
    Y_ABORT_UNLESS(result == 0, "rwlock wrlock failed (%s)", LastSystemErrorText(result));
}

bool TRWMutex::TImpl::TryAcquireWrite() noexcept {
    const int result = pthread_rwlock_trywrlock(&Lock_);
    Y_ABORT_UNLESS(result == 0 || result == EBUSY, "rwlock trywrlock failed (%s)", LastSystemErrorText(result));
    return result == 0;
}

void TRWMutex::TImpl::ReleaseWrite() noexcept {
    const int result = pthread_rwlock_unlock(&Lock_);
    Y_ABORT_UNLESS(result == 0, "rwlock (write) unlock failed (%s)", LastSystemErrorText(result));
}

void TRWMutex::TImpl::Release() noexcept {
    const int result = pthread_rwlock_unlock(&Lock_);
    Y_ABORT_UNLESS(result == 0, "rwlock unlock failed (%s)", LastSystemErrorText(result));
}

#endif

TRWMutex::TRWMutex()
    : Impl_(new TImpl())
{
}

TRWMutex::~TRWMutex() = default;

void TRWMutex::AcquireRead() noexcept {
    Impl_->AcquireRead();
}

bool TRWMutex::TryAcquireRead() noexcept {
    return Impl_->TryAcquireRead();
}

void TRWMutex::ReleaseRead() noexcept {
    Impl_->ReleaseRead();
}

void TRWMutex::AcquireWrite() noexcept {
    Impl_->AcquireWrite();
}

bool TRWMutex::TryAcquireWrite() noexcept {
    return Impl_->TryAcquireWrite();
}

void TRWMutex::ReleaseWrite() noexcept {
    Impl_->ReleaseWrite();
}

void TRWMutex::Release() noexcept {
    Impl_->Release();
}
