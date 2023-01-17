#pragma once

#include <library/cpp/threading/future/future.h>

#include <util/system/spinlock.h>
#include <util/generic/ptr.h>

#include <list>
#include <functional>

namespace NThreading {

class TAsyncSemaphore: public TThrRefBase {
public:
    using TPtr = TIntrusivePtr<TAsyncSemaphore>;

    class TAutoRelease {
    public:
        TAutoRelease(TAsyncSemaphore::TPtr sem)
            : Sem(std::move(sem))
        {
        }
        TAutoRelease(TAutoRelease&& other)
            : Sem(std::move(other.Sem))
        {
        }
        ~TAutoRelease();

        std::function<void (const TFuture<void>&)> DeferRelease();

    private:
        TAsyncSemaphore::TPtr Sem;
    };

    static TPtr Make(size_t count);

    TFuture<TPtr> AcquireAsync();
    void Release();
    void Cancel();

    TAutoRelease MakeAutoRelease() {
        return {this};
    }

private:
    TAsyncSemaphore(size_t count);

private:
    size_t Count_;
    bool Cancelled_ = false;
    TAdaptiveLock Lock_;
    std::list<TPromise<TPtr>> Promises_;
};

} // namespace NThreading
