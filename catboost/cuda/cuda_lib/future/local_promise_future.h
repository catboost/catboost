#pragma once

#include "future.h"
#include <catboost/cuda/cuda_lib/cuda_events_provider.h>
#include <catboost/cuda/utils/spin_wait.h>
#include <library/cpp/threading/future/future.h>

namespace NCudaLib {
    template <class T>
    class TLocalHostFuture: public IDeviceFuture<T> {
    private:
        NThreading::TFuture<T> Future;
        T Result;
        bool IsSet = false;

    public:
        explicit TLocalHostFuture(NThreading::TFuture<T>&& future)
            : Future(std::move(future))
        {
        }

        bool Has() final {
            return Future.HasValue() || Future.HasException();
        }

        const T& Get() final {
            if (!IsSet) {
                TSpinWaitHelper::Wait(TDuration::Max(), [&]() -> bool {
                    return Future.HasValue();
                });
                //                Future.Wait();
                Result = Future.GetValue(TDuration::Max());
                IsSet = true;
            }
            return Result;
        }

        void Wait() final {
            return Future.Wait();
        }
    };

    template <class T>
    class TLocalHostPromise: public TMoveOnly {
    public:
        using TFuturePtr = THolder<TLocalHostFuture<T>>;

        TFuturePtr GetFuture() {
            return MakeHolder<TLocalHostFuture<T>>(Promise.GetFuture());
        }

        template <class TC>
        void SetValue(TC&& value) {
            Promise.SetValue(std::forward<TC>(value));
        }

        void Load(IInputStream*) {
            CB_ENSURE(false, "Unimplemented");
        }

        void Save(IOutputStream*) const {
            CB_ENSURE(false, "Unimplemented");
        }

    private:
        TLocalHostPromise()
            : Promise(NThreading::NewPromise<T>())
        {
        }

        template <bool>
        friend class TPromiseFactory;

    private:
        NThreading::TPromise<T> Promise;
    };

    class TLocalDeviceRequest: public IDeviceRequest {
    private:
        NThreading::TFuture<TCudaEventPtr> Event;
        bool IsCompleteFlag = false;

    public:
        explicit TLocalDeviceRequest(NThreading::TFuture<TCudaEventPtr>&& event)
            : Event(std::move(event))
        {
        }

        TLocalDeviceRequest(TLocalDeviceRequest&& other) = default;

        bool IsComplete() final {
            if (IsCompleteFlag) {
                return true;
            }
            if (Event.HasValue()) {
                IsCompleteFlag = Event.GetValue(TDuration::Max())->IsComplete();
                return IsCompleteFlag;
            } else {
                return false;
            }
        }

        void WaitComplete() final {
            TSpinWaitHelper::Wait(TDuration::Max(), [&]() -> bool {
                return Event.HasValue();
            });
            Event.GetValue(TDuration::Max())->WaitComplete();
            //            TSpinWaitHelper::Wait(TDuration::Max(), [&]() -> bool {
            //                return event->IsComplete();
            //            });

            IsCompleteFlag = true;
        }
    };

}
