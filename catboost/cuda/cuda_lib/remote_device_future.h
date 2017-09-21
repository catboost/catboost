#pragma once

#include "cuda_base.h"
#include "cuda_events_provider.h"
#include "helpers.h"
#include <future>
#include <library/threading/future/future.h>

namespace NCudaLib {
    template <class T>
    class TDeviceFuture {
    private:
        NThreading::TFuture<T> Future;

    public:
        explicit TDeviceFuture(NThreading::TFuture<T>&& future)
            : Future(std::move(future))
        {
        }

        T Get() {
            Future.Wait();
            return Future.GetValue(TInstant::Max());
        }

        void Wait() {
            return Future.Wait();
        }
    };

    class TDeviceEvent: public TMoveOnly {
    private:
        NThreading::TFuture<TCudaEventPtr> Event;
        bool IsCompleteFlag = false;

    public:
        explicit TDeviceEvent(NThreading::TFuture<TCudaEventPtr>&& event)
            : Event(std::move(event))
        {
        }

        TDeviceEvent(TDeviceEvent&& other) = default;

        bool IsComplete() {
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

        void WaitComplete() {
            Event.GetValue(TDuration::Max())->WaitComplete();
            IsCompleteFlag = true;
        }
    };
}
