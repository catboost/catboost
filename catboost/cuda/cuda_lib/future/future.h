#pragma once

#include <util/generic/ptr.h>
#include <catboost/cuda/utils/spin_wait.h>
#include <catboost/libs/helpers/exception.h>

namespace NCudaLib {
    template <class T>
    class IDeviceFuture {
    public:
        virtual ~IDeviceFuture() {
        }

        virtual bool Has() = 0;

        virtual const T& Get() = 0;

        virtual void Wait() = 0;
    };

    class IDeviceRequest {
    public:
        virtual ~IDeviceRequest() {
        }

        virtual bool IsComplete() = 0;

        virtual void WaitComplete() {
            TSpinWaitHelper::Wait(TDuration::Max(), [&]() -> bool {
                return IsComplete();
            });
            CB_ENSURE(IsComplete());
        }
    };

    template <class T>
    using TDeviceFuturePtr = THolder<IDeviceFuture<T>>;
}
