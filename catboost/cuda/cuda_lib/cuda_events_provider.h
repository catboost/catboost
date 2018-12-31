#pragma once

#include "cuda_base.h"
#include <util/system/spinlock.h>
#include <util/generic/vector.h>

namespace NCudaLib {
    class TCudaEventsProvider {
    private:
        TVector<cudaEvent_t> FreeHandles;
        TVector<cudaEvent_t> FreeHandlesWithoutTiming;
        static const ui64 RequestHandlesBatchSize = 16;
        TSpinLock Lock;

        void RequestHandle(bool disableTimming = true);

    public:
        class TCudaEvent: private TNonCopyable {
        private:
            mutable cudaEvent_t Event;
            bool IsWithoutTiming;
            TCudaEventsProvider* Owner;

        public:
            TCudaEvent(cudaEvent_t event,
                       bool isWithoutTiming,
                       TCudaEventsProvider* owner)
                : Event(event)
                , IsWithoutTiming(isWithoutTiming)
                , Owner(owner)
            {
            }

            ~TCudaEvent() {
                TGuard<TSpinLock> lock(Owner->Lock);
                if (IsWithoutTiming) {
                    Owner->FreeHandlesWithoutTiming.push_back(Event);
                } else {
                    Owner->FreeHandles.push_back(Event);
                }
            }

            void Record(const TCudaStream& stream) const {
                CUDA_SAFE_CALL(cudaEventRecord(Event, stream.GetStream()));
            }

            void StreamWait(const TCudaStream& stream) const {
                CUDA_SAFE_CALL(cudaStreamWaitEvent(stream.GetStream(), Event, 0));
            }

            void WaitComplete() const {
                CUDA_SAFE_CALL(cudaEventSynchronize(Event));
            }

            bool IsComplete() const {
                cudaError_t errorCode = cudaEventQuery(Event);
                if (errorCode == cudaSuccess) {
                    return true;
                }
                if (errorCode != cudaErrorNotReady) {
                    ythrow TCatBoostException() << "CUDA error " << (int)errorCode << ": " << cudaGetErrorString(errorCode);
                }
                return false;
            }
        };

    public:
        using TCudaEventPtr = THolder<TCudaEvent>;

        ~TCudaEventsProvider() noexcept(false) {
            for (auto event : FreeHandles) {
                CUDA_SAFE_CALL(cudaEventDestroy(event));
            }

            for (auto event : FreeHandlesWithoutTiming) {
                CUDA_SAFE_CALL(cudaEventDestroy(event));
            }
        }

        TCudaEventPtr Create(bool disableTimming = true);
    };

    using TCudaEvent = TCudaEventsProvider::TCudaEvent;
    using TCudaEventPtr = TCudaEventsProvider::TCudaEventPtr;

    inline static TCudaEventsProvider& CudaEventProvider() {
        return *FastTlsSingleton<TCudaEventsProvider>();
    }

    inline TCudaEventPtr CreateCudaEvent(bool disableTimming = true) {
        return CudaEventProvider().Create(disableTimming);
    }
}
