#pragma once

#include "base.h"

class TCudaEvent {
private:
    struct Inner: public TThrRefBase {
    public:
        Inner(bool disableTiming)
            : WithoutTiming(disableTiming)
        {
            if (disableTiming) {
                CUDA_SAFE_CALL(cudaEventCreateWithFlags(&Event, cudaEventDisableTiming | cudaEventBlockingSync));
            } else {
                CUDA_SAFE_CALL(cudaEventCreate(&Event));
            }
        }

        ~Inner() {
            CUDA_SAFE_CALL(cudaEventDestroy(Event));
        }

        cudaEvent_t Event;
        bool WithoutTiming;
    };

    TCudaEvent(TIntrusivePtr<Inner> event)
        : Inner_(std::move(event))
    {
    }

private:
    TIntrusivePtr<Inner> Inner_;

public:
    static TCudaEvent NewEvent(bool disableTiming = true) {
        return TCudaEvent(new Inner(disableTiming));
    }
    void Record(const TCudaStream& stream) const {
        CUDA_SAFE_CALL(cudaEventRecord(Inner_->Event, stream));
    }

    void StreamWait(const TCudaStream& stream) const {
        CUDA_SAFE_CALL(cudaStreamWaitEvent(stream, Inner_->Event, 0));
    }

    void WaitComplete() const {
        CUDA_SAFE_CALL(cudaEventSynchronize(Inner_->Event));
    }

    bool IsComplete() const {
        cudaError_t errorCode = cudaEventQuery(Inner_->Event);
        if (errorCode == cudaSuccess) {
            return true;
        }
        if (errorCode != cudaErrorNotReady) {
            ythrow TCudaException(errorCode) << "CUDA error " << (int)errorCode << ": " << cudaGetErrorString(errorCode);
        }
        return false;
    }

    void Swap(TCudaEvent& other) {
        Inner_.Swap(other.Inner_);
    }
};
