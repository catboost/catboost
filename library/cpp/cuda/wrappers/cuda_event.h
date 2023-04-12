#pragma once

#include "base.h"

class TCudaEvent {
private:
    struct Inner : public TThrRefBase {
    public:
        Inner(bool disableTiming);

        ~Inner() {
            CUDA_SAFE_CALL_FOR_DESTRUCTOR(cudaEventDestroy(Event));
        }

        cudaEvent_t Event;
        bool WithoutTiming;
    };

    explicit TCudaEvent(TIntrusivePtr<Inner> event);

private:
    TIntrusivePtr<Inner> Inner_;

public:
    static TCudaEvent NewEvent(bool disableTiming = true);

    void Record(cudaStream_t stream) const;

    void Record(const TCudaStream& stream) const;

    void StreamWait(const TCudaStream& stream) const;

    void WaitComplete() const;

    bool IsComplete() const;

    void Swap(TCudaEvent& other);

};

