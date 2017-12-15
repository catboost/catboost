#pragma once

#include <util/system/atomic.h>
#include <util/system/event.h>
#include <catboost/libs/helpers/exception.h>

class TCountDownLatch {
public:
    explicit TCountDownLatch(ui32 init)
        : Counter(init)
    {
    }

    void Countdown() {
        AtomicDecrement(Counter);
        if (Counter <= 0) {
            Done.Signal();
        }
    }

    void Wait() {
        Done.Wait();
        CB_ENSURE(Counter == 0);
    }

private:
    TAtomic Counter;
    TAutoEvent Done;
};
