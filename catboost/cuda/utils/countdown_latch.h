#pragma once

#include <library/cpp/deprecated/atomic/atomic.h>
#include <condition_variable>
#include <catboost/libs/helpers/exception.h>
#include <util/system/mutex.h>
#include <util/system/condvar.h>

class TCountDownLatch {
public:
    explicit TCountDownLatch(ui32 init)
        : Counter(init)
    {
    }

    void Countdown() {
        with_lock (Mutex) {
            AtomicDecrement(Counter);
            if (Counter <= 0) {
                CondVar.BroadCast();
            }
        }
    }

    void Wait() {
        with_lock (Mutex) {
            while (Counter > 0) {
                CondVar.WaitI(Mutex);
            }
        }
        CB_ENSURE(Counter == 0);
    }

private:
    TMutex Mutex;
    TAtomic Counter;
    TCondVar CondVar;
};
