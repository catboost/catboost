#pragma once

#include <condition_variable>
#include <catboost/libs/helpers/exception.h>
#include <util/system/mutex.h>
#include <util/system/condvar.h>

#include <atomic>


class TCountDownLatch {
public:
    explicit TCountDownLatch(ui32 init)
        : Counter(static_cast<i64>(init))
    {
    }

    void Countdown() {
        with_lock (Mutex) {
            if (--Counter <= 0) {
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
    std::atomic<i64> Counter;
    TCondVar CondVar;
};
