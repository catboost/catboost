#pragma once

#include "events.h"
#include "mutex.h"

class TContCondVar {
public:
    int WaitD(TCont* current, TContMutex* mutex, TInstant deadline) {
        mutex->UnLock();

        const int ret = WaitQueue_.WaitD(current, deadline);

        if (ret != EWAKEDUP) {
            return ret;
        }

        return mutex->LockD(current, deadline);
    }

    int WaitT(TCont* current, TContMutex* mutex, TDuration timeout) {
        return WaitD(current, mutex, timeout.ToDeadLine());
    }

    int WaitI(TCont* current, TContMutex* mutex) {
        return WaitD(current, mutex, TInstant::Max());
    }

    void Signal() noexcept {
        WaitQueue_.Signal();
    }

    void BroadCast() noexcept {
        WaitQueue_.BroadCast();
    }

private:
    TContWaitQueue WaitQueue_;
};
