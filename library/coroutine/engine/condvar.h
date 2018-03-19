#pragma once

class TContCondVar {
public:
    inline int WaitD(TCont* current, TContMutex* mutex, TInstant deadline) {
        mutex->UnLock();

        const int ret = WaitQueue_.WaitD(current, deadline);

        if (ret != EWAKEDUP) {
            return ret;
        }

        return mutex->LockD(current, deadline);
    }

    inline int WaitT(TCont* current, TContMutex* mutex, TDuration timeout) {
        return WaitD(current, mutex, timeout.ToDeadLine());
    }

    inline int WaitI(TCont* current, TContMutex* mutex) {
        return WaitD(current, mutex, TInstant::Max());
    }

    inline void Signal() noexcept {
        WaitQueue_.Signal();
    }

    inline void BroadCast() noexcept {
        WaitQueue_.BroadCast();
    }

private:
    TContWaitQueue WaitQueue_;
};
