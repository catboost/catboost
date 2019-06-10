#pragma once

#include "impl.h"
#include "events.h"

class TContMutex {
public:
    TContMutex() noexcept
        : Token_(true)
    {
    }

    ~TContMutex() {
        Y_ASSERT(Token_);
    }

    int LockD(TCont* current, TInstant deadline) {
        while (!Token_) {
            const int ret = WaitQueue_.WaitD(current, deadline);

            if (ret != EWAKEDUP) {
                return ret;
            }
        }

        Token_ = false;

        return 0;
    }

    int LockT(TCont* current, TDuration timeout) {
        return LockD(current, timeout.ToDeadLine());
    }

    int LockI(TCont* current) {
        return LockD(current, TInstant::Max());
    }

    void UnLock() noexcept {
        Y_ASSERT(!Token_);

        Token_ = true;
        WaitQueue_.Signal();
    }

private:
    TContWaitQueue WaitQueue_;
    bool Token_;
};
