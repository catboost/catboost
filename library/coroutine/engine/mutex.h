#pragma once

#include "impl.h"
#include "events.h"

class TContMutex {
public:
    inline TContMutex() noexcept
        : Token_(true)
    {
    }

    inline ~TContMutex() {
        Y_ASSERT(Token_);
    }

    inline int LockD(TCont* current, TInstant deadline) {
        while (!Token_) {
            const int ret = WaitQueue_.WaitD(current, deadline);

            if (ret != EWAKEDUP) {
                return ret;
            }
        }

        Token_ = false;

        return 0;
    }

    inline int LockT(TCont* current, TDuration timeout) {
        return LockD(current, timeout.ToDeadLine());
    }

    inline int LockI(TCont* current) {
        return LockD(current, TInstant::Max());
    }

    inline void UnLock() noexcept {
        Y_ASSERT(!Token_);

        Token_ = true;
        WaitQueue_.Signal();
    }

private:
    TContWaitQueue WaitQueue_;
    bool Token_;
};
