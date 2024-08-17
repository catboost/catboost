#pragma once

#include "mutex.h"

#include <util/generic/ptr.h>
#include <util/generic/noncopyable.h>
#include <util/datetime/base.h>

#include <utility>

class TCondVar {
public:
    TCondVar();
    ~TCondVar();

    void BroadCast() noexcept;
    void Signal() noexcept;

    /*
     * returns false if failed by timeout
     */
    bool WaitD(TMutex& m, TInstant deadline) noexcept;

    template <typename P>
    inline bool WaitD(TMutex& m, TInstant deadline, P pred) noexcept {
        while (!pred()) {
            if (!WaitD(m, deadline)) {
                return pred();
            }
        }
        return true;
    }

    /*
     * returns false if failed by timeout
     */
    inline bool WaitT(TMutex& m, TDuration timeout) noexcept {
        return WaitD(m, timeout.ToDeadLine());
    }

    template <typename P>
    inline bool WaitT(TMutex& m, TDuration timeout, P pred) noexcept {
        return WaitD(m, timeout.ToDeadLine(), std::move(pred));
    }

    /*
     * infinite wait
     */
    inline void WaitI(TMutex& m) noexcept {
        WaitD(m, TInstant::Max());
    }

    template <typename P>
    inline void WaitI(TMutex& m, P pred) noexcept {
        WaitD(m, TInstant::Max(), std::move(pred));
    }

    // deprecated
    inline void Wait(TMutex& m) noexcept {
        WaitI(m);
    }

    template <typename P>
    inline void Wait(TMutex& m, P pred) noexcept {
        WaitI(m, std::move(pred));
    }

private:
    class TImpl;
    THolder<TImpl> Impl_;
};
