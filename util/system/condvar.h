#pragma once

#include "mutex.h"

#include <util/generic/ptr.h>
#include <util/generic/noncopyable.h>
#include <util/datetime/base.h>

class TCondVar {
public:
    TCondVar();
    ~TCondVar();

    void BroadCast() noexcept;
    void Signal() noexcept;

    /*
     * returns false if failed by timeout
     */
    bool WaitD(TMutex& m, TInstant deadLine) noexcept;

    /*
     * returns false if failed by timeout
     */
    inline bool WaitT(TMutex& m, TDuration timeOut) noexcept {
        return WaitD(m, timeOut.ToDeadLine());
    }

    /*
     * infinite wait
     */
    inline void WaitI(TMutex& m) noexcept {
        WaitD(m, TInstant::Max());
    }

    //deprecated
    inline void Wait(TMutex& m) noexcept {
        WaitI(m);
    }

    inline bool TimedWait(TMutex& m, TDuration timeOut) noexcept {
        return WaitT(m, timeOut);
    }

private:
    class TImpl;
    THolder<TImpl> Impl_;
};
