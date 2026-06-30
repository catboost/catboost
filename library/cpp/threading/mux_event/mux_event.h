#pragma once

#include <iterator>

#include <util/datetime/base.h>
#include <library/cpp/deprecated/atomic/atomic.h>
#include <util/system/defaults.h>
#include <util/system/event.h>
#include <util/system/guard.h>
#include <util/system/mutex.h>
#include <util/generic/list.h>
#include <util/generic/vector.h>
#include <util/generic/noncopyable.h>

class TMuxEvent: public TNonCopyable {
    friend inline int WaitForAnyEvent(TMuxEvent** array, const int size, TDuration timeout);

public:
    enum ResetMode {
        rManual,
        // TODO: rAuto is not supported yet
    };

    TMuxEvent(ResetMode rmode = rManual) {
        Y_UNUSED(rmode);
    }
    ~TMuxEvent() {
        Y_ABORT_UNLESS(WaitList.empty(), "");
    }

    // TODO: potentially unsafe, but currently I can't add "virtual" to TSystemEvent methods
    operator TSystemEvent&() {
        return MyEvent;
    }
    operator const TSystemEvent&() const {
        return MyEvent;
    }

    bool WaitD(TInstant deadLine) noexcept {
        return MyEvent.WaitD(deadLine);
    }

    // for rManual it's OK to ignore WaitList
    void Reset() noexcept {
        TGuard<TMutex> lock(WaitListLock);
        MyEvent.Reset(); // TODO: do we actually need to be locked here?
    }

    void Signal() noexcept {
        TGuard<TMutex> lock(WaitListLock);
        for (auto& i : WaitList) {
            i->Signal();
        }
        MyEvent.Signal(); // TODO: do we actually need to be locked here?
    }

    // same as in TSystemEvent
    inline bool WaitT(TDuration timeOut) noexcept {
        return WaitD(timeOut.ToDeadLine());
    }
    inline void WaitI() noexcept {
        WaitD(TInstant::Max());
    }
    inline bool Wait(ui32 timer) noexcept {
        return WaitT(TDuration::MilliSeconds(timer));
    }
    inline bool Wait() noexcept {
        WaitI();
        return true;
    }

private:
    TSystemEvent MyEvent;
    TMutex WaitListLock;
    TList<TSystemEvent*> WaitList;
};

///////////////////////////////////////////////////////////////////////////////

inline int WaitForAnyEvent(TMuxEvent** array, const int size, const TDuration timeout = TDuration::Max()) {
    TVector<TList<TSystemEvent*>::iterator> listIters;
    listIters.reserve(size);

    int result = -1;
    TSystemEvent e;

    for (int i = 0; i != size; ++i) {
        TMuxEvent& me = *array[i];

        TGuard<TMutex> lock(me.WaitListLock);
        if (me.MyEvent.Wait(0)) {
            result = i;
            break;
        }
        listIters.push_back(me.WaitList.insert(me.WaitList.end(), &e));
    }

    const bool timedOut = result == -1 && !e.WaitT(timeout);

    for (int i = 0; i != size; ++i) {
        TMuxEvent& me = *array[i];

        TGuard<TMutex> lock(me.WaitListLock);
        if (i < listIters.ysize()) {
            me.WaitList.erase(listIters[i]);
        }
        if (!timedOut && result == -1 && me.MyEvent.Wait(0)) { // always returns first signalled event
            result = i;
        }
    }

    Y_ASSERT(timedOut == (result == -1));
    return result;
}

///////////////////////////////////////////////////////////////////////////////

// TODO: rewrite via template<class TIter...>
inline int WaitForAnyEvent(TMuxEvent& e0, const TDuration timeout = TDuration::Max()) {
    TMuxEvent* array[] = {&e0};
    return WaitForAnyEvent(array, Y_ARRAY_SIZE(array), timeout);
}

inline int WaitForAnyEvent(TMuxEvent& e0, TMuxEvent& e1, const TDuration timeout = TDuration::Max()) {
    TMuxEvent* array[] = {&e0, &e1};
    return WaitForAnyEvent(array, Y_ARRAY_SIZE(array), timeout);
}

inline int WaitForAnyEvent(TMuxEvent& e0, TMuxEvent& e1, TMuxEvent& e2, const TDuration timeout = TDuration::Max()) {
    TMuxEvent* array[] = {&e0, &e1, &e2};
    return WaitForAnyEvent(array, Y_ARRAY_SIZE(array), timeout);
}
