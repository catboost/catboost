#pragma once

#include <util/generic/ptr.h>
#include <util/datetime/base.h>

struct TEventResetType {
    enum ResetMode {
        rAuto,   // the state will be nonsignaled after Wait() returns
        rManual, // we need call Reset() to set the state to nonsignaled.
    };
};

/**
 * DEPRECATED!
 *
 * Use TAutoEvent, TManualEvent for the direct replacement.
 * Use TManualEvent to prevent SEGFAULT (http://nga.at.yandex-team.ru/5772).
 */
class Event: public TEventResetType {
public:
    Event(ResetMode rmode = rManual);
    Event(const Event& other) noexcept;
    Event& operator=(const Event& other) noexcept;

    ~Event();

    void Reset() noexcept;
    void Signal() noexcept;

    /*
     * return true if signaled, false if timed out.
     */
    bool WaitD(TInstant deadLine) noexcept;

    /*
     * return true if signaled, false if timed out.
     */
    inline bool WaitT(TDuration timeOut) noexcept {
        return WaitD(timeOut.ToDeadLine());
    }

    /*
     * wait infinite time
     */
    inline void WaitI() noexcept {
        WaitD(TInstant::Max());
    }

    //return true if signaled, false if timed out.
    inline bool Wait(ui32 timer) noexcept {
        return WaitT(TDuration::MilliSeconds(timer));
    }

    inline bool Wait() noexcept {
        WaitI();

        return true;
    }

private:
    class TEvImpl;
    TIntrusivePtr<TEvImpl> EvImpl_;
};

class TAutoEvent: public Event {
public:
    TAutoEvent()
        : Event(Event::rAuto)
    {
    }

private:
    void Reset() noexcept;
};

/**
 * Prevents from a "shortcut problem" (see http://nga.at.yandex-team.ru/5772): if Wait will be called after Signaled
 * flag set to true in Signal method but before CondVar.BroadCast - Wait will shortcut (without actual wait on condvar).
 * If Wait thread will destruct event - Signal thread will do broadcast on a destructed CondVar.
 */
class TManualEvent {
public:
    TManualEvent()
        : Ev(TEventResetType::rManual)
    {
    }

    void Reset() noexcept {
        Event(Ev).Reset();
    }

    void Signal() noexcept {
        Event(Ev).Signal();
    }

    /** return true if signaled, false if timed out. */
    bool WaitD(TInstant deadLine) noexcept {
        return Event(Ev).WaitD(deadLine);
    }

    /** return true if signaled, false if timed out. */
    inline bool WaitT(TDuration timeOut) noexcept {
        return Event(Ev).WaitT(timeOut);
    }

    /** Wait infinite time */
    inline void WaitI() noexcept {
        Event(Ev).WaitI();
    }

    /** return true if signaled, false if timed out. */
    inline bool Wait(ui32 timer) noexcept {
        return Event(Ev).Wait(timer);
    }

    inline bool Wait() noexcept {
        return Event(Ev).Wait();
    }

private:
    Event Ev;
};
