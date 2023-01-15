#include "cont_poller.h"
#include "impl.h"

namespace NCoro {
    namespace {
        template <class T>
        int DoExecuteEvent(T* event) noexcept {
            auto* cont = event->Cont();

            if (cont->Cancelled()) {
                return ECANCELED;
            }

            cont->Executor()->ScheduleIoWait(event);
            cont->Switch();

            if (cont->Cancelled()) {
                return ECANCELED;
            }

            return event->Status();
        }
    }

    void TContPollEvent::Wake() noexcept {
        UnLink();
        Cont()->ReSchedule();
    }


    TInstant TEventWaitQueue::WakeTimedout(TInstant now) noexcept {
        TIoWait::TIterator it = IoWait_.Begin();

        if (it != IoWait_.End()) {
            if (it->DeadLine() > now) {
                return it->DeadLine();
            }

            do {
                (it++)->Wake(ETIMEDOUT);
            } while (it != IoWait_.End() && it->DeadLine() <= now);
        }

        return now;
    }

    void TEventWaitQueue::Register(NCoro::TContPollEvent* event) {
        IoWait_.Insert(event);
        event->Cont()->Unlink();
    }

    void TEventWaitQueue::Abort() noexcept {
        auto visitor = [](TContPollEvent& e) {
            e.Cont()->Cancel();
        };
        IoWait_.ForEach(visitor);
    }
}

void TFdEvent::RemoveFromIOWait() noexcept {
    this->Cont()->Executor()->Poller()->Remove(this);
}

int ExecuteEvent(TFdEvent* event) noexcept {
    return NCoro::DoExecuteEvent(event);
}

int ExecuteEvent(TTimerEvent* event) noexcept {
    return NCoro::DoExecuteEvent(event);
}
