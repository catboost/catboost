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
            cont->_SwitchToScheduler();

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
