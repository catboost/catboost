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
            cont->SwitchTo(cont->Executor()->SchedContext());

            if (cont->Cancelled()) {
                return ECANCELED;
            }

            return event->Status();
        }
    }

    void TContPollEvent::Wake() noexcept {
        TTreeNode::UnLink();
        TListNode::Unlink();
        Cont()->ReSchedule();
    }


    TInstant TEventWaitQueue::WakeTimedout(TInstant now) noexcept {
        ZeroWait_.ForEach([](TContPollEvent* ev) {
            ev->Wake(ETIMEDOUT);
        });

        TIoWaitTree::TIterator it = DeadlineWait_.Begin();

        if (it != DeadlineWait_.End()) {
            if (it->DeadLine() > now) {
                return it->DeadLine();
            }

            do {
                (it++)->Wake(ETIMEDOUT);
            } while (it != DeadlineWait_.End() && it->DeadLine() <= now);
        }

        return now;
    }

    void TEventWaitQueue::Register(NCoro::TContPollEvent* event) {
        const auto deadline = event->DeadLine();
        if (!deadline) {
            ZeroWait_.PushBack(event);
        } else if (deadline == TInstant::Max()) {
            InfiniteWait_.PushBack(event);
        } else {
            DeadlineWait_.Insert(event);
        }
        event->Cont()->Unlink();
    }

    void TEventWaitQueue::Abort() noexcept {
        auto canceler = [](TContPollEvent& e) {
            e.Cont()->Cancel();
        };
        auto cancelerPtr = [&](TContPollEvent* e) {
            canceler(*e);
        };
        ZeroWait_.ForEach(cancelerPtr);
        InfiniteWait_.ForEach(cancelerPtr);
        DeadlineWait_.ForEach(canceler);
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
