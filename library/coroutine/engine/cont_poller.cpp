#include "cont_poller.h"
#include "impl.h"

namespace NCoro {

    void Reshedule(TCont* cont) noexcept {
        cont->ReSchedule();
    }

    void RemoveFromPoller(TCont* cont, IPollEvent* event) noexcept {
        cont->Executor()->Poller()->Remove(event);
    }

    TContPollEventHolder::TContPollEventHolder(void* memory, TCont* rep, SOCKET fds[], int what[], size_t nfds, TInstant deadline)
        : Events_((TFdEvent*)memory)
        , Count_(nfds)
    {
        for (size_t i = 0; i < Count_; ++i) {
            new (&(Events_[i])) TFdEvent(rep, fds[i], (ui16)what[i], deadline);
        }
    }

    TContPollEventHolder::~TContPollEventHolder() {
        for (size_t i = 0; i < Count_; ++i) {
            Events_[i].~TFdEvent();
        }
    }

    void TContPollEventHolder::ScheduleIoWait(TContExecutor* executor) noexcept {
        for (size_t i = 0; i < Count_; ++i) {
            executor->ScheduleIoWait(&(Events_[i]));
        }
    }

    TFdEvent* TContPollEventHolder::TriggeredEvent() noexcept {
        TFdEvent* ret = nullptr;
        int status = EINPROGRESS;

        for (size_t i = 0; i < Count_; ++i) {
            TFdEvent& ev = Events_[i];

            switch (ev.Status()) {
            case EINPROGRESS:
                break;

            case ETIMEDOUT:
                if (status != EINPROGRESS) {
                    break;
                } // else fallthrough

            default:
                status = ev.Status();
                ret = &ev;
            }
        }

        return ret;
    }
}
