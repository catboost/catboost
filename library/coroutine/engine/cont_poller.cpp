#include "cont_poller.h"
#include "impl.h"

namespace NCoro {
    void Reshedule(TCont* cont) noexcept {
        cont->ReSchedule();
    }

    void RemoveFromPoller(TCont* cont, IPollEvent* event) noexcept {
        cont->Executor()->Poller()->Remove(event);
    }
}
