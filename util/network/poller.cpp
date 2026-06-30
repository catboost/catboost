#include "poller.h"
#include "pollerimpl.h"

#include <util/memory/tempbuf.h>

namespace {
    struct TMutexLocking {
        using TMyMutex = TMutex;
    };
} // namespace

class TSocketPoller::TImpl: public TPollerImpl<TMutexLocking> {
public:
    inline size_t DoWaitReal(void** ev, TEvent* events, size_t len, const TInstant& deadLine) {
        const size_t ret = WaitD(events, len, deadLine);

        for (size_t i = 0; i < ret; ++i) {
            ev[i] = ExtractEvent(&events[i]);
        }

        return ret;
    }

    inline size_t DoWait(void** ev, size_t len, const TInstant& deadLine) {
        if (len == 1) {
            TEvent tmp;

            return DoWaitReal(ev, &tmp, 1, deadLine);
        } else {
            TTempArray<TEvent> tmpEvents(len);

            return DoWaitReal(ev, tmpEvents.Data(), len, deadLine);
        }
    }
};

TSocketPoller::TSocketPoller()
    : Impl_(new TImpl())
{
}

TSocketPoller::~TSocketPoller() = default;

void TSocketPoller::WaitRead(SOCKET sock, void* cookie) {
    Impl_->Set(cookie, sock, CONT_POLL_READ);
}

void TSocketPoller::WaitWrite(SOCKET sock, void* cookie) {
    Impl_->Set(cookie, sock, CONT_POLL_WRITE);
}

void TSocketPoller::WaitReadWrite(SOCKET sock, void* cookie) {
    Impl_->Set(cookie, sock, CONT_POLL_READ | CONT_POLL_WRITE);
}

void TSocketPoller::WaitRdhup(SOCKET sock, void* cookie) {
    Impl_->Set(cookie, sock, CONT_POLL_RDHUP);
}

void TSocketPoller::WaitReadOneShot(SOCKET sock, void* cookie) {
    Impl_->Set(cookie, sock, CONT_POLL_READ | CONT_POLL_ONE_SHOT);
}

void TSocketPoller::WaitWriteOneShot(SOCKET sock, void* cookie) {
    Impl_->Set(cookie, sock, CONT_POLL_WRITE | CONT_POLL_ONE_SHOT);
}

void TSocketPoller::WaitReadWriteOneShot(SOCKET sock, void* cookie) {
    Impl_->Set(cookie, sock, CONT_POLL_READ | CONT_POLL_WRITE | CONT_POLL_ONE_SHOT);
}

void TSocketPoller::WaitReadWriteEdgeTriggered(SOCKET sock, void* cookie) {
    Impl_->Set(cookie, sock, CONT_POLL_READ | CONT_POLL_WRITE | CONT_POLL_EDGE_TRIGGERED);
}

void TSocketPoller::RestartReadWriteEdgeTriggered(SOCKET sock, void* cookie, bool empty) {
    Impl_->Set(cookie, sock, CONT_POLL_READ | CONT_POLL_WRITE | CONT_POLL_MODIFY | CONT_POLL_EDGE_TRIGGERED | (empty ? CONT_POLL_BACKLOG_EMPTY : 0));
}

void TSocketPoller::Unwait(SOCKET sock) {
    Impl_->Remove(sock);
}

size_t TSocketPoller::WaitD(void** ev, size_t len, const TInstant& deadLine) {
    return Impl_->DoWait(ev, len, deadLine);
}
