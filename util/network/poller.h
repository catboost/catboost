#pragma once

#include "socket.h"

#include <util/generic/ptr.h>
#include <util/datetime/base.h>

class TSocketPoller {
public:
    TSocketPoller();
    ~TSocketPoller();

    void WaitRead(SOCKET sock, void* cookie);
    void WaitWrite(SOCKET sock, void* cookie);
    void WaitReadWrite(SOCKET sock, void* cookie);
    void WaitRdhup(SOCKET sock, void* cookie);

    void WaitReadOneShot(SOCKET sock, void* cookie);
    void WaitWriteOneShot(SOCKET sock, void* cookie);
    void WaitReadWriteOneShot(SOCKET sock, void* cookie);

    void WaitReadWriteEdgeTriggered(SOCKET sock, void* cookie);
    void RestartReadWriteEdgeTriggered(SOCKET sock, void* cookie, bool empty = true);

    void Unwait(SOCKET sock);

    size_t WaitD(void** events, size_t len, const TInstant& deadLine);

    inline size_t WaitT(void** events, size_t len, const TDuration& timeOut) {
        return WaitD(events, len, timeOut.ToDeadLine());
    }

    inline size_t WaitI(void** events, size_t len) {
        return WaitD(events, len, TInstant::Max());
    }

    inline void* WaitD(const TInstant& deadLine) {
        void* ret;

        if (WaitD(&ret, 1, deadLine)) {
            return ret;
        }

        return nullptr;
    }

    inline void* WaitT(const TDuration& timeOut) {
        return WaitD(timeOut.ToDeadLine());
    }

    inline void* WaitI() {
        return WaitD(TInstant::Max());
    }

private:
    class TImpl;
    THolder<TImpl> Impl_;
};
