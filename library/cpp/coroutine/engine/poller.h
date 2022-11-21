#pragma once

#include <util/generic/ptr.h>
#include <util/generic/vector.h>
#include <util/network/socket.h>
#include <util/network/pollerimpl.h>
#include <util/datetime/base.h>

enum class EContPoller {
    Default /* "default" */,
    Select /* "select" */,
    Poll /* "poll" */,
    Epoll /* "epoll" */,
    Kqueue /* "kqueue" */
};

class IPollerFace {
public:
    struct TChange {
        SOCKET Fd;
        void* Data;
        ui16 Flags;
    };

    struct TEvent {
        void* Data;
        int Status;
        ui16 Filter;
    };

    using TEvents = TVector<TEvent>;

    virtual ~IPollerFace() {
    }

    void Set(void* ptr, SOCKET fd, ui16 flags) {
        const TChange c = {fd, ptr, flags};

        Set(c);
    }

    virtual void Set(const TChange& change) = 0;
    virtual void Wait(TEvents& events, TInstant deadLine) = 0;
    virtual EContPoller PollEngine() const = 0;

    static THolder<IPollerFace> Default();
    static THolder<IPollerFace> Construct(TStringBuf name);
    static THolder<IPollerFace> Construct(EContPoller poller);
};
