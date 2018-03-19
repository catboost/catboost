#pragma once

#include <util/generic/ptr.h>
#include <util/generic/vector.h>
#include <util/network/socket.h>
#include <util/network/pollerimpl.h>
#include <util/datetime/base.h>

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

    typedef TVector<TEvent> TEvents;

    virtual ~IPollerFace() {
    }

    inline void Set(void* ptr, SOCKET fd, ui16 flags) {
        const TChange c = {fd, ptr, flags};

        Set(c);
    }

    virtual void Set(const TChange& change) = 0;
    virtual void Wait(TEvents& events, TInstant deadLine) = 0;

    static TAutoPtr<IPollerFace> Default();
    static TAutoPtr<IPollerFace> Construct(const TStringBuf& name);
};
