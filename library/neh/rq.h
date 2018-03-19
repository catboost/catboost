#pragma once

#include "rpc.h"

namespace NNeh {
    class IRequestQueue {
    public:
        virtual ~IRequestQueue() {
        }

        virtual void Clear() = 0;
        virtual void Schedule(IRequestRef req) = 0;
        virtual IRequestRef Next() = 0;
    };

    typedef TAutoPtr<IRequestQueue> IRequestQueueRef;
    IRequestQueueRef CreateRequestQueue();
}
