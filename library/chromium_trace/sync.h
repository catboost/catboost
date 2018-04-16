#pragma once

#include "consumer.h"

#include <util/system/mutex.h>

namespace NChromiumTrace {
    template <typename TRealTraceConsumer>
    class TSyncTraceConsumer: public ITraceConsumer {
        TMutex Lock;
        TRealTraceConsumer RealTraceConsumer;

    public:
        template <typename... Args>
        TSyncTraceConsumer(Args&&... args)
            : RealTraceConsumer(std::forward<Args>(args)...)
        {
        }

#define SYNC_ADD_EVENT(type__)                                            \
    void AddEvent(const type__& event, const TEventArgs* args) override { \
        with_lock (Lock) {                                                \
            RealTraceConsumer.AddEvent(event, args);                      \
        }                                                                 \
    }

        SYNC_ADD_EVENT(TDurationBeginEvent)
        SYNC_ADD_EVENT(TDurationEndEvent)
        SYNC_ADD_EVENT(TDurationCompleteEvent)
        SYNC_ADD_EVENT(TCounterEvent)
        SYNC_ADD_EVENT(TMetadataEvent)

#undef SYNC_ADD_EVENT
    };

}
