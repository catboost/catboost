#pragma once

#include "event.h"

namespace NChromiumTrace {
    class ITraceConsumer {
    public:
        virtual ~ITraceConsumer();

        virtual void AddEvent(const TDurationBeginEvent& event, const TEventArgs* args) = 0;
        virtual void AddEvent(const TDurationEndEvent& event, const TEventArgs* args) = 0;
        virtual void AddEvent(const TDurationCompleteEvent& event, const TEventArgs* arg) = 0;
        virtual void AddEvent(const TCounterEvent& event, const TEventArgs* args) = 0;
        virtual void AddEvent(const TMetadataEvent& event, const TEventArgs* args) = 0;
    };

}
