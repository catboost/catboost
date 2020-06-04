#pragma once

#include "event.h"
#include "tracer.h"

#include <util/generic/maybe.h>

namespace NChromiumTrace {
    class TDurationEventGuard {
        TTracer* Tracer;

    public:
        TDurationEventGuard(TTracer* tracer, TStringBuf name, TStringBuf cat, const TEventArgs* args = nullptr) noexcept
            : Tracer(tracer)
        {
            Tracer->AddDurationBeginNow(name, cat, args);
        }

        ~TDurationEventGuard() noexcept {
            Tracer->AddDurationEndNow();
        }
    };

    class TCompleteEventGuard {
        TTracer* Tracer;
        TMaybe<TDurationCompleteEvent> Event;
        const TEventArgs* EventArgs;

    public:
        TCompleteEventGuard(TTracer* tracer, TStringBuf name, TStringBuf cat, const TEventArgs* args = nullptr) noexcept
            : Tracer(tracer)
            , Event(Tracer->BeginDurationCompleteNow(name, cat))
            , EventArgs(args)
        {
        }

        ~TCompleteEventGuard() noexcept {
            if (!Event)
                return;

            Tracer->EndDurationCompleteNow(*Event, EventArgs);
        }

        void SetInFlow(ui64 flowBindId) noexcept {
            if (!Event)
                return;

            auto& event = *Event;
            Y_ASSERT(event.Flow.Type == EFlowType::None || event.Flow.BindId == flowBindId);

            event.Flow.BindId = flowBindId;
            event.Flow.Type = event.Flow.Type | EFlowType::Consumer;
        }

        void SetOutFlow(ui64 flowBindId) noexcept {
            if (!Event)
                return;

            auto& event = *Event;
            Y_ASSERT(event.Flow.Type == EFlowType::None || event.Flow.BindId == flowBindId);

            event.Flow.BindId = flowBindId;
            event.Flow.Type = event.Flow.Type | EFlowType::Producer;
        }
    };

}
