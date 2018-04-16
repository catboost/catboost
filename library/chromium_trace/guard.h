#pragma once

#include "event.h"
#include "tracer.h"

#include <util/generic/maybe.h>

namespace NChromiumTrace {
    class TDurationEventGuard {
        TTracer* Tracer;

    public:
        TDurationEventGuard(TTracer* tracer, TStringBuf name, TStringBuf cat) noexcept
            : Tracer(tracer)
        {
            Tracer->AddDurationBeginNow(name, cat);
        }

        ~TDurationEventGuard() noexcept {
            Tracer->AddDurationEndNow();
        }
    };

    class TCompleteEventGuard {
        TTracer* Tracer;
        TMaybe<TDurationCompleteEvent> Event;

    public:
        TCompleteEventGuard(TTracer* tracer, TStringBuf name, TStringBuf cat) noexcept
            : Tracer(tracer)
            , Event(Tracer->BeginDurationCompleteNow(name, cat))
        {
        }

        ~TCompleteEventGuard() noexcept {
            if (!Event)
                return;

            Tracer->EndDurationCompleteNow(*Event);
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
