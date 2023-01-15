#include "tracer.h"

#include <library/cpp/json/json_writer.h>

namespace NChromiumTrace {
    void TTracer::NotifySuppressedException() {
        static bool messageWritten = false;
        if (!messageWritten) {
            Cerr << "WARNING: Exception in trace consumer. " << CurrentExceptionMessage() << " (further messages will be suppressed)" << Endl;
            messageWritten = true;
        }
    }

    void TTracer::AddDurationBeginNow(TStringBuf name, TStringBuf cat, const TEventArgs* args) noexcept {
        if (!Output)
            return;

        SuppressExceptions([&] {
            Output->AddEvent(TDurationBeginEvent{
                                 TEventOrigin::Here(),
                                 name,
                                 cat,
                                 TEventTime::Now(),
                                 TEventFlow{EFlowType::None, 0},
                             },
                             args);
        });
    }

    void TTracer::AddDurationEndNow() noexcept {
        if (!Output)
            return;

        SuppressExceptions([&] {
            Output->AddEvent(TDurationEndEvent{
                                 TEventOrigin::Here(),
                                 TEventTime::Now(),
                                 TEventFlow{EFlowType::None, 0},
                             },
                             nullptr);
        });
    }

    TMaybe<TDurationCompleteEvent> TTracer::BeginDurationCompleteNow(TStringBuf name, TStringBuf cat) noexcept {
        if (!Output)
            return Nothing();

        return TDurationCompleteEvent{
            TEventOrigin::Here(),
            name,
            cat,
            TEventTime::Now(),
            TEventTime(),
            TEventFlow{EFlowType::None, 0},
        };
    }

    void TTracer::EndDurationCompleteNow(TDurationCompleteEvent& event, const TEventArgs* args) noexcept {
        event.EndTime = TEventTime::Now();
        AddEvent(event, args);
    }

    void TTracer::AddCounterNow(TStringBuf name, TStringBuf cat, const TEventArgs& args) noexcept {
        if (!Output)
            return;

        SuppressExceptions([&] {
            Output->AddEvent(TCounterEvent{
                                 TEventOrigin::Here(),
                                 name,
                                 cat,
                                 TEventTime::Now(),
                             },
                             &args);
        });
    }

    void TTracer::AddCurrentProcessName(TStringBuf name) noexcept {
        if (!Output)
            return;

        SuppressExceptions([&] {
            Output->AddEvent(TMetadataEvent{
                                 TEventOrigin::Here(),
                                 TStringBuf("process_name"),
                             },
                             &TEventArgs().Add(TStringBuf("name"), name));
        });
    }

    void TTracer::AddCurrentThreadName(TStringBuf name) noexcept {
        if (!Output)
            return;

        SuppressExceptions([&] {
            Output->AddEvent(TMetadataEvent{
                                 TEventOrigin::Here(),
                                 TStringBuf("thread_name"),
                             },
                             &TEventArgs().Add(TStringBuf("name"), name));
        });
    }

    void TTracer::AddCurrentThreadIndex(i64 index) noexcept {
        if (!Output)
            return;

        SuppressExceptions([&] {
            Output->AddEvent(TMetadataEvent{
                                 TEventOrigin::Here(),
                                 TStringBuf("thread_sort_index"),
                             },
                             &TEventArgs().Add(TStringBuf("sort_index"), index));
        });
    }

}
