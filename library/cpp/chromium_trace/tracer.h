#pragma once

#include "consumer.h"
#include "event.h"

#include <util/generic/maybe.h>
#include <util/generic/strbuf.h>

namespace NChromiumTrace {
    class TTracer {
        ITraceConsumer* Output = nullptr;

    public:
        TTracer() = default;
        TTracer(ITraceConsumer* output)
            : Output(output)
        {
        }

        void SetOutput(ITraceConsumer* output) {
            Output = output;
        }

        template <typename TEvent>
        void AddEvent(const TEvent& event, const TEventArgs* args = nullptr) {
            if (!Output) {
                return;
            }

            SuppressExceptions([&] {
                Output->AddEvent(event, args);
            });
        }

        void AddDurationBeginNow(TStringBuf name, TStringBuf cat, const TEventArgs* args = nullptr) noexcept;
        void AddDurationEndNow() noexcept;

        TMaybe<TDurationCompleteEvent> BeginDurationCompleteNow(TStringBuf name, TStringBuf cat) noexcept;
        void EndDurationCompleteNow(TDurationCompleteEvent& event, const TEventArgs* args = nullptr) noexcept;

        void AddCounterNow(TStringBuf name, TStringBuf cat, const TEventArgs& args) noexcept;

        void AddCurrentProcessName(TStringBuf name) noexcept;

        void AddCurrentThreadName(TStringBuf name) noexcept;

        void AddCurrentThreadIndex(i64 index) noexcept;

    private:
        template <typename F>
        static void SuppressExceptions(F f) {
            try {
                f();
            } catch (...) {
                NotifySuppressedException();
            }
        }

        static void NotifySuppressedException();
    };

}
