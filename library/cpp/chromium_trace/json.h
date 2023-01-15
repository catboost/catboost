#pragma once

#include "consumer.h"

#include <library/cpp/json/writer/json.h>

namespace NChromiumTrace {
    class TJsonTraceConsumer final: public ITraceConsumer {
        NJsonWriter::TBuf Json;

    public:
        TJsonTraceConsumer(IOutputStream* stream);
        ~TJsonTraceConsumer() override;

        void AddEvent(const TDurationBeginEvent& event, const TEventArgs* args) override;
        void AddEvent(const TDurationEndEvent& event, const TEventArgs* args) override;
        void AddEvent(const TDurationCompleteEvent& event, const TEventArgs* arg) override;
        void AddEvent(const TCounterEvent& event, const TEventArgs* args) override;
        void AddEvent(const TMetadataEvent& event, const TEventArgs* args) override;

    private:
        NJsonWriter::TPairContext BeginEvent(char type, const TEventOrigin& origin);
        void EndEvent(const TEventArgs* args);
        void WriteArgs(const TEventArgs& args);
        void WriteFlow(const TEventFlow& flow);
    };

}
