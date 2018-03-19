#include "json.h"

using namespace NChromiumTrace;

TJsonTraceConsumer::TJsonTraceConsumer(IOutputStream* stream)
    : Json(NJsonWriter::HEM_UNSAFE, stream)
{
    Json.BeginList();
}

TJsonTraceConsumer::~TJsonTraceConsumer() {
    try {
        Json.EndList();
    } catch (...) {
    }
}

void TJsonTraceConsumer::AddEvent(const TDurationCompleteEvent& event, const TEventArgs* args) {
    BeginEvent('X', event.Origin)
        .WriteKey(STRINGBUF("ts"))
        .WriteULongLong(event.BeginTime.WallTime.MicroSeconds())
        .WriteKey(STRINGBUF("tts"))
        .WriteULongLong(event.BeginTime.ThreadCPUTime.MicroSeconds())
        .WriteKey(STRINGBUF("dur"))
        .WriteULongLong((event.EndTime.WallTime - event.BeginTime.WallTime).MicroSeconds())
        .WriteKey(STRINGBUF("tdur"))
        .WriteULongLong((event.EndTime.ThreadCPUTime - event.BeginTime.ThreadCPUTime).MicroSeconds())
        .WriteKey(STRINGBUF("name"))
        .WriteString(event.Name)
        .WriteKey(STRINGBUF("cat"))
        .WriteString(event.Categories);
    WriteFlow(event.Flow);
    EndEvent(args);
}

void TJsonTraceConsumer::AddEvent(const TDurationBeginEvent& event, const TEventArgs* args) {
    BeginEvent('B', event.Origin)
        .WriteKey(STRINGBUF("ts"))
        .WriteULongLong(event.Time.WallTime.MicroSeconds())
        .WriteKey(STRINGBUF("tts"))
        .WriteULongLong(event.Time.ThreadCPUTime.MicroSeconds())
        .WriteKey(STRINGBUF("name"))
        .WriteString(event.Name)
        .WriteKey(STRINGBUF("cat"))
        .WriteString(event.Categories);
    WriteFlow(event.Flow);
    EndEvent(args);
}

void TJsonTraceConsumer::AddEvent(const TDurationEndEvent& event, const TEventArgs* args) {
    BeginEvent('E', event.Origin)
        .WriteKey(STRINGBUF("ts"))
        .WriteULongLong(event.Time.WallTime.MicroSeconds())
        .WriteKey(STRINGBUF("tts"))
        .WriteULongLong(event.Time.ThreadCPUTime.MicroSeconds());
    WriteFlow(event.Flow);
    EndEvent(args);
}

void TJsonTraceConsumer::AddEvent(const TCounterEvent& event, const TEventArgs* args) {
    BeginEvent('C', event.Origin)
        .WriteKey(STRINGBUF("ts"))
        .WriteULongLong(event.Time.WallTime.MicroSeconds())
        .WriteKey(STRINGBUF("name"))
        .WriteString(event.Name)
        .WriteKey(STRINGBUF("cat"))
        .WriteString(event.Categories);
    EndEvent(args);
}

void TJsonTraceConsumer::AddEvent(const TMetadataEvent& event, const TEventArgs* args) {
    BeginEvent('M', event.Origin)
        .WriteKey(STRINGBUF("name"))
        .WriteString(event.Name);
    EndEvent(args);
}

NJsonWriter::TPairContext TJsonTraceConsumer::BeginEvent(char type, const TEventOrigin& origin) {
    const char ph[2] = {type, 0};
    return Json.BeginObject()
        .WriteKey(STRINGBUF("ph"))
        .WriteString(STRINGBUF(ph))
        .WriteKey(STRINGBUF("pid"))
        .WriteULongLong(origin.ProcessId)
        .WriteKey(STRINGBUF("tid"))
        .WriteULongLong(origin.ThreadId);
}

void TJsonTraceConsumer::EndEvent(const TEventArgs* args) {
    if (args) {
        WriteArgs(*args);
    }
    Json.EndObject().UnsafeWriteRawBytes(STRINGBUF("\n"));
}

void TJsonTraceConsumer::WriteArgs(const TEventArgs& args) {
    struct TWriteArg {
        NJsonWriter::TBuf* Output;

        void operator()(TStringBuf value) {
            Output->WriteString(value);
        }

        void operator()(i64 value) {
            Output->WriteLongLong(value);
        }

        void operator()(double value) {
            Output->WriteDouble(value);
        }
    };

    Json.WriteKey(STRINGBUF("args")).BeginObject();
    for (const auto& item : args.Items) {
        Json.WriteKey(item.Name);
        item.Value.Visit(TWriteArg{&Json});
    }
    Json.EndObject();
}

void TJsonTraceConsumer::WriteFlow(const TEventFlow& flow) {
    if (flow.Type == EFlowType::None) {
        return;
    }

    if (flow.Type == EFlowType::Producer || flow.Type == EFlowType::Step) {
        Json.WriteKey(STRINGBUF("flow_out")).WriteBool(true);
    }

    if (flow.Type == EFlowType::Consumer || flow.Type == EFlowType::Step) {
        Json.WriteKey(STRINGBUF("flow_in")).WriteBool(true);
    }

    Json.WriteKey(STRINGBUF("bind_id")).WriteULongLong(flow.BindId);
}
