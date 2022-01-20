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
        .WriteKey(TStringBuf("ts"))
        .WriteULongLong(event.BeginTime.WallTime.MicroSeconds())
        .WriteKey(TStringBuf("tts"))
        .WriteULongLong(event.BeginTime.ThreadCPUTime.MicroSeconds())
        .WriteKey(TStringBuf("dur"))
        .WriteULongLong((event.EndTime.WallTime - event.BeginTime.WallTime).MicroSeconds())
        .WriteKey(TStringBuf("tdur"))
        .WriteULongLong((event.EndTime.ThreadCPUTime - event.BeginTime.ThreadCPUTime).MicroSeconds())
        .WriteKey(TStringBuf("name"))
        .WriteString(event.Name)
        .WriteKey(TStringBuf("cat"))
        .WriteString(event.Categories);
    WriteFlow(event.Flow);
    EndEvent(args);
}

void TJsonTraceConsumer::AddEvent(const TDurationBeginEvent& event, const TEventArgs* args) {
    BeginEvent('B', event.Origin)
        .WriteKey(TStringBuf("ts"))
        .WriteULongLong(event.Time.WallTime.MicroSeconds())
        .WriteKey(TStringBuf("tts"))
        .WriteULongLong(event.Time.ThreadCPUTime.MicroSeconds())
        .WriteKey(TStringBuf("name"))
        .WriteString(event.Name)
        .WriteKey(TStringBuf("cat"))
        .WriteString(event.Categories);
    WriteFlow(event.Flow);
    EndEvent(args);
}

void TJsonTraceConsumer::AddEvent(const TDurationEndEvent& event, const TEventArgs* args) {
    BeginEvent('E', event.Origin)
        .WriteKey(TStringBuf("ts"))
        .WriteULongLong(event.Time.WallTime.MicroSeconds())
        .WriteKey(TStringBuf("tts"))
        .WriteULongLong(event.Time.ThreadCPUTime.MicroSeconds());
    WriteFlow(event.Flow);
    EndEvent(args);
}

void TJsonTraceConsumer::AddEvent(const TCounterEvent& event, const TEventArgs* args) {
    BeginEvent('C', event.Origin)
        .WriteKey(TStringBuf("ts"))
        .WriteULongLong(event.Time.WallTime.MicroSeconds())
        .WriteKey(TStringBuf("name"))
        .WriteString(event.Name)
        .WriteKey(TStringBuf("cat"))
        .WriteString(event.Categories);
    EndEvent(args);
}

void TJsonTraceConsumer::AddEvent(const TMetadataEvent& event, const TEventArgs* args) {
    BeginEvent('M', event.Origin)
        .WriteKey(TStringBuf("name"))
        .WriteString(event.Name);
    EndEvent(args);
}

NJsonWriter::TPairContext TJsonTraceConsumer::BeginEvent(char type, const TEventOrigin& origin) {
    return Json.BeginObject()
        .WriteKey(TStringBuf("ph"))
        .WriteString(TStringBuf(&type, 1))
        .WriteKey(TStringBuf("pid"))
        .WriteULongLong(origin.ProcessId)
        .WriteKey(TStringBuf("tid"))
        .WriteULongLong(origin.ThreadId);
}

void TJsonTraceConsumer::EndEvent(const TEventArgs* args) {
    if (args) {
        WriteArgs(*args);
    }
    Json.EndObject().UnsafeWriteRawBytes(TStringBuf("\n"));
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

    Json.WriteKey(TStringBuf("args")).BeginObject();
    for (const auto& item : args.Items) {
        Json.WriteKey(item.Name);
        std::visit(TWriteArg{&Json}, item.Value);
    }
    Json.EndObject();
}

void TJsonTraceConsumer::WriteFlow(const TEventFlow& flow) {
    if (flow.Type == EFlowType::None) {
        return;
    }

    if (flow.Type == EFlowType::Producer || flow.Type == EFlowType::Step) {
        Json.WriteKey(TStringBuf("flow_out")).WriteBool(true);
    }

    if (flow.Type == EFlowType::Consumer || flow.Type == EFlowType::Step) {
        Json.WriteKey(TStringBuf("flow_in")).WriteBool(true);
    }

    Json.WriteKey(TStringBuf("bind_id")).WriteULongLong(flow.BindId);
}
