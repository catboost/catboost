#include "yson.h"

#include <util/generic/singleton.h>
#include <util/string/builder.h>
#include <util/system/env.h>
#include <util/system/hostname.h>

namespace {
    struct TWriteArg {
        NYT::TYsonWriter* Output;

        void operator()(i64 value) {
            Output->OnInt64Scalar(value);
        }

        void operator()(ui64 value) {
            Output->OnUint64Scalar(value);
        }

        void operator()(double value) {
            Output->OnDoubleScalar(value);
        }

        void operator()(TStringBuf value) {
            Output->OnStringScalar(value);
        }
    };
}

using namespace NChromiumTrace;

TYsonTraceConsumer::TYsonTraceConsumer(IOutputStream* stream, NYT::EYsonFormat format)
    : Yson(stream, format, NYT::YT_LIST_FRAGMENT)
    , JobId(GetCurrentJobId())
{
}

TYsonTraceConsumer::~TYsonTraceConsumer() = default;

TString TYsonTraceConsumer::GetCurrentJobId() {
    return GetEnv("YT_JOB_ID");
}

void TYsonTraceConsumer::AddEvent(const TDurationCompleteEvent& event, const TEventArgs* args) {
    BeginEvent('X', event.Origin);
    Yson.OnKeyedItem(AsStringBuf("ts"));
    Yson.OnUint64Scalar(event.BeginTime.WallTime.MicroSeconds());
    Yson.OnKeyedItem(AsStringBuf("tts"));
    Yson.OnUint64Scalar(event.BeginTime.ThreadCPUTime.MicroSeconds());
    Yson.OnKeyedItem(AsStringBuf("dur"));
    Yson.OnUint64Scalar((event.EndTime.WallTime - event.BeginTime.WallTime).MicroSeconds());
    Yson.OnKeyedItem(AsStringBuf("tdur"));
    Yson.OnUint64Scalar((event.EndTime.ThreadCPUTime - event.BeginTime.ThreadCPUTime).MicroSeconds());
    Yson.OnKeyedItem(AsStringBuf("name"));
    Yson.OnStringScalar(event.Name);
    Yson.OnKeyedItem(AsStringBuf("cat"));
    Yson.OnStringScalar(event.Categories);
    WriteFlow(event.Flow);
    EndEvent(args);
}

void TYsonTraceConsumer::AddEvent(const TDurationBeginEvent& event, const TEventArgs* args) {
    BeginEvent('B', event.Origin);
    Yson.OnKeyedItem(AsStringBuf("ts"));
    Yson.OnUint64Scalar(event.Time.WallTime.MicroSeconds());
    Yson.OnKeyedItem(AsStringBuf("tts"));
    Yson.OnUint64Scalar(event.Time.ThreadCPUTime.MicroSeconds());
    Yson.OnKeyedItem(AsStringBuf("name"));
    Yson.OnStringScalar(event.Name);
    Yson.OnKeyedItem(AsStringBuf("cat"));
    Yson.OnStringScalar(event.Categories);
    WriteFlow(event.Flow);
    EndEvent(args);
}

void TYsonTraceConsumer::AddEvent(const TDurationEndEvent& event, const TEventArgs* args) {
    BeginEvent('E', event.Origin);
    Yson.OnKeyedItem(AsStringBuf("ts"));
    Yson.OnUint64Scalar(event.Time.WallTime.MicroSeconds());
    Yson.OnKeyedItem(AsStringBuf("tts"));
    Yson.OnUint64Scalar(event.Time.ThreadCPUTime.MicroSeconds());
    WriteFlow(event.Flow);
    EndEvent(args);
}

void TYsonTraceConsumer::AddEvent(const TCounterEvent& event, const TEventArgs* args) {
    BeginEvent('C', event.Origin);
    Yson.OnKeyedItem(AsStringBuf("ts"));
    Yson.OnUint64Scalar(event.Time.WallTime.MicroSeconds());
    Yson.OnKeyedItem(AsStringBuf("name"));
    Yson.OnStringScalar(event.Name);
    Yson.OnKeyedItem(AsStringBuf("cat"));
    Yson.OnStringScalar(event.Categories);
    EndEvent(args);
}

void TYsonTraceConsumer::AddEvent(const TMetadataEvent& event, const TEventArgs* args) {
    BeginEvent('M', event.Origin);
    Yson.OnKeyedItem(AsStringBuf("name"));
    Yson.OnStringScalar(event.Name);
    EndEvent(args);
}

void TYsonTraceConsumer::BeginEvent(char type, const TEventOrigin& origin) {
    Yson.OnBeginMap();
    char ph[2] = {type, 0};

    if (JobId) {
        Yson.OnKeyedItem(AsStringBuf("host"));
        Yson.OnStringScalar(HostName());
        Yson.OnKeyedItem(AsStringBuf("job_id"));
        Yson.OnStringScalar(JobId);
    }

    Yson.OnKeyedItem(AsStringBuf("ph"));
    Yson.OnStringScalar(AsStringBuf(ph));
    Yson.OnKeyedItem(AsStringBuf("pid"));
    Yson.OnUint64Scalar(origin.ProcessId);
    Yson.OnKeyedItem(AsStringBuf("tid"));
    Yson.OnUint64Scalar(origin.ThreadId);
}

void TYsonTraceConsumer::EndEvent(const TEventArgs* args) {
    if (args) {
        WriteArgs(*args);
    }
    Yson.OnEndMap();
}

void TYsonTraceConsumer::WriteArgs(const TEventArgs& args) {
    Yson.OnKeyedItem(AsStringBuf("args"));
    Yson.OnBeginMap();
    for (const auto& item : args.Items) {
        Yson.OnKeyedItem(item.Name);
        Visit(TWriteArg{&Yson}, item.Value);
    }
    Yson.OnEndMap();
}

void TYsonTraceConsumer::WriteFlow(const TEventFlow& flow) {
    if (flow.Type == EFlowType::None) {
        return;
    }

    if (flow.Type == EFlowType::Producer || flow.Type == EFlowType::Step) {
        Yson.OnKeyedItem(AsStringBuf("flow_out"));
        Yson.OnBooleanScalar(true);
    }

    if (flow.Type == EFlowType::Consumer || flow.Type == EFlowType::Step) {
        Yson.OnKeyedItem(AsStringBuf("flow_in"));
        Yson.OnBooleanScalar(true);
    }

    TString bindId = TStringBuilder() << JobId << "::" << flow.BindId;
    Yson.OnKeyedItem(AsStringBuf("bind_id"));
    Yson.OnStringScalar(bindId);
}
