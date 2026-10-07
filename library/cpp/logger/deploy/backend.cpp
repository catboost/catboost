#include "backend.h"

#include <library/cpp/json/writer/json.h>

#include <util/datetime/base.h>
#include <util/generic/algorithm.h>
#include <util/generic/hash.h>
#include <util/string/cast.h>

namespace {
    // Keys with dedicated root getters in Unified Agent PayloadAttributesProvider
    // (logbroker/unified_agent/plugins/yd_module/lib/attributes.*).
    bool IsTopLevelMetaKey(TStringBuf key) noexcept {
        return EqualToOneOf(
            key,
            "request_id",
            "user_id",
            "stackTrace",
            "threadName");
    }
} // namespace

TString FormatDeployJsonLogRecord(const TLogRecord& rec, TStringBuf loggerName) {
    NJsonWriter::TBuf buf;
    buf.BeginObject();
    buf.WriteKey("@timestamp").WriteString(TInstant::Now().ToString());
    buf.WriteKey("levelStr").WriteString(ToString(rec.Priority));

    TStringBuf message(rec.Data, rec.Len);
    message.ChopSuffix(TStringBuf("\n"));
    buf.WriteKey("message").WriteString(message);

    if (loggerName) {
        buf.WriteKey("loggerName").WriteString(loggerName);
    }

    // TLogElement::With appends MetaFlags, so keys may repeat; last value wins.
    THashMap<TStringBuf, TStringBuf> meta;
    for (const auto& [key, value] : rec.MetaFlags) {
        meta[key] = value;
    }

    for (const auto& [key, value] : meta) {
        if (IsTopLevelMetaKey(key)) {
            buf.WriteKey(key).WriteString(value);
        }
    }

    bool openedFields = false;
    for (const auto& [key, value] : meta) {
        if (key == "loggerName" || IsTopLevelMetaKey(key)) {
            continue;
        }
        if (!openedFields) {
            buf.WriteKey("@fields");
            buf.BeginObject();
            openedFields = true;
        }
        buf.WriteKey(key).WriteString(value);
    }
    if (openedFields) {
        buf.EndObject();
    }

    buf.EndObject();
    return buf.Str() + '\n';
}

TDeployJsonLogBackend::TDeployJsonLogBackend(
    THolder<TLogBackend> slave,
    TString loggerName)
    : Slave_(std::move(slave))
    , LoggerName_(std::move(loggerName))
{
}

void TDeployJsonLogBackend::WriteData(const TLogRecord& rec) {
    const TString formatted = FormatDeployJsonLogRecord(rec, LoggerName_);
    Slave_->WriteData(TLogRecord(rec.Priority, formatted.data(), formatted.size()));
}

void TDeployJsonLogBackend::ReopenLog() {
    Slave_->ReopenLog();
}
