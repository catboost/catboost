#pragma once

#include <library/cpp/logger/backend.h>
#include <library/cpp/logger/priority.h>
#include <library/cpp/logger/record.h>

#include <util/generic/ptr.h>
#include <util/generic/string.h>

// Formats a single log record as one Deploy JSON line (with trailing '\n').
// See https://deploy.yandex-team.ru/docs/logs/format
TString FormatDeployJsonLogRecord(
    const TLogRecord& rec,
    TStringBuf loggerName = {});

// Wraps a slave backend and rewrites each record as Deploy JSON.
//
// Must be a backend (not TLog::SetFormatter): MetaFlags are only available in
// TLogBackend::WriteData.
//
// Recommended stack (outer → inner):
//   Filter → Thread → TDeployJsonLogBackend → sink
class TDeployJsonLogBackend: public TLogBackend {
public:
    explicit TDeployJsonLogBackend(
        THolder<TLogBackend> slave,
        TString loggerName = {});

    void WriteData(const TLogRecord& rec) override;
    void ReopenLog() override;

private:
    THolder<TLogBackend> Slave_;
    TString LoggerName_;
};
