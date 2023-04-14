#pragma once

#include "backend.h"

#include <util/generic/string.h>


class IOutputStream;

class TStreamLogBackend : public TLogBackend {
public:
    explicit TStreamLogBackend(IOutputStream* slave);
    ~TStreamLogBackend() override;

    void WriteData(const TLogRecord& rec) override;
    void ReopenLog() override;

private:
    IOutputStream* Slave_;
};

class TStreamWithContextLogBackend : public TLogBackend {
private:
    static constexpr TStringBuf DELIMITER = "; ";

public:
    explicit TStreamWithContextLogBackend(IOutputStream* slave);
    ~TStreamWithContextLogBackend() override;

    void WriteData(const TLogRecord& rec) override;
    void ReopenLog() override;

private:
    IOutputStream* Slave_;
};
