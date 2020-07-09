#pragma once

#include "log.h"
#include "backend.h"
#include "priority.h"

#define YSYSLOG(priority, ...) SysLogInstance().AddLog((priority), __VA_ARGS__)
#define YSYSLOGINIT_FLAGS(ident, facility, flags)                                                                                    \
    struct TLogIniter {                                                                                                              \
        TLogIniter() {                                                                                                               \
            SysLogInstance().ResetBackend(THolder<TLogBackend>(                                                                      \
                (ident) ? (TLogBackend*)(new TSysLogBackend((ident), (facility), (flags))) : (TLogBackend*)(new TNullLogBackend())));\
        }                                                                                                                            \
    } Y_CAT(loginit, __LINE__);

#define YSYSLOGINIT(ident, facility) YSYSLOGINIT_FLAGS((ident), (facility), 0)

class TSysLogBackend: public TLogBackend {
public:
    enum EFacility {
        TSYSLOG_LOCAL0 = 0,
        TSYSLOG_LOCAL1 = 1,
        TSYSLOG_LOCAL2 = 2,
        TSYSLOG_LOCAL3 = 3,
        TSYSLOG_LOCAL4 = 4,
        TSYSLOG_LOCAL5 = 5,
        TSYSLOG_LOCAL6 = 6,
        TSYSLOG_LOCAL7 = 7
    };

    enum EFlags {
        LogPerror = 1,
        LogCons = 2
    };

    TSysLogBackend(const char* ident, EFacility facility, int flags = 0);
    ~TSysLogBackend() override;

    void WriteData(const TLogRecord& rec) override;
    void ReopenLog() override;

    virtual TString GetIdent() const {
        return Ident;
    }

    virtual EFacility GetFacility() const {
        return Facility;
    }

    virtual int GetFlags() const {
        return Flags;
    }

protected:
    int ELogPriority2SyslogPriority(ELogPriority priority);

    TString Ident;
    EFacility Facility;
    int Flags;
};

/*
 * return system-wide logger instance
 * better do not use in real programs(instead of robot, of course)
 */
TLog& SysLogInstance();
