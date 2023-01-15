#pragma once

#include <library/cpp/logger/element.h>
#include <library/cpp/logger/log.h>
#include <library/cpp/logger/priority.h>

namespace NHnsw {
    using TCustomLoggingFunction = void(*)(const char*, size_t len);

    class THnswLog : public TLog {
        THnswLog() : TLog(CreateLogBackend("cerr")) {}

    public:
        static THnswLog& Instance() {
            static THnswLog instance;
            return instance;
        }
    };

    void SetCustomLoggingFunction(TCustomLoggingFunction loggingFunc);
    void RestoreOriginalLogger();

    #define HNSW_LOG TLogElement(&THnswLog::Instance(), TLOG_INFO)
}
