#include "logging.h"

namespace NHnsw {

    class TCustomFuncLogger : public TLogBackend {
    public:
        TCustomFuncLogger(TCustomLoggingFunction func)
            : LoggerFunc(func)
        {
        }
        void WriteData(const TLogRecord& rec) override {
            LoggerFunc(rec.Data, rec.Len);
        }
        void ReopenLog() override {
        }

    private:
        TCustomLoggingFunction LoggerFunc = nullptr;
    };

    void SetCustomLoggingFunction(TCustomLoggingFunction loggingFunc) {
        THnswLog::Instance().ResetBackend(MakeHolder<TCustomFuncLogger>(loggingFunc));
    }

    void RestoreOriginalLogger() {
        THnswLog::Instance().ResetBackend(CreateLogBackend("cerr"));
    }
}
