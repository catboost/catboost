#include "backend.h"
#include <util/generic/vector.h>
#include <util/system/mutex.h>
#include <util/generic/singleton.h>
#include <util/generic/yexception.h>

namespace {
    class TGlobalLogsStorage {
    private:
        TVector<TLogBackend*> Backends;
        TMutex Mutex;

    public:
        void Register(TLogBackend* backend) {
            TGuard<TMutex> g(Mutex);
            Backends.push_back(backend);
        }

        void UnRegister(TLogBackend* backend) {
            TGuard<TMutex> g(Mutex);
            for (ui32 i = 0; i < Backends.size(); ++i) {
                if (Backends[i] == backend) {
                    Backends.erase(Backends.begin() + i);
                    return;
                }
            }
            Y_ABORT("Incorrect pointer for log backend");
        }

        void Reopen(bool flush) {
            TGuard<TMutex> g(Mutex);
            for (auto& b : Backends) {
                if (typeid(*b) == typeid(TLogBackend)) {
                    continue;
                }
                if (flush) {
                    b->ReopenLog();
                } else {
                    b->ReopenLogNoFlush();
                }
            }
        }
    };
}

template <>
class TSingletonTraits<TGlobalLogsStorage> {
public:
    static const size_t Priority = 50;
};

ELogPriority TLogBackend::FiltrationLevel() const {
    return LOG_MAX_PRIORITY;
}

TLogBackend::TLogBackend() noexcept {
    Singleton<TGlobalLogsStorage>()->Register(this);
}

TLogBackend::~TLogBackend() {
    Singleton<TGlobalLogsStorage>()->UnRegister(this);
}

void TLogBackend::ReopenLogNoFlush() {
    ReopenLog();
}

void TLogBackend::ReopenAllBackends(bool flush) {
    Singleton<TGlobalLogsStorage>()->Reopen(flush);
}

size_t TLogBackend::QueueSize() const {
    ythrow yexception() << "Not implemented.";
}
