#include "composite.h"
#include "uninitialized_creator.h"

void TCompositeLogBackend::WriteData(const TLogRecord& rec) {
    for (auto& slave: Slaves) {
        slave->WriteData(rec);
    }
}

void TCompositeLogBackend::ReopenLog() {
    for (auto& slave : Slaves) {
        slave->ReopenLog();
    }
}

void TCompositeLogBackend::AddLogBackend(THolder<TLogBackend>&& backend) {
    LogPriority = Max(LogPriority, backend->FiltrationLevel());
    Slaves.emplace_back(std::move(backend));
}
