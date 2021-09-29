#include "file_lock.h"
#include "flock.h"

#include <util/generic/yexception.h>

#include <cerrno>

namespace {
    int GetMode(const EFileLockType type) {
        switch (type) {
            case EFileLockType::Exclusive:
                return LOCK_EX;
            case EFileLockType::Shared:
                return LOCK_SH;
            default:
                Y_UNREACHABLE();
        }
        Y_UNREACHABLE();
    }
}

TFileLock::TFileLock(const TString& filename, const EFileLockType type)
    : TFile(filename, OpenAlways | RdOnly)
    , Type(type)
{
}

void TFileLock::Acquire() {
    Flock(GetMode(Type));
}

bool TFileLock::TryAcquire() {
    try {
        Flock(GetMode(Type) | LOCK_NB);
        return true;
    } catch (const TSystemError& e) {
        if (e.Status() != EWOULDBLOCK) {
            throw;
        }
        return false;
    }
}

void TFileLock::Release() {
    Flock(LOCK_UN);
}
