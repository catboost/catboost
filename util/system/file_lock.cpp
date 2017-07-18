#include "file_lock.h"
#include "flock.h"

#include <util/generic/string.h>
#include <util/generic/yexception.h>

#include <errno.h>

TFileLock::TFileLock(const TString& filename)
    : TFile(filename, OpenAlways | RdWr)
{
}

void TFileLock::Acquire() {
    Flock(LOCK_EX);
}

bool TFileLock::TryAcquire() {
    try {
        Flock(LOCK_EX | LOCK_NB);
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
