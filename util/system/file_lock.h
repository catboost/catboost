#pragma once

#include "file.h"

#include <util/generic/noncopyable.h>

class TString;

enum class EFileLockType{
    Exclusive,
    Shared
};

class TFileLock: public TFile {
public:
    TFileLock(const TString& filename, const EFileLockType type = EFileLockType::Exclusive);

    void Acquire();
    bool TryAcquire();
    void Release();

    inline void lock() {
        Acquire();
    }

    inline bool try_lock() {
        return TryAcquire();
    }

    inline void unlock() {
        Release();
    }

private:
    EFileLockType Type;
};
