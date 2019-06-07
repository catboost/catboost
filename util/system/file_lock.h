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

private:
    EFileLockType Type;
};
