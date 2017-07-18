#pragma once

#include "file.h"

#include <util/generic/noncopyable.h>

class TString;

struct TFileLock: public TFile {
    TFileLock(const TString& filename);

    void Acquire();
    bool TryAcquire();
    void Release();
};
