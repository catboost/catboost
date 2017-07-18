#pragma once

#include "path.h"
#include <util/generic/string.h>

class TTempDir {
public:
    TTempDir();
    TTempDir(const TString& tempDir);
    ~TTempDir();

    const TString& operator()() const {
        return Name();
    }

    const TString& Name() const {
        return TempDir.GetPath();
    }

    const TFsPath& Path() const {
        return TempDir;
    }

    void DoNotRemove();

private:
    TFsPath TempDir;
    bool Remove;
};
