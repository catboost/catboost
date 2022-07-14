#pragma once

#include "fwd.h"
#include "path.h"
#include <util/generic/string.h>

class TTempDir {
public:
    /// Create new directory in system tmp folder.
    TTempDir();

    /// Create new directory with this fixed name. If it already exists, clear it.
    TTempDir(const TString& tempDir);

    ~TTempDir();

    /// Create new directory in given folder.
    static TTempDir NewTempDir(const TString& root);

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
    struct TCreationToken {};

    // Prevent people from confusing this ctor with the public one
    // by requiring additional fake argument.
    TTempDir(const char* prefix, TCreationToken);

    TFsPath TempDir;
    bool Remove;
};
