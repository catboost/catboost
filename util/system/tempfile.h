#pragma once

#include "fs.h"
#include "file.h"

#include <util/generic/string.h>

class TTempFile {
public:
    inline TTempFile(const TString& fname)
        : Name_(fname)
    {
    }

    inline ~TTempFile() {
        NFs::Remove(Name());
    }

    inline const TString& Name() const noexcept {
        return Name_;
    }

private:
    const TString Name_;
};

class TTempFileHandle: public TTempFile, public TFile {
public:
    TTempFileHandle();
    TTempFileHandle(const TString& fname);

private:
    TFile CreateFile() const;
};
