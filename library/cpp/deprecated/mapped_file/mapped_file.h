#pragma once

#include <util/generic/flags.h>
#include <util/generic/ptr.h>
#include <util/generic/string.h>
#include <util/generic/utility.h>
#include <util/generic/yexception.h>
#include <util/system/align.h>
#include <util/system/file.h>
#include <util/system/filemap.h>
#include <util/system/yassert.h>

#include <cstdio>
#include <new>

/// Deprecated (by pg@), use TFileMap or TMemoryMap instead
class TMappedFile {
private:
    TFileMap* Map_;

private:
    TMappedFile(TFileMap* map, const char* dbgName);

public:
    TMappedFile() {
        Map_ = nullptr;
    }

    ~TMappedFile() {
        term();
    }

    explicit TMappedFile(const TString& name) {
        Map_ = nullptr;
        init(name, TFileMap::oRdOnly);
    }

    TMappedFile(const TFile& file, TFileMap::EOpenMode om = TFileMap::oRdOnly, const char* dbgName = "unknown");

    void init(const TString& name);

    void init(const TString& name, TFileMap::EOpenMode om);

    void init(const TString& name, size_t length, TFileMap::EOpenMode om);

    void init(const TFile&, TFileMap::EOpenMode om = TFileMap::oRdOnly, const char* dbgName = "unknown");

    void flush();

    void term() {
        if (Map_) {
            Map_->Unmap();
            delete Map_;
            Map_ = nullptr;
        }
    }

    size_t getSize() const {
        return (Map_ ? Map_->MappedSize() : 0);
    }

    void* getData(size_t pos = 0) const {
        Y_ASSERT(!Map_ || (pos <= getSize()));
        return (Map_ ? (void*)((unsigned char*)Map_->Ptr() + pos) : nullptr);
    }

    void precharge(size_t pos = 0, size_t size = (size_t)-1) const;

    void swap(TMappedFile& file) noexcept {
        DoSwap(Map_, file.Map_);
    }
};
