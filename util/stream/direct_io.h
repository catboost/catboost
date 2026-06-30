#pragma once

#include "input.h"
#include "output.h"
#include <util/system/direct_io.h>

class TRandomAccessFileInput: public IInputStream {
public:
    TRandomAccessFileInput(TDirectIOBufferedFile& file Y_LIFETIME_BOUND, ui64 position);

protected:
    size_t DoRead(void* buf, size_t len) override;
    size_t DoSkip(size_t len) override;

private:
    TDirectIOBufferedFile& File;
    ui64 Position;
};

class TRandomAccessFileOutput: public IOutputStream {
public:
    TRandomAccessFileOutput(TDirectIOBufferedFile& file Y_LIFETIME_BOUND);

    TRandomAccessFileOutput(TRandomAccessFileOutput&&) noexcept = default;
    TRandomAccessFileOutput& operator=(TRandomAccessFileOutput&&) noexcept = default;

protected:
    TDirectIOBufferedFile* File;

private:
    void DoWrite(const void* buf, size_t len) override;
    void DoFlush() override;
};

class TBufferedFileOutputEx: public TRandomAccessFileOutput {
public:
    TBufferedFileOutputEx(const TString& path, EOpenMode oMode, size_t buflen = 1 << 17);

private:
    void DoFlush() override;
    void DoFinish() override;
    THolder<TDirectIOBufferedFile> FileHolder;
};
