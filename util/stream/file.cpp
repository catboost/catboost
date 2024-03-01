#include "file.h"

#include <util/memory/blob.h>
#include <util/generic/yexception.h>

TUnbufferedFileInput::TUnbufferedFileInput(const char* path)
    : TUnbufferedFileInput(TFile(path, OPEN_MODE))
{
}

TUnbufferedFileInput::TUnbufferedFileInput(const TString& path)
    : TUnbufferedFileInput(TFile(path, OPEN_MODE))
{
}

TUnbufferedFileInput::TUnbufferedFileInput(const std::filesystem::path& path)
    : TUnbufferedFileInput(TFile(path, OPEN_MODE))
{
}

TUnbufferedFileInput::TUnbufferedFileInput(const TFile& file)
    : File_(file)
{
    if (!File_.IsOpen()) {
        ythrow TIoException() << "file (" << file.GetName() << ") not open";
    }
}

size_t TUnbufferedFileInput::DoRead(void* buf, size_t len) {
    return File_.ReadOrFail(buf, len);
}

size_t TUnbufferedFileInput::DoSkip(size_t len) {
    if (len < 384) {
        /* Base implementation calls DoRead, which results in one system call
         * instead of three as in fair skip implementation. For small sizes
         * actually doing one read is cheaper. Experiments show that the
         * border that separates two implementations performance-wise lies
         * in the range of 384-512 bytes (assuming that the file is in OS cache). */
        return IInputStream::DoSkip(len);
    }

    /* TFile::Seek can seek beyond the end of file, so we need to do
     * size check here. */
    i64 size = File_.GetLength();
    i64 oldPos = File_.GetPosition();
    i64 newPos = File_.Seek(Min<i64>(size, oldPos + len), sSet);

    return newPos - oldPos;
}

TUnbufferedFileOutput::TUnbufferedFileOutput(const char* path)
    : TUnbufferedFileOutput(TFile(path, OPEN_MODE))
{
}

TUnbufferedFileOutput::TUnbufferedFileOutput(const TString& path)
    : TUnbufferedFileOutput(TFile(path, OPEN_MODE))
{
}

TUnbufferedFileOutput::TUnbufferedFileOutput(const std::filesystem::path& path)
    : TUnbufferedFileOutput(TFile(path, OPEN_MODE))
{
}

TUnbufferedFileOutput::TUnbufferedFileOutput(const TFile& file)
    : File_(file)
{
    if (!File_.IsOpen()) {
        ythrow TIoException() << "closed file(" << file.GetName() << ") passed";
    }
}

TUnbufferedFileOutput::~TUnbufferedFileOutput() = default;

void TUnbufferedFileOutput::DoWrite(const void* buf, size_t len) {
    File_.Write(buf, len);
}

void TUnbufferedFileOutput::DoFlush() {
    if (File_.IsOpen()) {
        File_.Flush();
    }
}

class TMappedFileInput::TImpl: public TBlob {
public:
    inline TImpl(const TFile& file)
        : TBlob(TBlob::FromFile(file))
    {
    }

    inline ~TImpl() = default;
};

TMappedFileInput::TMappedFileInput(const TFile& file)
    : TMemoryInput(nullptr, 0)
    , Impl_(new TImpl(file))
{
    Reset(Impl_->Data(), Impl_->Size());
}

TMappedFileInput::TMappedFileInput(const TString& path)
    : TMemoryInput(nullptr, 0)
    , Impl_(new TImpl(TFile(path, OpenExisting | RdOnly)))
{
    Reset(Impl_->Data(), Impl_->Size());
}

TMappedFileInput::~TMappedFileInput() = default;
