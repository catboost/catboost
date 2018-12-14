#include "direct_io.h"

#include <util/generic/utility.h>

size_t TRandomAccessFileInput::DoRead(void* buf, size_t len) {
    const size_t result = File.Pread(buf, len, Position);
    Position += result;
    return result;
}

TRandomAccessFileInput::TRandomAccessFileInput(TDirectIOBufferedFile& file, ui64 position)
    : File(file)
    , Position(position)
{
}

size_t TRandomAccessFileInput::DoSkip(size_t len) {
    size_t skiped = Min(len, (size_t)Min((ui64)Max<size_t>(), File.GetLength() - Position));
    Position += skiped;
    return skiped;
}

TRandomAccessFileOutput::TRandomAccessFileOutput(TDirectIOBufferedFile& file)
    : File(&file)
{
}

void TRandomAccessFileOutput::DoWrite(const void* buf, size_t len) {
    File->Write(buf, len);
}

void TRandomAccessFileOutput::DoFlush() {
    File->FlushData();
}

TBufferedFileOutputEx::TBufferedFileOutputEx(const TString& path, EOpenMode oMode, size_t buflen)
    : TRandomAccessFileOutput(*(new TDirectIOBufferedFile(path, oMode, buflen)))
    , FileHolder(File)
{
}

void TBufferedFileOutputEx::DoFinish() {
    FileHolder->Finish();
}

void TBufferedFileOutputEx::DoFlush() {
}
