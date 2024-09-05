#include "direct_io.h"

#include <util/generic/singleton.h>
#include <util/generic/yexception.h>
#include <util/system/info.h>
#include "align.h"

#ifdef _linux_
    #include <util/string/cast.h>
    #include <linux/version.h>
    #include <sys/utsname.h>
#endif

namespace {
    struct TAlignmentCalcer {
        inline TAlignmentCalcer()
            : Alignment(0)
        {
#ifdef _linux_
            utsname sysInfo;

            Y_ABORT_UNLESS(!uname(&sysInfo), "Error while call uname: %s", LastSystemErrorText());

            TStringBuf release(sysInfo.release);
            release = release.substr(0, release.find_first_not_of(".0123456789"));

            int v1 = FromString<int>(release.NextTok('.'));
            int v2 = FromString<int>(release.NextTok('.'));
            int v3 = FromString<int>(release.NextTok('.'));
            int linuxVersionCode = KERNEL_VERSION(v1, v2, v3);

            if (linuxVersionCode < KERNEL_VERSION(2, 4, 10)) {
                Alignment = 0;
            } else if (linuxVersionCode < KERNEL_VERSION(2, 6, 0)) {
                Alignment = NSystemInfo::GetPageSize();
            } else {
                // Default alignment used to be 512, but most modern devices rely on 4k physical blocks.
                // 4k alignment works well for both 512 and 4k blocks and doesn't require 512e support in the kernel.
                // See IGNIETFERRO-946.
                Alignment = 4096;
            }
#endif
        }

        size_t Alignment;
    };
} // namespace

TDirectIOBufferedFile::TDirectIOBufferedFile(const TString& path, EOpenMode oMode, size_t buflen /*= 1 << 17*/)
    : File(path, oMode)
    , Alignment(0)
    , DataLen(0)
    , ReadPosition(0)
    , WritePosition(0)
    , DirectIO(false)
{
    if (buflen == 0) {
        ythrow TFileError() << "unbuffered usage is not supported";
    }

    if (oMode & Direct) {
        Alignment = Singleton<TAlignmentCalcer>()->Alignment;
        SetDirectIO(true);
    }

    WritePosition = File.GetLength();
    FlushedBytes = WritePosition;
    FlushedToDisk = FlushedBytes;
    BufLen = (!!Alignment) ? AlignUp(buflen, Alignment) : buflen;
    BufferStorage.Resize(BufLen + Alignment);
    Buffer = (!!Alignment) ? AlignUp(BufferStorage.Data(), Alignment) : BufferStorage.Data();
}

#define DIRECT_IO_FLAGS (O_DIRECT | O_SYNC)

void TDirectIOBufferedFile::SetDirectIO(bool value) {
#ifdef _linux_
    if (DirectIO == value) {
        return;
    }

    if (!!Alignment && value) {
        (void)fcntl(File.GetHandle(), F_SETFL, fcntl(File.GetHandle(), F_GETFL) | DIRECT_IO_FLAGS);
    } else {
        (void)fcntl(File.GetHandle(), F_SETFL, fcntl(File.GetHandle(), F_GETFL) & ~DIRECT_IO_FLAGS);
    }

    DirectIO = value;
#else
    DirectIO = value;
#endif
}

TDirectIOBufferedFile::~TDirectIOBufferedFile() {
    try {
        Finish();
    } catch (...) {
    }
}

void TDirectIOBufferedFile::FlushData() {
    WriteToFile(Buffer, DataLen, FlushedBytes);
    DataLen = 0;
    File.FlushData();
}

void TDirectIOBufferedFile::Finish() {
    FlushData();
    File.Flush();
    File.Close();
}

void TDirectIOBufferedFile::Write(const void* buffer, size_t byteCount) {
    WriteToBuffer(buffer, byteCount, DataLen);
    WritePosition += byteCount;
}

void TDirectIOBufferedFile::WriteToBuffer(const void* buf, size_t len, ui64 position) {
    while (len > 0) {
        size_t writeLen = Min<size_t>(BufLen - position, len);

        if (writeLen > 0) {
            memcpy((char*)Buffer + position, buf, writeLen);
            buf = (char*)buf + writeLen;
            len -= writeLen;
            DataLen = (size_t)Max(position + writeLen, (ui64)DataLen);
            position += writeLen;
        }

        if (DataLen == BufLen) {
            WriteToFile(Buffer, DataLen, FlushedBytes);
            DataLen = 0;
            position = 0;
        }
    }
}

void TDirectIOBufferedFile::WriteToFile(const void* buf, size_t len, ui64 position) {
    if (!!len) {
        SetDirectIO(IsAligned(buf) && IsAligned(len) && IsAligned(position));

        File.Pwrite(buf, len, position);

        FlushedBytes = Max(FlushedBytes, position + len);
        FlushedToDisk = Min(FlushedToDisk, position);
    }
}

size_t TDirectIOBufferedFile::PreadSafe(void* buffer, size_t byteCount, ui64 offset) {
    if (FlushedToDisk < offset + byteCount) {
        File.FlushData();
        FlushedToDisk = FlushedBytes;
    }

#ifdef _linux_
    ssize_t bytesRead = 0;
    do {
        bytesRead = pread(File.GetHandle(), buffer, byteCount, offset);
    } while (bytesRead == -1 && errno == EINTR);

    if (bytesRead < 0) {
        ythrow yexception() << "error while pread file: " << LastSystemError() << "(" << LastSystemErrorText() << ")";
    }

    return bytesRead;
#else
    return File.Pread(buffer, byteCount, offset);
#endif
}

size_t TDirectIOBufferedFile::ReadFromFile(void* buffer, size_t byteCount, ui64 offset) {
    SetDirectIO(true);

    ui64 bytesRead = 0;

    while (byteCount) {
        if (!Alignment || IsAligned(buffer) && IsAligned(byteCount) && IsAligned(offset)) {
            if (const ui64 fromFile = PreadSafe(buffer, byteCount, offset)) {
                buffer = (char*)buffer + fromFile;
                byteCount -= fromFile;
                offset += fromFile;
                bytesRead += fromFile;
            } else {
                return bytesRead;
            }
        } else {
            break;
        }
    }

    if (!byteCount) {
        return bytesRead;
    }

    ui64 bufSize = AlignUp(Min<size_t>(BufferStorage.Size(), byteCount + (Alignment << 1)), Alignment);
    TBuffer readBufferStorage(bufSize + Alignment);
    char* readBuffer = AlignUp((char*)readBufferStorage.Data(), Alignment);

    while (byteCount) {
        ui64 begin = AlignDown(offset, (ui64)Alignment);
        ui64 end = AlignUp(offset + byteCount, (ui64)Alignment);
        ui64 toRead = Min(end - begin, bufSize);
        ui64 fromFile = PreadSafe(readBuffer, toRead, begin);

        if (!fromFile) {
            break;
        }

        ui64 delta = offset - begin;
        ui64 count = Min<ui64>(fromFile - delta, byteCount);

        memcpy(buffer, readBuffer + delta, count);
        buffer = (char*)buffer + count;
        byteCount -= count;
        offset += count;
        bytesRead += count;
    }
    return bytesRead;
}

size_t TDirectIOBufferedFile::Read(void* buffer, size_t byteCount) {
    size_t bytesRead = Pread(buffer, byteCount, ReadPosition);
    ReadPosition += bytesRead;
    return bytesRead;
}

size_t TDirectIOBufferedFile::Pread(void* buffer, size_t byteCount, ui64 offset) {
    if (!byteCount) {
        return 0;
    }

    size_t readFromFile = 0;
    if (offset < FlushedBytes) {
        readFromFile = Min<ui64>(byteCount, FlushedBytes - offset);
        size_t bytesRead = ReadFromFile(buffer, readFromFile, offset);
        if (bytesRead != readFromFile || readFromFile == byteCount) {
            return bytesRead;
        }
    }
    ui64 start = offset > FlushedBytes ? offset - FlushedBytes : 0;
    ui64 count = Min<ui64>(DataLen - start, byteCount - readFromFile);
    if (count) {
        memcpy((char*)buffer + readFromFile, (const char*)Buffer + start, count);
    }
    return count + readFromFile;
}

void TDirectIOBufferedFile::Pwrite(const void* buffer, size_t byteCount, ui64 offset) {
    if (offset > WritePosition) {
        ythrow yexception() << "cannot frite to position" << offset;
    }

    size_t writeToBufer = byteCount;
    size_t writeToFile = 0;

    if (FlushedBytes > offset) {
        writeToFile = Min<ui64>(byteCount, FlushedBytes - offset);
        WriteToFile(buffer, writeToFile, offset);
        writeToBufer -= writeToFile;
    }

    if (writeToBufer > 0) {
        ui64 bufferOffset = offset + writeToFile - FlushedBytes;
        WriteToBuffer((const char*)buffer + writeToFile, writeToBufer, bufferOffset);
    }
}
