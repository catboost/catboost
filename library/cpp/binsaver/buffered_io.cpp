#include "buffered_io.h"

i64 IBinaryStream::LongWrite(const void* userBuffer, i64 size) {
    Y_ABORT_UNLESS(size >= 0, "IBinaryStream::Write() called with a negative buffer size.");

    i64 leftToWrite = size;
    while (leftToWrite != 0) {
        int writeSz = static_cast<int>(Min<i64>(leftToWrite, std::numeric_limits<int>::max()));
        int written = WriteImpl(userBuffer, writeSz);
        Y_ASSERT(written <= writeSz);
        leftToWrite -= written;
        // Assumption: if WriteImpl(buf, writeSz) returns < writeSz, the stream is
        // full and there's no sense in continuing.
        if (written < writeSz)
            break;
    }
    Y_ASSERT(size >= leftToWrite);
    return size - leftToWrite;
}

i64 IBinaryStream::LongRead(void* userBuffer, i64 size) {
    Y_ABORT_UNLESS(size >= 0, "IBinaryStream::Read() called with a negative buffer size.");

    i64 leftToRead = size;
    while (leftToRead != 0) {
        int readSz = static_cast<int>(Min<i64>(leftToRead, std::numeric_limits<int>::max()));
        int read = ReadImpl(userBuffer, readSz);
        Y_ASSERT(read <= readSz);
        leftToRead -= read;
        // Assumption: if ReadImpl(buf, readSz) returns < readSz, the stream is
        // full and there's no sense in continuing.
        if (read < readSz) {
            memset(static_cast<char*>(userBuffer) + (size - leftToRead), 0, leftToRead);
            break;
        }
    }
    Y_ASSERT(size >= leftToRead);
    return size - leftToRead;
}
