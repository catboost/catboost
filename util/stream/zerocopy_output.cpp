#include "zerocopy_output.h"

#include <util/generic/utility.h>

void IZeroCopyOutput::DoWrite(const void* buf, size_t len) {
    void* ptr = nullptr;
    size_t writtenBytes = 0;
    while (writtenBytes < len) {
        size_t bufferSize = DoNext(&ptr);
        Y_ASSERT(ptr && bufferSize > 0);
        size_t toWrite = Min(bufferSize, len - writtenBytes);
        memcpy(ptr, static_cast<const char*>(buf) + writtenBytes, toWrite);
        writtenBytes += toWrite;
        if (toWrite < bufferSize) {
            DoUndo(bufferSize - toWrite);
        }
    }
}
