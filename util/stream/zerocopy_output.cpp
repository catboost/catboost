#include "zerocopy_output.h"

#include <util/generic/utility.h>

void IZeroCopyOutput::DoWrite(const void* buf, size_t len) {
    void* ptr;
    while (len > 0) {
        size_t toWrite = Min(DoNext(&ptr), len);
        Y_ASSERT(ptr && toWrite > 0);
        memcpy(ptr, buf, toWrite);
        DoAdvance(toWrite);
        len -= toWrite;
    }
}
