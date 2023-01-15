#include "murmur.h"

#include <util/system/unaligned_mem.h>

size_t MurmurHashSizeT(const char* buf, size_t len) noexcept {
    return MurmurHash<size_t>(buf, len);
}
