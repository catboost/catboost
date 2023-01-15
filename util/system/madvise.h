#pragma once

#include "defaults.h"

/// see linux madvise(MADV_SEQUENTIAL)
void MadviseSequentialAccess(const void* begin, size_t size);

/// see linux madvise(MADV_RANDOM)
void MadviseRandomAccess(const void* begin, size_t size);

/// see linux madvise(MADV_DONTNEED)
void MadviseEvict(const void* begin, size_t size);

/// see linux madvise(MADV_DONTDUMP)
void MadviseExcludeFromCoreDump(const void* begin, size_t size);
