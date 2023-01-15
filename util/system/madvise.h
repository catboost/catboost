#pragma once

#include "defaults.h"

#include <util/generic/array_ref.h>

/// see linux madvise(MADV_SEQUENTIAL)
void MadviseSequentialAccess(const void* begin, size_t size);
void MadviseSequentialAccess(TArrayRef<const char> data);
void MadviseSequentialAccess(TArrayRef<const ui8> data);

/// see linux madvise(MADV_RANDOM)
void MadviseRandomAccess(const void* begin, size_t size);
void MadviseRandomAccess(TArrayRef<const char> data);
void MadviseRandomAccess(TArrayRef<const ui8> data);

/// see linux madvise(MADV_DONTNEED)
void MadviseEvict(const void* begin, size_t size);
void MadviseEvict(TArrayRef<const char> data);
void MadviseEvict(TArrayRef<const ui8> data);

/// see linux madvise(MADV_DONTDUMP)
void MadviseExcludeFromCoreDump(const void* begin, size_t size);
void MadviseExcludeFromCoreDump(TArrayRef<const char> data);
void MadviseExcludeFromCoreDump(TArrayRef<const ui8> data);

/// see linux madvise(MADV_DODUMP)
void MadviseIncludeIntoCoreDump(const void* begin, size_t size);
void MadviseIncludeIntoCoreDump(TArrayRef<const char> data);
void MadviseIncludeIntoCoreDump(TArrayRef<const ui8> data);
