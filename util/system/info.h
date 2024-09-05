#pragma once

#include "defaults.h"

namespace NSystemInfo {
    size_t NumberOfCpus();
    size_t NumberOfMillicores();
    size_t CachedNumberOfCpus();
    size_t CachedNumberOfMillicores();
    size_t LoadAverage(double* la, size_t len);
    size_t GetPageSize() noexcept;
    size_t TotalMemorySize();
    size_t MaxOpenFiles();
} // namespace NSystemInfo
