#pragma once

#include <util/system/types.h>

namespace NAllocStats {

bool IsEnabled() noexcept;
void EnableAllocStats(bool enable) noexcept;
void IncThreadAllocStats(i64 size) noexcept;
void DecThreadAllocStats(i64 size) noexcept;
void ResetThreadAllocStats() noexcept;
i64 GetThreadAllocMax() noexcept;
void IncLiveLockCounter() noexcept;
ui64 GetLiveLockCounter() noexcept;
void IncMmapCounter(ui64 amount) noexcept;
ui64 GetMmapCounter() noexcept;

}  // namespace NAllocStats
