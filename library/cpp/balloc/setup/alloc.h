#pragma once

#include <stddef.h>

namespace NAllocSetup {
    void ThrowOnError(size_t allocSize);
    void SetThrowConditions(size_t currSize, size_t maxSize);
    void SetSoftLimit(size_t softLimit);
    void SetHardLimit(size_t hardLimit);
    void SetAllocationThreshold(size_t allocationThreshold);
    void SetSoftReclaimDivisor(size_t softReclaimDivisor);
    void SetAngryReclaimDivisor(size_t angryReclaimDivisor);
    bool CanAlloc(size_t allocSize, size_t totalAllocSize);
    bool NeedReclaim(size_t gcSize_, size_t counter);
    size_t GetTotalAllocSize();
    size_t GetCurSize();
    size_t GetGCSize();

    size_t GetSoftLimit();
    size_t GetHardLimit();
    size_t GetAllocationThreshold();
    size_t GetSoftReclaimDivisor();
    size_t GetAngryReclaimDivisor();

    bool IsEnabledByDefault();

    struct TAllocGuard {
        TAllocGuard(size_t maxSize) {
            SetThrowConditions(0, maxSize);
        }
        ~TAllocGuard() {
            SetThrowConditions(0, 0);
        }
    };
}
