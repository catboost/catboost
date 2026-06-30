#include <new>
#include <stdio.h>
#include <stdlib.h>

#include "alloc.h"
#include "enable.h"
#include <util/system/platform.h>

namespace NAllocSetup {
    size_t softLimit = size_t(-1);
    size_t hardLimit = size_t(-1);
    size_t allocationThreshold = size_t(-1);
    size_t softReclaimDivisor = 100;
    size_t angryReclaimDivisor = 100;

    struct TThrowInfo {
        size_t CurrSize;
        size_t MaxSize;
    };
#if defined(_unix_) && !defined(_darwin_)
    __thread TThrowInfo info;
    void ThrowOnError(size_t allocSize) {
        info.CurrSize += allocSize;
        if (info.MaxSize && info.MaxSize < info.CurrSize) {
#ifndef NDEBUG
            __builtin_trap();
#endif
            info.CurrSize = 0;
            throw std::bad_alloc();
        }
    }
    void SetThrowConditions(size_t currSize, size_t maxSize) {
        info.CurrSize = currSize;
        info.MaxSize = maxSize;
    }
#else  // _unix_ && ! _darwin_
    void ThrowOnError(size_t /*allocSize*/) {
    }
    void SetThrowConditions(size_t /*currSize*/, size_t /*maxSize*/) {
    }
#endif // _unix_ && ! _darwin_

    void SetSoftLimit(size_t softLimit_) {
        softLimit = softLimit_;
    }
    void SetHardLimit(size_t hardLimit_) {
        hardLimit = hardLimit_;
    }
    void SetAllocationThreshold(size_t allocationThreshold_) {
        allocationThreshold = allocationThreshold_;
    }
    void SetSoftReclaimDivisor(size_t softReclaimDivisor_) {
        softReclaimDivisor = softReclaimDivisor_;
    }
    void SetAngryReclaimDivisor(size_t angryReclaimDivisor_) {
        angryReclaimDivisor = angryReclaimDivisor_;
    }
    size_t GetSoftLimit() {
        return softLimit;
    }
    size_t GetHardLimit() {
        return hardLimit;
    }
    size_t GetAllocationThreshold() {
        return allocationThreshold;
    }
    size_t GetSoftReclaimDivisor() {
        return softReclaimDivisor;
    }
    size_t GetAngryReclaimDivisor() {
        return angryReclaimDivisor;
    }

    size_t allocSize;
    size_t totalAllocSize;
    size_t gcSize;

    size_t GetTotalAllocSize() {
        return totalAllocSize;
    }
    size_t GetCurSize() {
        return allocSize;
    }
    size_t GetGCSize() {
        return gcSize;
    }

    bool CanAlloc(size_t allocSize_, size_t totalAllocSize_) {
        allocSize = allocSize_;
        totalAllocSize = totalAllocSize_;
        return allocSize_ < hardLimit || totalAllocSize_ < allocationThreshold;
    }
    bool NeedReclaim(size_t gcSize_, size_t counter) {
        gcSize = gcSize_;
        size_t limit = gcSize_ < softLimit ? softReclaimDivisor : angryReclaimDivisor;
        return counter > limit;
    }

    bool IsEnabledByDefault() {
        return EnableByDefault;
    }
}
