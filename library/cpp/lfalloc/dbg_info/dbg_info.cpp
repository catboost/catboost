#include "dbg_info.h"

#include <library/cpp/malloc/api/malloc.h>

namespace NAllocDbg {
    ////////////////////////////////////////////////////////////////////////////////

    using TGetAllocationCounter = i64(int counter);

    using TSetThreadAllocTag = int(int tag);
    using TGetPerTagAllocInfo = void(
        bool flushPerThreadCounters,
        TPerTagAllocInfo* info,
        int& maxTag,
        int& numSizes);

    using TSetProfileCurrentThread = bool(bool newVal);
    using TSetProfileAllThreads = bool(bool newVal);
    using TSetAllocationSamplingEnabled = bool(bool newVal);

    using TSetAllocationSampleRate = size_t(size_t newVal);
    using TSetAllocationSampleMaxSize = size_t(size_t newVal);

    using TSetAllocationCallback = TAllocationCallback*(TAllocationCallback* newVal);
    using TSetDeallocationCallback = TDeallocationCallback*(TDeallocationCallback* newVal);

    struct TAllocFn {
        TGetAllocationCounter* GetAllocationCounterFast = nullptr;
        TGetAllocationCounter* GetAllocationCounterFull = nullptr;

        TSetThreadAllocTag* SetThreadAllocTag = nullptr;
        TGetPerTagAllocInfo* GetPerTagAllocInfo = nullptr;

        TSetProfileCurrentThread* SetProfileCurrentThread = nullptr;
        TSetProfileAllThreads* SetProfileAllThreads = nullptr;
        TSetAllocationSamplingEnabled* SetAllocationSamplingEnabled = nullptr;

        TSetAllocationSampleRate* SetAllocationSampleRate = nullptr;
        TSetAllocationSampleMaxSize* SetAllocationSampleMaxSize = nullptr;

        TSetAllocationCallback* SetAllocationCallback = nullptr;
        TSetDeallocationCallback* SetDeallocationCallback = nullptr;

        TAllocFn() {
            auto mallocInfo = NMalloc::MallocInfo();

            GetAllocationCounterFast = (TGetAllocationCounter*)mallocInfo.GetParam("GetLFAllocCounterFast");
            GetAllocationCounterFull = (TGetAllocationCounter*)mallocInfo.GetParam("GetLFAllocCounterFull");

            SetThreadAllocTag = (TSetThreadAllocTag*)mallocInfo.GetParam("SetThreadAllocTag");
            GetPerTagAllocInfo = (TGetPerTagAllocInfo*)mallocInfo.GetParam("GetPerTagAllocInfo");

            SetProfileCurrentThread = (TSetProfileCurrentThread*)mallocInfo.GetParam("SetProfileCurrentThread");
            SetProfileAllThreads = (TSetProfileAllThreads*)mallocInfo.GetParam("SetProfileAllThreads");
            SetAllocationSamplingEnabled = (TSetAllocationSamplingEnabled*)mallocInfo.GetParam("SetAllocationSamplingEnabled");

            SetAllocationSampleRate = (TSetAllocationSampleRate*)mallocInfo.GetParam("SetAllocationSampleRate");
            SetAllocationSampleMaxSize = (TSetAllocationSampleMaxSize*)mallocInfo.GetParam("SetAllocationSampleMaxSize");

            SetAllocationCallback = (TSetAllocationCallback*)mallocInfo.GetParam("SetAllocationCallback");
            SetDeallocationCallback = (TSetDeallocationCallback*)mallocInfo.GetParam("SetDeallocationCallback");
        }
    };

    ////////////////////////////////////////////////////////////////////////////////

    static TAllocFn AllocFn;

    i64 GetAllocationCounterFast(ELFAllocCounter counter) {
        return AllocFn.GetAllocationCounterFast ? AllocFn.GetAllocationCounterFast(counter) : 0;
    }

    i64 GetAllocationCounterFull(ELFAllocCounter counter) {
        return AllocFn.GetAllocationCounterFull ? AllocFn.GetAllocationCounterFull(counter) : 0;
    }

    int SetThreadAllocTag(int tag) {
        return AllocFn.SetThreadAllocTag ? AllocFn.SetThreadAllocTag(tag) : 0;
    }

    TArrayPtr<TPerTagAllocInfo> GetPerTagAllocInfo(
        bool flushPerThreadCounters,
        int& maxTag,
        int& numSizes) {
        if (AllocFn.GetPerTagAllocInfo) {
            AllocFn.GetPerTagAllocInfo(flushPerThreadCounters, nullptr, maxTag, numSizes);
            TArrayPtr<TPerTagAllocInfo> info = new TPerTagAllocInfo[maxTag * numSizes];
            AllocFn.GetPerTagAllocInfo(flushPerThreadCounters, info.Get(), maxTag, numSizes);
            return info;
        }
        maxTag = 0;
        numSizes = 0;
        return nullptr;
    }

    bool SetProfileCurrentThread(bool newVal) {
        return AllocFn.SetProfileCurrentThread ? AllocFn.SetProfileCurrentThread(newVal) : false;
    }

    bool SetProfileAllThreads(bool newVal) {
        return AllocFn.SetProfileAllThreads ? AllocFn.SetProfileAllThreads(newVal) : false;
    }

    bool SetAllocationSamplingEnabled(bool newVal) {
        return AllocFn.SetAllocationSamplingEnabled ? AllocFn.SetAllocationSamplingEnabled(newVal) : false;
    }

    size_t SetAllocationSampleRate(size_t newVal) {
        return AllocFn.SetAllocationSampleRate ? AllocFn.SetAllocationSampleRate(newVal) : 0;
    }

    size_t SetAllocationSampleMaxSize(size_t newVal) {
        return AllocFn.SetAllocationSampleMaxSize ? AllocFn.SetAllocationSampleMaxSize(newVal) : 0;
    }

    TAllocationCallback* SetAllocationCallback(TAllocationCallback* newVal) {
        return AllocFn.SetAllocationCallback ? AllocFn.SetAllocationCallback(newVal) : nullptr;
    }

    TDeallocationCallback* SetDeallocationCallback(TDeallocationCallback* newVal) {
        return AllocFn.SetDeallocationCallback ? AllocFn.SetDeallocationCallback(newVal) : nullptr;
    }

}
