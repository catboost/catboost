#pragma once

#include <library/cpp/containers/stack_vector/stack_vec.h>
#include <library/cpp/cache/cache.h>
#include <library/cpp/deprecated/atomic/atomic.h>

#include <util/generic/noncopyable.h>
#include <util/generic/ptr.h>
#include <util/stream/output.h>

namespace NAllocProfiler {

struct TStats {
    intptr_t Allocs = 0;
    intptr_t Frees = 0;
    intptr_t CurrentSize = 0;

    void Clear()
    {
        Allocs = 0;
        Frees = 0;
        CurrentSize = 0;
    }

    void Alloc(size_t size)
    {
        AtomicIncrement(Allocs);
        AtomicAdd(CurrentSize, size);
    }

    void Free(size_t size)
    {
        AtomicIncrement(Frees);
        AtomicSub(CurrentSize, size);
    }
};

struct TAllocationInfo {
    int Tag;
    TStats Stats;
    TStackVec<void*, 64> Stack;

    void Clear() {
        Tag = 0;
        Stats.Clear();
        Stack.clear();
    }
};


class IAllocationStatsDumper {
public:
    virtual ~IAllocationStatsDumper() = default;

    // Total stats
    virtual void DumpTotal(const TStats& total) = 0;

    // Stats for individual stack
    virtual void DumpEntry(const TAllocationInfo& allocInfo) = 0;

    // App-specific tag printer
    virtual TString FormatTag(int tag);

    // Size printer (e.g. "10KB", "100MB", "over 9000")
    virtual TString FormatSize(intptr_t sz);
};

// Default implementation
class TAllocationStatsDumper: public IAllocationStatsDumper {
public:
    explicit TAllocationStatsDumper(IOutputStream& out);
    void DumpTotal(const TStats& total) override;
    void DumpEntry(const TAllocationInfo& allocInfo) override;

private:
    void FormatBackTrace(void* const* stack, size_t sz);

private:
    struct TSymbol {
        const void* Address;
        TString Name;
    };

    size_t PrintedCount;
    IOutputStream& Out;
    TLFUCache<void*, TSymbol> SymbolCache;
};

////////////////////////////////////////////////////////////////////////////////

class TAllocationStackCollector: private TNonCopyable {
private:
    class TImpl;
    THolder<TImpl> Impl;

public:
    TAllocationStackCollector();
    ~TAllocationStackCollector();

    int Alloc(void** stack, size_t frameCount, int tag, size_t size);
    void Free(int stackId, size_t size);

    void Clear();

    void Dump(int count, IAllocationStatsDumper& out) const;
};

}   // namespace NAllocProfiler
