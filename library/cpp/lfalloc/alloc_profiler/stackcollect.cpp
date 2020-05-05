#include "stackcollect.h"

#include "profiler.h"

#include <util/generic/algorithm.h>
#include <util/generic/vector.h>
#include <util/stream/format.h>
#include <util/stream/str.h>
#include <util/string/cast.h>
#include <util/string/printf.h>
#include <util/system/backtrace.h>
#include <util/system/spinlock.h>
#include <util/system/yassert.h>


namespace NAllocProfiler {

////////////////////////////////////////////////////////////////////////////////

template <typename T>
class TStackCollector: private TNonCopyable {
public:
    struct TFrameInfo {
        int PrevInd;
        void* Addr;
        int Tag;
        T Stats;

        void Clear()
        {
            PrevInd = 0;
            Addr = nullptr;
            Tag = 0;
            Stats.Clear();
        }
    };

private:
    static const size_t STACKS_HASH_MAP_SIZE = 256 * 1024;
    TFrameInfo Frames[STACKS_HASH_MAP_SIZE];

    ui64 Samples;            // Saved samples count
    ui64 UniqueSamples;      // Number of unique addresses
    ui64 UsedSlots;          // Number of occupied slots in the hashtable
    ui64 DroppedSamples;     // Number of unsaved addresses
    ui64 SearchSkipCount;    // Total number of linear hash table probes due to collisions

    TAdaptiveLock Lock;

public:
    TStackCollector()
    {
        Clear();
    }

    int AddStack(void** stack, size_t frameCount, int tag)
    {
        Y_ASSERT(frameCount > 0);

        int prevInd = -1;
        with_lock (Lock) {
            for (int i = frameCount - 1; i >= 0; --i) {
                prevInd = AddFrame(stack[i], prevInd, ((i == 0) ? tag : 0), (i == 0));
                if (prevInd == -1) {
                    break;
                }
            }
        }
        return prevInd;
    }

    T& GetStats(int stackId)
    {
        Y_ASSERT(stackId >= 0 && (size_t)stackId < Y_ARRAY_SIZE(Frames));
        Y_ASSERT(!IsSlotEmpty(stackId));

        return Frames[stackId].Stats;
    }

    const TFrameInfo* GetFrames() const
    {
        return Frames;
    }

    size_t GetFramesCount() const
    {
        return Y_ARRAY_SIZE(Frames);
    }

    void BackTrace(const TFrameInfo* stack, TStackVec<void*, 64>& frames) const
    {
        frames.clear();
        for (size_t i = 0; i < 100; ++i) {
            frames.push_back(stack->Addr);
            int prevInd = stack->PrevInd;
            if (prevInd == -1) {
                break;
            }
            stack = &Frames[prevInd];
        }
    }

    void Clear()
    {
        for (auto& frame: Frames) {
            frame.Clear();
        }

        Samples = 0;
        DroppedSamples = 0;
        UniqueSamples = 0;
        UsedSlots = 0;
        SearchSkipCount = 0;
    }

private:
    // Hash function applied to the addresses
    static ui32 Hash(void* addr, int prevInd, int tag)
    {
        return (((size_t)addr + ((size_t)addr / STACKS_HASH_MAP_SIZE)) + prevInd + tag) % STACKS_HASH_MAP_SIZE;
    }

    static bool EqualFrame(const TFrameInfo& frame, void* addr, int prevInd, int tag)
    {
        return (frame.Addr == addr && frame.PrevInd == prevInd && frame.Tag == tag);
    }

    bool IsSlotEmpty(ui32 slot) const
    {
        return Frames[slot].Addr == 0;
    }

    bool InsertsAllowed() const
    {
        return UsedSlots < STACKS_HASH_MAP_SIZE / 2;
    }

    // returns the index in the hashmap
    int AddFrame(void* addr, int prevFrameIndex, int tag, bool last)
    {
        ui32 slot = Hash(addr, prevFrameIndex, tag);
        ui32 prevSlot = (slot - 1) % STACKS_HASH_MAP_SIZE;

        while (!EqualFrame(Frames[slot], addr, prevFrameIndex, tag) && !IsSlotEmpty(slot) && slot != prevSlot) {
            slot = (slot + 1) % STACKS_HASH_MAP_SIZE;
            SearchSkipCount++;
        }

        if (EqualFrame(Frames[slot], addr, prevFrameIndex, tag)) {
            if (last) {
                ++Samples;
            }
        } else if (InsertsAllowed() && IsSlotEmpty(slot)) {
            // add new sample
            Frames[slot].Clear();
            Frames[slot].Addr = addr;
            Frames[slot].PrevInd = prevFrameIndex;
            Frames[slot].Tag = tag;
            ++UsedSlots;
            if (last) {
                ++UniqueSamples;
                ++Samples;
            }
        } else {
            // don't insert new sample if the search is becoming too slow
            ++DroppedSamples;
            return -1;
        }

        return slot;
    }
};


////////////////////////////////////////////////////////////////////////////////

class TAllocationStackCollector::TImpl: public TStackCollector<TStats> {
    using TBase = TStackCollector<TStats>;

private:
    TStats Total;

public:
    int Alloc(void** stack, size_t frameCount, int tag, size_t size)
    {
        int stackId = TBase::AddStack(stack, frameCount, tag);
        if (stackId >= 0) {
            TBase::GetStats(stackId).Alloc(size);
            Total.Alloc(size);
        }
        return stackId;
    }

    void Free(int stackId, size_t size)
    {
        TBase::GetStats(stackId).Free(size);
        Total.Free(size);
    }

    void Clear()
    {
        TBase::Clear();
        Total.Clear();
    }

    void Dump(int count, IAllocationStatsDumper& out) const
    {
        const TFrameInfo* frames = TBase::GetFrames();
        size_t framesCount = TBase::GetFramesCount();

        TVector<const TFrameInfo*> stacks;
        for (size_t i = 0; i < framesCount; ++i) {
            if (frames[i].Stats.Allocs) {
                stacks.push_back(&frames[i]);
            }
        }

        Sort(stacks, [] (const TFrameInfo* l, const TFrameInfo* r) {
            const auto& ls = l->Stats;
            const auto& rs = r->Stats;
            return ls.CurrentSize != rs.CurrentSize
                ? ls.CurrentSize > rs.CurrentSize
                : ls.Allocs != rs.Allocs
                    ? ls.Allocs > rs.Allocs
                    : ls.Frees > rs.Frees;
        });

        out.DumpTotal(Total);

        TAllocationInfo allocInfo;
        int printedCount = 0;
        for (const TFrameInfo* stack: stacks) {
            allocInfo.Clear();
            allocInfo.Tag = stack->Tag;
            allocInfo.Stats = stack->Stats;
            TBase::BackTrace(stack, allocInfo.Stack);

            out.DumpEntry(allocInfo);

            if (++printedCount >= count) {
                break;
            }
        }
    }
};

////////////////////////////////////////////////////////////////////////////////

TAllocationStackCollector::TAllocationStackCollector()
    : Impl(new TImpl())
{}

TAllocationStackCollector::~TAllocationStackCollector()
{}

int TAllocationStackCollector::Alloc(void** stack, size_t frameCount, int tag, size_t size)
{
    return Impl->Alloc(stack, frameCount, tag, size);
}

void TAllocationStackCollector::Free(int stackId, size_t size)
{
    Impl->Free(stackId, size);
}

void TAllocationStackCollector::Clear()
{
    Impl->Clear();
}

void TAllocationStackCollector::Dump(int count, IAllocationStatsDumper &out) const
{
    Impl->Dump(count, out);
}


TString IAllocationStatsDumper::FormatTag(int tag) {
    return ToString(tag);
}

TString IAllocationStatsDumper::FormatSize(intptr_t sz) {
    return ToString(sz);
}


TAllocationStatsDumper::TAllocationStatsDumper(IOutputStream& out)
    : PrintedCount(0)
    , Out(out)
    , SymbolCache(2048)
{}

void TAllocationStatsDumper::DumpTotal(const TStats& total) {
    Out << "TOTAL"
        << "\tAllocs: " << total.Allocs
        << "\tFrees: " << total.Frees
        << "\tCurrentSize: " << FormatSize(total.CurrentSize)
        << Endl;
}

void TAllocationStatsDumper::DumpEntry(const TAllocationInfo& allocInfo) {
    Out << Endl
        << "STACK #" << PrintedCount+1 << ": " << FormatTag(allocInfo.Tag)
        << "\tAllocs: " << allocInfo.Stats.Allocs
        << "\tFrees: " << allocInfo.Stats.Frees
        << "\tCurrentSize: " << FormatSize(allocInfo.Stats.CurrentSize)
        << Endl;
    FormatBackTrace(allocInfo.Stack.data(), allocInfo.Stack.size());
    PrintedCount++;
}

void TAllocationStatsDumper::FormatBackTrace(void* const* stack, size_t sz) {
    char name[1024];
    for (size_t i = 0; i < sz; ++i) {
        TSymbol symbol;
        auto it = SymbolCache.Find(stack[i]);
        if (it != SymbolCache.End()) {
            symbol = it.Value();
        } else {
            TResolvedSymbol rs = ResolveSymbol(stack[i], name, sizeof(name));
            symbol = {rs.NearestSymbol, rs.Name};
            SymbolCache.Insert(stack[i], symbol);
        }

        Out << Hex((intptr_t)stack[i], HF_FULL) << "\t" << symbol.Name;
        intptr_t offset = (intptr_t)stack[i] - (intptr_t)symbol.Address;
        if (offset)
            Out << " +" << offset;
        Out << Endl;
    }
}

}   // namespace NAllocProfiler
