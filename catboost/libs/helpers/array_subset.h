#pragma once

#include "dbg_output.h"
#include "dynamic_iterator.h"
#include "exception.h"
#include "math_utils.h"
#include "maybe_owning_array_holder.h"
#include "parallel_tasks.h"

#include <catboost/private/libs/index_range/index_range.h>

#include <library/dbg_output/dump.h>
#include <library/threading/local_executor/local_executor.h>

#include <util/generic/algorithm.h>
#include <util/generic/array_ref.h>
#include <util/generic/maybe.h>
#include <util/generic/variant.h>
#include <util/generic/vector.h>
#include <util/generic/xrange.h>
#include <util/generic/ylimits.h>
#include <util/generic/ymath.h>
#include <util/system/compiler.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <type_traits>

namespace NCB {

    template <class TSize>
    struct TSubsetBlock {
        TSize SrcBegin = 0;
        TSize SrcEnd = 0;
        TSize DstBegin = 0;

    public:
        TSubsetBlock() = default;

        TSubsetBlock(NCB::TIndexRange<TSize> srcRange, TSize dstBegin)
            : SrcBegin(srcRange.Begin)
            , SrcEnd(srcRange.End)
            , DstBegin(dstBegin)
        {}

        TSize GetSize() const noexcept {
            return SrcEnd - SrcBegin;
        }

        TSize GetDstEnd() const noexcept {
            return DstBegin + GetSize();
        }

        bool operator ==(const TSubsetBlock& lhs) const {
            return (SrcBegin == lhs.SrcBegin) && (SrcEnd == lhs.SrcEnd) && (DstBegin == lhs.DstBegin);
        }
    };

    template <class TSize>
    struct TRangesSubset {
        using TBlocks = TVector<TSubsetBlock<TSize>>;

        TSize Size;
        TBlocks Blocks;

    public:
        explicit TRangesSubset(const NCB::IIndexRangesGenerator<TSize>& srcRangesGenerator) {
            TSize blockCount = srcRangesGenerator.RangesCount();
            Blocks.yresize(blockCount);

            TSize dstBegin = 0;
            for (auto blockIdx : xrange(blockCount)) {
                Blocks[blockIdx] = TSubsetBlock<TSize>(srcRangesGenerator.GetRange(blockIdx), dstBegin);
                dstBegin += Blocks[blockIdx].GetSize();
            }
            Size = dstBegin;
        }

        // if blocks were precalulated
        TRangesSubset(TSize size, TBlocks&& blocks)
            : Size(size)
            , Blocks(std::move(blocks))
        {}

        bool operator ==(const TRangesSubset& lhs) const {
            return (Size == lhs.Size) && (Blocks == lhs.Blocks);
        }

        bool operator !=(const TRangesSubset& lhs) const {
            return !(*this == lhs);
        }
    };

    template <class TSize>
    class TRangesSubsetIterator final : public IDynamicIterator<TSize> {
    public:
        TRangesSubsetIterator(
            const TSubsetBlock<TSize>* startBlock,
            TSize startInBlockIdx,
            const TSubsetBlock<TSize>* endBlock,
            TSize lastBlockEndInBlockIdx)
            : CurrentBlock(startBlock)
            , CurrentIdx(0) // properly inited below
            , EndIdx(0) // properly inited below
            , EndBlock(endBlock)
            , LastBlockInBlockEndIdx(lastBlockEndInBlockIdx)
        {
            if (startBlock != endBlock) {
                CurrentIdx = startBlock->SrcBegin + startInBlockIdx;
                if (startBlock + 1 == endBlock) {
                    EndIdx = startBlock->SrcBegin + lastBlockEndInBlockIdx;
                } else {
                    EndIdx = startBlock->SrcEnd;
                }
            }
        }

        TRangesSubsetIterator(const TRangesSubset<TSize>& rangesSubset, TSize offset = 0)
            : CurrentBlock(nullptr) // properly inited below
            , CurrentIdx(0) // properly inited below
            , EndIdx(0) // properly inited below
            , EndBlock(rangesSubset.Blocks.data() + rangesSubset.Blocks.size())
            , LastBlockInBlockEndIdx(0) // properly inited below
        {
            const auto& blocks = rangesSubset.Blocks;
            CurrentBlock = LowerBound(
                blocks.begin(),
                blocks.end(),
                offset,
                [] (const TSubsetBlock<TSize>& block, TSize offset) {
                    return block.GetDstEnd() <= offset;
                }
            );
            if (CurrentBlock != EndBlock) {
                CurrentIdx = CurrentBlock->SrcBegin + (offset - CurrentBlock->DstBegin);
                EndIdx = CurrentBlock->SrcEnd;
                LastBlockInBlockEndIdx = rangesSubset.Blocks.back().GetSize();
            }
        }


        inline TMaybe<TSize> Next() override {
            if (CurrentBlock == EndBlock) {
                return IDynamicIterator<TSize>::END_VALUE;
            }
            if (CurrentIdx == EndIdx) {
                ++CurrentBlock;
                if (CurrentBlock == EndBlock) {
                    return IDynamicIterator<TSize>::END_VALUE;
                } else if (CurrentBlock + 1 == EndBlock) {
                    EndIdx = CurrentBlock->SrcBegin + LastBlockInBlockEndIdx;
                } else {
                    EndIdx = CurrentBlock->SrcEnd;
                }
                CurrentIdx = CurrentBlock->SrcBegin + 1;
                return CurrentBlock->SrcBegin;
            }
            return CurrentIdx++;
        }

    private:
        const TSubsetBlock<TSize>* CurrentBlock;
        TSize CurrentIdx; // src idx
        TSize EndIdx; // src idx, in current block
        const TSubsetBlock<TSize>* const EndBlock;
        TSize LastBlockInBlockEndIdx;
    };


    template <class TSize>
    using TIndexedSubset = TVector<TSize>; // index in src data

    template <class TSize>
    struct TFullSubset {
        TSize Size;

    public:
        explicit TFullSubset(TSize size)
            : Size(size)
        {}

        bool operator ==(const TFullSubset& lhs) const {
            return Size == lhs.Size;
        }

        bool operator !=(const TFullSubset& lhs) const {
            return !(*this == lhs);
        }
    };

    template <class TSize>
    class TArraySubsetIndexing
        : public TVariant<TFullSubset<TSize>, TRangesSubset<TSize>, TIndexedSubset<TSize>>
    {
    public:
        using TBase = TVariant<TFullSubset<TSize>, TRangesSubset<TSize>, TIndexedSubset<TSize>>;

    public:
        // default constructor is necessary for BinSaver serialization & Cython
        TArraySubsetIndexing()
            : TArraySubsetIndexing(TFullSubset<TSize>(0))
        {
            ConsecutiveSubsetBeginCache = GetConsecutiveSubsetBeginImpl();
        }

        explicit TArraySubsetIndexing(TFullSubset<TSize>&& subset)
            : TBase(std::move(subset))
        {
            ConsecutiveSubsetBeginCache = GetConsecutiveSubsetBeginImpl();
        }

        explicit TArraySubsetIndexing(TRangesSubset<TSize>&& subset)
            : TBase(std::move(subset))
        {
            ConsecutiveSubsetBeginCache = GetConsecutiveSubsetBeginImpl();
        }

        explicit TArraySubsetIndexing(TIndexedSubset<TSize>&& subset)
            : TBase(std::move(subset))
        {
            ConsecutiveSubsetBeginCache = GetConsecutiveSubsetBeginImpl();
        }

        friend bool operator ==(const TArraySubsetIndexing& a, const TArraySubsetIndexing& b) {
            return static_cast<const TBase&>(a) == static_cast<const TBase&>(b);
        }

        bool IsFullSubset() const {
            return HoldsAlternative<TFullSubset<TSize>>(*this);
        }

        TSize Size() const {
            switch (TBase::index()) {
                case TVariantIndexV<TFullSubset<TSize>, TBase>:
                    return Get<TFullSubset<TSize>>().Size;
                case TVariantIndexV<TRangesSubset<TSize>, TBase>:
                    return Get<TRangesSubset<TSize>>().Size;
                case TVariantIndexV<TIndexedSubset<TSize>, TBase>:
                    return static_cast<TSize>(Get<TIndexedSubset<TSize>>().size());
            }
            return 0; // just to silence compiler warnings
        }

        // number of elements for TFullSubset or TIndexedSubset, number of ranges for TRangesSubset
        TSize GetParallelizableUnitsCount() const {
            switch (TBase::index()) {
                case TVariantIndexV<TFullSubset<TSize>, TBase>:
                    return Get<TFullSubset<TSize>>().Size;
                case TVariantIndexV<TRangesSubset<TSize>, TBase>:
                    return Get<TRangesSubset<TSize>>().Blocks.size();
                case TVariantIndexV<TIndexedSubset<TSize>, TBase>:
                    return static_cast<TSize>(Get<TIndexedSubset<TSize>>().size());
            }
            return 0; // just to silence compiler warnings
        }

        // Had to redefine Get because automatic resolution does not work with current TVariant implementation
        template <class T>
        decltype(auto) Get() {
            return ::Get<T>((TBase&)*this);
        }

        template <class T>
        decltype(auto) Get() const {
            return ::Get<T>((const TBase&)*this);
        }
        TMaybe<TSize> GetConsecutiveSubsetBegin() const {
            return ConsecutiveSubsetBeginCache;
        }
        // returns Nothing() if subset is not consecutive
        TMaybe<TSize> GetConsecutiveSubsetBeginImpl() const {
            switch (TBase::index()) {
                case TVariantIndexV<TFullSubset<TSize>, TBase>:
                    return TSize(0);
                case TVariantIndexV<TRangesSubset<TSize>, TBase>:
                    {
                        const auto& blocks = Get<TRangesSubset<TSize>>().Blocks;
                        if (blocks.size() == 0) {
                            return TSize(0);
                        }
                        for (auto i : xrange(blocks.size() - 1)) {
                            if (blocks[i].SrcEnd != blocks[i + 1].SrcBegin) {
                                return Nothing();
                            }
                        }
                        return blocks[0].SrcBegin;
                    }
                case TVariantIndexV<TIndexedSubset<TSize>, TBase>:
                    {
                        TConstArrayRef<TSize> indices = Get<TIndexedSubset<TSize>>();
                        if (indices.size() == 0) {
                            return TSize(0);
                        }
                        for (auto i : xrange(indices.size() - 1)) {
                            if ((indices[i] + 1) != indices[i + 1]) {
                                return Nothing();
                            }
                        }
                        return indices[0];
                    }
            }
            return Nothing(); // just to silence compiler warnings
        }

        bool IsConsecutive() const {
            return GetConsecutiveSubsetBegin().Defined();
        }

        // f is a visitor function that will be repeatedly called with (index, srcIndex) arguments
        template <class F>
        void ForEach(F&& f) const {
            ForEachInSubRange(NCB::TIndexRange<TSize>(GetParallelizableUnitsCount()), f);
        }

        /*
         * unit Ranges are index ranges for TFullSubset and TIndexedSubset and ranges ranges for TRangesSubset
         *
         * for TRangesSubset block sizes might not be equal to approximateBlockSize because
         * block sizes might not be divisible by approximateBlockSize, that's why 'approximate'
         * is in the name of this block size parameter
         *
         * public because might be used for externally-runned parallelism
         *  (for example when additional initialization/finalization not provided by ParallelForEach is needed
         *   per each subRange)
         */
        const NCB::TSimpleIndexRangesGenerator<TSize> GetParallelUnitRanges(TSize approximateBlockSize) const {
            CB_ENSURE(
                approximateBlockSize > 0,
                "approximateBlockSize (" << approximateBlockSize  << ") is <= 0"
            );
            const TSize parallelizableUnitsCount = GetParallelizableUnitsCount();
            if (!parallelizableUnitsCount) {
                return NCB::TSimpleIndexRangesGenerator<TSize>(NCB::TIndexRange<TSize>(0), 1);
            }
            Y_ASSERT(Size());
            return NCB::TSimpleIndexRangesGenerator<TSize>(
                NCB::TIndexRange<TSize>(parallelizableUnitsCount),
                std::max(
                    (TSize)std::llround(
                        double(parallelizableUnitsCount)
                        / double(Size())
                        * double(approximateBlockSize)
                    ),
                    TSize(1)
                )
            );
        }

        // might be used for external parallelism if you need element indices of the subrange
        NCB::TIndexRange<TSize> GetElementRangeFromUnitRange(NCB::TIndexRange<TSize> unitRange) const {
            switch (TBase::index()) {
                case TVariantIndexV<TFullSubset<TSize>, TBase>:
                    return unitRange;
                case TVariantIndexV<TRangesSubset<TSize>, TBase>:
                    {
                        const auto& rangesSubset = Get<TRangesSubset<TSize>>();
                        const auto& blocks = rangesSubset.Blocks;
                        return NCB::TIndexRange<TSize>(
                            blocks[unitRange.Begin].DstBegin,
                            unitRange.End < blocks.size() ? blocks[unitRange.End].DstBegin : rangesSubset.Size
                        );
                    }
                    break;
                case TVariantIndexV<TIndexedSubset<TSize>, TBase>:
                    return unitRange;
            }
            return NCB::TIndexRange<TSize>(0); // silence compiler warnings
        }

// need to be able to use 'break' or 'return' in iteration
#define LOOP_SUB_RANGE(unitSubRange, LOOP_BODY_MACRO) \
        switch (TBase::index()) { \
            case TVariantIndexV<TFullSubset<TSize>, TBase>: \
                { \
                    for (TSize index : unitSubRange.Iter()) { \
                        LOOP_BODY_MACRO(index, index) \
                    } \
                } \
                break; \
            case TVariantIndexV<TRangesSubset<TSize>, TBase>: \
                { \
                    const auto& blocks = Get<TRangesSubset<TSize>>().Blocks; \
                    for (TSize blockIndex : unitSubRange.Iter()) { \
                        auto block = blocks[blockIndex]; \
                        TSize index = block.DstBegin; \
                        for (TSize srcIndex = block.SrcBegin; \
                             srcIndex != block.SrcEnd; \
                             ++srcIndex, ++index) \
                        { \
                            LOOP_BODY_MACRO(index, srcIndex) \
                        } \
                    } \
                } \
                break; \
            case TVariantIndexV<TIndexedSubset<TSize>, TBase>: \
                { \
                    const auto& srcIndex = Get<TIndexedSubset<TSize>>(); \
                    for (TSize index : unitSubRange.Iter()) { \
                        LOOP_BODY_MACRO(index, srcIndex[index]) \
                    } \
                } \
                break; \
        }


// need to be able to use 'break' or 'return' in iteration
#define LOOP_SUB_RANGE_BLOCKWISE(unitSubRange, BLOCK_MACRO) \
        switch (TBase::index()) { \
            case TVariantIndexV<TFullSubset<TSize>, TBase>: \
                { \
                    BLOCK_MACRO(unitSubRange.Begin, unitSubRange.End, unitSubRange.Begin, (TSize*)nullptr) \
                } \
                break; \
            case TVariantIndexV<TRangesSubset<TSize>, TBase>: \
                { \
                    const auto& blocks = Get<TRangesSubset<TSize>>().Blocks; \
                    for (TSize blockIndex : unitSubRange.Iter()) { \
                        const auto block = blocks[blockIndex]; \
                        BLOCK_MACRO(block.SrcBegin, block.SrcEnd, block.DstBegin, (TSize*)nullptr) \
                    } \
                } \
                break; \
            case TVariantIndexV<TIndexedSubset<TSize>, TBase>: \
                { \
                    const auto& srcIndex = Get<TIndexedSubset<TSize>>(); \
                    BLOCK_MACRO(unitSubRange.Begin, unitSubRange.End, unitSubRange.Begin, srcIndex.data()) \
                } \
                break; \
        }


        /* unitSubRange is index range for TFullSubset and TIndexedSubset and ranges range for TRangesSubset
         *
         * for TRangesSubset block sizes might not be equal to approximateBlockSize because
         * block sizes might not be divisible by approximateBlockSize, that's why 'approximate'
         * is in the name of this block size parameter
         *
         * f is a visitor function that will be repeatedly called with (index, srcIndex) arguments
         */
        template <class F>
        void ForEachInSubRange(NCB::TIndexRange<TSize> unitSubRange, const F& f) const {
#define FOR_EACH_BODY(index, subIndex) f(index, subIndex);
            LOOP_SUB_RANGE(unitSubRange, FOR_EACH_BODY)
#undef FOR_EACH_BODY
        }

        /* unitSubRange is index range for TFullSubset and TIndexedSubset and ranges range for TRangesSubset
         *
         * for TRangesSubset block sizes might not be equal to approximateBlockSize because
         * block sizes might not be divisible by approximateBlockSize, that's why 'approximate'
         * is in the name of this block size parameter
         *
         * f is a visitor function that will be repeatedly called with (index, srcIndex) arguments
         */
        template <class F>
        void ForEachBlockwiseInSubRange(NCB::TIndexRange<TSize> unitSubRange, const F& f) const {
#define BLOCK_MACRO(srcBegin, srcEnd, dstBegin, srcIndices) f(srcBegin, srcEnd, dstBegin, srcIndices);
            LOOP_SUB_RANGE_BLOCKWISE(unitSubRange, BLOCK_MACRO)
#undef FOR_EACH_BODY
        }

        /* predicate is a visitor function that returns bool
         * it will be repeatedly called with (index, srcIndex) arguments
         * until it returns true or all elements are iterated over
         *
         *  @returns true if a pair of (index, subIndex) for which predicate returned true was found
         *  and false otherwise
         */
        template <class TPredicate>
        bool Find(const TPredicate& predicate) const {
#define FIND_BODY(index, subIndex) if (predicate(index, subIndex)) return true;
            NCB::TIndexRange<TSize> unitSubRange(GetParallelizableUnitsCount());
            LOOP_SUB_RANGE(unitSubRange, FIND_BODY)
#undef FIND_BODY
            return false;
        }

#undef LOOP_SUB_RANGE


        /* f is a visitor function that will be repeatedly called with (index, srcIndex) arguments
         * for TRangesSubset block sizes might not be equal to approximateBlockSize because
         * block sizes might not be divisible by approximateBlockSize, that's why 'approximate'
         * is in the name of this block size parameter
         * if approximateBlockSize is undefined divide data approximately evenly between localExecutor
         * threads
         */
        template <class F>
        void ParallelForEach(
            F&& f,
            NPar::TLocalExecutor* localExecutor,
            TMaybe<TSize> approximateBlockSize = Nothing()
        ) const {
            if (!Size()) {
                return;
            }

            if (!approximateBlockSize.Defined()) {
                TSize localExecutorThreadsPlusCurrentCount = (TSize)localExecutor->GetThreadCount() + 1;
                approximateBlockSize = CeilDiv(Size(), localExecutorThreadsPlusCurrentCount);
            }

            const NCB::TSimpleIndexRangesGenerator<TSize> parallelUnitRanges =
                GetParallelUnitRanges(*approximateBlockSize);

            CB_ENSURE(
                    (sizeof(TSize) < sizeof(int))
                 || (parallelUnitRanges.RangesCount() <= (TSize)std::numeric_limits<int>::max()),
                "Number of parallel processing data ranges (" << parallelUnitRanges.RangesCount()
                << ") is greater than the max limit for LocalExecutor (" << std::numeric_limits<int>::max()
                << ')'
            );

            localExecutor->ExecRangeWithThrow(
                [this, parallelUnitRanges, f = std::move(f)] (int id) {
                    ForEachInSubRange(parallelUnitRanges.GetRange(id), f);
                },
                0,
                (int)parallelUnitRanges.RangesCount(),
                NPar::TLocalExecutor::WAIT_COMPLETE
            );
        }

        /* f is a visitor function that will be repeatedly called with (srcBegin, srcEnd, dstBegin, srcIndices) arguments
         * for TRangesSubset block sizes might not be equal to approximateBlockSize because
         * block sizes might not be divisible by approximateBlockSize, that's why 'approximate'
         * is in the name of this block size parameter
         * if approximateBlockSize is undefined divide data approximately evenly between localExecutor
         * threads
         */
        template <class F>
        void ParallelForEachBlockwise(
            F&& f,
            NPar::TLocalExecutor* localExecutor,
            TMaybe<TSize> approximateBlockSize = Nothing()
        ) const {
            if (!Size()) {
                return;
            }

            if (!approximateBlockSize.Defined()) {
                TSize localExecutorThreadsPlusCurrentCount = (TSize)localExecutor->GetThreadCount() + 1;
                approximateBlockSize = CeilDiv(Size(), localExecutorThreadsPlusCurrentCount);
            }

            const NCB::TSimpleIndexRangesGenerator<TSize> parallelUnitRanges =
                GetParallelUnitRanges(*approximateBlockSize);

            CB_ENSURE(
                    (sizeof(TSize) < sizeof(int))
                 || (parallelUnitRanges.RangesCount() <= (TSize)std::numeric_limits<int>::max()),
                "Number of parallel processing data ranges (" << parallelUnitRanges.RangesCount()
                << ") is greater than the max limit for LocalExecutor (" << std::numeric_limits<int>::max()
                << ')'
            );

            localExecutor->ExecRangeWithThrow(
                [this, parallelUnitRanges, f = std::move(f)] (int id) {
                    ForEachBlockwiseInSubRange(parallelUnitRanges.GetRange(id), f);
                },
                0,
                (int)parallelUnitRanges.RangesCount(),
                NPar::TLocalExecutor::WAIT_COMPLETE
            );
        }
    private:
        TMaybe<TSize> ConsecutiveSubsetBeginCache;
    };

    template <class TS, class TSize>
    class TDumperArraySubsetVisitor {
    public:
        TDumperArraySubsetVisitor(TS& s)
            : S(s)
        {}

        void operator()(const TFullSubset<TSize>& fullSubset) const {
            S << "FullSubset(size=" << fullSubset.Size << ")";
        }

        void operator()(const TRangesSubset<TSize>& rangesSubset) const {
            S << "RangesSubset(size=" << rangesSubset.Size << ", blocks="
              << DbgDumpWithIndices<TSubsetBlock<TSize>>(rangesSubset.Blocks, true) << ")";
        }

        void operator()(const TIndexedSubset<TSize>& indexedSubset) const {
            S << "IndexedSubset(size=" << indexedSubset.size() << ", indices="
              << DbgDumpWithIndices<TSize>(indexedSubset, true) << ")";
        }

    private:
        TS& S;
    };
}


template <class TSize>
struct TDumper<NCB::TSubsetBlock<TSize>> {
    template <class S>
    static inline void Dump(S& s, const NCB::TSubsetBlock<TSize>& block) {
        s << "Src=[" << block.SrcBegin << "," << block.SrcEnd << "), DstBegin=" << block.DstBegin;
    }
};


template <class TSize>
struct TDumper<NCB::TArraySubsetIndexing<TSize>> {
    template <class S>
    static inline void Dump(S& s, const NCB::TArraySubsetIndexing<TSize>& subset) {
        Visit(NCB::TDumperArraySubsetVisitor<S, TSize>(s), subset);
    }
};


namespace NCB {

    // TODO(akhropov): too expensive for release?
    template <class TSize = size_t>
    void CheckSubsetIndices(const TArraySubsetIndexing<TSize>& srcSubset, TSize srcSize) {
        using TVariantType = typename TArraySubsetIndexing<TSize>::TBase;

        switch (srcSubset.index()) {
            case TVariantIndexV<TFullSubset<TSize>, TVariantType>:
                CB_ENSURE(
                    srcSize == srcSubset.Size(),
                    "srcSubset is TFullSubset, but has different size from src's size"
                );
                break;
            case TVariantIndexV<TRangesSubset<TSize>, TVariantType>: {
                    const auto& rangesSrcSubset = srcSubset.template Get<TRangesSubset<TSize>>();
                    for (auto i : xrange(rangesSrcSubset.Blocks.size())) {
                        CB_ENSURE(
                            rangesSrcSubset.Blocks[i].SrcEnd <= srcSize,
                            "TRangesSubset.Blocks[" <<  i << "].SrcEnd (" << rangesSrcSubset.Blocks[i].SrcEnd
                            << ") > srcSize (" << srcSize << ')'
                        );
                    }
                }
                break;
            case TVariantIndexV<TIndexedSubset<TSize>, TVariantType>: {
                    const auto& indexedSrcSubset = srcSubset.template Get<TIndexedSubset<TSize>>();
                    for (auto i : xrange(indexedSrcSubset.size())) {
                        CB_ENSURE(
                            indexedSrcSubset[i] < srcSize,
                            "TIndexedSubset[" <<  i << "] (" << indexedSrcSubset[i]
                            << ") >= srcSize (" << srcSize << ')'
                        );
                    }
                }
                break;
        }
    }


    // subcases of global Compose

    template <class TSize = size_t>
    TArraySubsetIndexing<TSize> Compose(
        const TFullSubset<TSize>& src,
        const TArraySubsetIndexing<TSize>& srcSubset
    ) {
        CheckSubsetIndices(srcSubset, src.Size);
        return srcSubset;
    }

    template <class TSize = size_t>
    TArraySubsetIndexing<TSize> Compose(
        const TRangesSubset<TSize>& src,
        const TRangesSubset<TSize>& srcSubset
    ) {
        TVector<TSubsetBlock<TSize>> newBlocks;

        for (const auto& srcSubsetBlock : srcSubset.Blocks) {
            auto srcIt = LowerBound(
                src.Blocks.begin(),
                src.Blocks.end(),
                srcSubsetBlock.SrcBegin,
                [] (const TSubsetBlock<TSize>& srcBlock, TSize srcSubsetBlockSrcBegin) {
                    return srcBlock.GetDstEnd() <= srcSubsetBlockSrcBegin;
                }
            );
            CB_ENSURE(
                srcIt != src.Blocks.end(),
                "TRangesSubset srcSubset Block[" << srcSubsetBlock.SrcBegin << ',' << srcSubsetBlock.SrcEnd
                << ") not found in TRangesSubset src"
            );

            TSize dstBegin = srcSubsetBlock.DstBegin;
            TSize srcSubsetOffset = srcSubsetBlock.SrcBegin - srcIt->DstBegin;

            while (true) {
                TSize srcBegin = srcIt->SrcBegin + srcSubsetOffset;

                if (srcSubsetBlock.SrcEnd <= srcIt->GetDstEnd()) {
                    newBlocks.push_back(
                        TSubsetBlock<TSize>(
                            {srcBegin, srcIt->SrcBegin + (srcSubsetBlock.SrcEnd - srcIt->DstBegin)},
                            dstBegin
                        )
                    );
                    break;
                }

                TIndexRange<TSize> srcRange(srcBegin, srcIt->SrcEnd);
                if (srcRange.GetSize()) { // skip empty blocks
                    newBlocks.push_back(TSubsetBlock<TSize>(srcRange, dstBegin));
                    dstBegin += newBlocks.back().GetSize();
                }

                srcSubsetOffset = 0;
                ++srcIt;

                CB_ENSURE(
                    srcIt != src.Blocks.end(),
                    "TRangesSubset srcSubset Block[" << srcSubsetBlock.SrcBegin << ','
                    << srcSubsetBlock.SrcEnd << ") exceeds TRangesSubset src size"
                );
            }
        }

        return TArraySubsetIndexing<TSize>(TRangesSubset<TSize>(srcSubset.Size, std::move(newBlocks)));
    }

    template <class TSize = size_t>
    TArraySubsetIndexing<TSize> Compose(
        const TRangesSubset<TSize>& src,
        const TIndexedSubset<TSize>& srcSubset
    ) {
        TIndexedSubset<TSize> result;
        result.yresize(srcSubset.size());

        for (auto i : xrange(srcSubset.size())) {
            auto subsetIdx = srcSubset[i];
            auto srcIt = LowerBound(
                src.Blocks.begin(),
                src.Blocks.end(),
                subsetIdx,
                [] (const TSubsetBlock<TSize>& srcBlock, TSize subsetIdx) {
                    return srcBlock.GetDstEnd() <= subsetIdx;
                }
            );
            CB_ENSURE(
                srcIt != src.Blocks.end(),
                "TIndexedSubset srcSubset index " << subsetIdx << " not found in TRangesSubset src"
            );
            result[i] = srcIt->SrcBegin + (subsetIdx - srcIt->DstBegin);
        }

        return TArraySubsetIndexing<TSize>(std::move(result));
    }

    template <class TSize = size_t>
    TArraySubsetIndexing<TSize> Compose(
        const TRangesSubset<TSize>& src,
        const TArraySubsetIndexing<TSize>& srcSubset
    ) {
        using TVariantType = typename TArraySubsetIndexing<TSize>::TBase;

        switch (srcSubset.index()) {
            case TVariantIndexV<TFullSubset<TSize>, TVariantType>:
                CB_ENSURE(
                    src.Size == srcSubset.Size(),
                    "srcSubset is TFullSubset, but has different size from src's size"
                );
                return TArraySubsetIndexing<TSize>(TRangesSubset<TSize>(src));
            case TVariantIndexV<TRangesSubset<TSize>, TVariantType>:
                return Compose(src, srcSubset.template Get<TRangesSubset<TSize>>());
            case TVariantIndexV<TIndexedSubset<TSize>, TVariantType>:
                return Compose(src, srcSubset.template Get<TIndexedSubset<TSize>>());
        }
        Y_FAIL("This should be unreachable");
        // return something to keep compiler happy
        return TArraySubsetIndexing<TSize>( TFullSubset<TSize>(0) );
    }

    template <class TSize = size_t>
    TArraySubsetIndexing<TSize> Compose(
        const TIndexedSubset<TSize>& src,
        const TRangesSubset<TSize>& srcSubset
    ) {
        TIndexedSubset<TSize> result;
        result.yresize(srcSubset.Size);

        auto dstIt = result.begin();
        for (const auto& srcSubsetBlock : srcSubset.Blocks) {
            for (auto srcIdx : xrange(srcSubsetBlock.SrcBegin, srcSubsetBlock.SrcEnd)) {
                // use CB_ENSURE instead of standard src.at to throw standard TCatBoostException
                CB_ENSURE(
                    srcIdx < src.size(),
                    "srcSubset's has index (" << srcIdx << ") greater than src size (" << src.size() << ")"
                );
                *dstIt++ = src[srcIdx];
            }
        }

        return TArraySubsetIndexing<TSize>(std::move(result));
    }

    template <class TSize = size_t>
    TArraySubsetIndexing<TSize> Compose(
        const TIndexedSubset<TSize>& src,
        const TIndexedSubset<TSize>& srcSubset
    ) {
        TIndexedSubset<TSize> result;
        result.yresize(srcSubset.size());

        auto dstIt = result.begin();
        for (auto srcIdx : srcSubset) {
            // use CB_ENSURE instead of standard src.at to throw standard TCatBoostException
            CB_ENSURE(
                srcIdx < src.size(),
                "srcSubset's has index (" << srcIdx << ") greater than src size (" << src.size() << ")"
            );
            *dstIt++ = src[srcIdx];
        }

        return TArraySubsetIndexing<TSize>(std::move(result));
    }

    template <class TSize = size_t>
    TArraySubsetIndexing<TSize> Compose(
        const TIndexedSubset<TSize>& src,
        const TArraySubsetIndexing<TSize>& srcSubset
    ) {
        using TVariantType = typename TArraySubsetIndexing<TSize>::TBase;

        switch (srcSubset.index()) {
            case TVariantIndexV<TFullSubset<TSize>, TVariantType>:
                CB_ENSURE(
                    src.size() == srcSubset.Size(),
                    "srcSubset is TFullSubset, but has different size from src's size"
                );
                return TArraySubsetIndexing<TSize>(TIndexedSubset<TSize>(src));
            case TVariantIndexV<TRangesSubset<TSize>, TVariantType>:
                return Compose(src, srcSubset.template Get<TRangesSubset<TSize>>());
            case TVariantIndexV<TIndexedSubset<TSize>, TVariantType>:
                return Compose(src, srcSubset.template Get<TIndexedSubset<TSize>>());
        }
        Y_FAIL("This should be unreachable");
        // return something to keep compiler happy
        return TArraySubsetIndexing<TSize>( TFullSubset<TSize>(0) );
    }


    // main Compose

    template <class TSize = size_t>
    TArraySubsetIndexing<TSize> Compose(
        const TArraySubsetIndexing<TSize>& src,
        const TArraySubsetIndexing<TSize>& srcSubset
    ) {
        return ::Visit([&](const auto& val) {
            return Compose(val, srcSubset);
        }, src);
    }


    template <class TSize = size_t>
    bool IndicesEqual(const TArraySubsetIndexing<TSize>& lhs, TConstArrayRef<TSize> rhs) {
        return !lhs.Find([rhs](TSize idx, TSize srcIdx) { return rhs[idx] != srcIdx; });
    }


    template <class TSize = size_t>
    TArraySubsetIndexing<TSize> MakeIncrementalIndexing(
        const TArraySubsetIndexing<TSize>& indexing,
        NPar::TLocalExecutor* localExecutor
    ) {
        if (HoldsAlternative<TFullSubset<TSize>>(indexing)) {
            return indexing;
        } else {
            TVector<TSize> indices;
            indices.yresize(indexing.Size());
            TArrayRef<TSize> indicesRef = indices;

            indexing.ParallelForEach(
                [=] (TSize objectIdx, TSize srcObjectIdx) {
                  indicesRef[objectIdx] = srcObjectIdx;
                },
                localExecutor
            );

            Sort(indices);

            return TArraySubsetIndexing<TSize>(std::move(indices));
        }
    }


    // TArrayLike must have O(1) random-access operator[].
    template <class TArrayLike, class TSize = size_t>
    class TArraySubset {
    public:
        TArraySubset(TArrayLike* src, const TArraySubsetIndexing<TSize>* subsetIndexing)
            : Src(src)
            , SubsetIndexing(subsetIndexing)
        {
            CB_ENSURE(Src, "TArraySubset constructor: src argument is nullptr");
            CB_ENSURE(SubsetIndexing, "TArraySubset constructor: subsetIndexing argument is nullptr");
        }

        TSize Size() const {
            return SubsetIndexing->Size();
        }

        // f is a visitor function that will be repeatedly called with (index, element) arguments
        template <class F>
        void ForEach(F&& f) {
            SubsetIndexing->ForEach(
                [src = this->Src, f = std::move(f)](TSize index, TSize srcIndex) {
                    f(index, (*src)[srcIndex]);
                }
            );
        };

        template <class F>
        void ForEach(F&& f) const {
            SubsetIndexing->ForEach(
                [src = this->Src, f = std::move(f)](TSize index, TSize srcIndex) {
                    f(index, (*(const TArrayLike*)src)[srcIndex]);
                }
            );
        };

        /* predicate is a visitor function that returns bool
         * it will be repeatedly called with (index, srcIndex) arguments
         * until it returns true or all elements are iterated over
         *
         *  @returns true if a pair of (index, element) for which predicate returned true was found
         *  and false otherwise
         */
        template <class TPredicate>
        bool Find(TPredicate&& predicate) const {
            return SubsetIndexing->Find(
                [src = this->Src, predicate = std::move(predicate)](TSize index, TSize srcIndex) {
                    return predicate(index, (*(const TArrayLike*)src)[srcIndex]);
                }
            );
        }

        /*
         * f is a visitor function that will be repeatedly called with (index, element) arguments
         *
         * for TRangesSubset block sizes might not be equal to approximateBlockSize because
         * block sizes might not be divisible by approximateBlockSize, that's why 'approximate'
         * is in the name of this block size parameter
         * if approximateBlockSize is undefined divide data approximately evenly between localExecutor
         * threads
         */
        template <class F>
        void ParallelForEach(
            F&& f,
            NPar::TLocalExecutor* localExecutor,
            TMaybe<TSize> approximateBlockSize = Nothing()
        ) {
            SubsetIndexing->ParallelForEach(
                [src = this->Src, f = std::move(f)](TSize index, TSize srcIndex) {
                    f(index, (*src)[srcIndex]);
                },
                localExecutor,
                approximateBlockSize
            );
        };

        template <class F>
        void ParallelForEach(
            F&& f,
            NPar::TLocalExecutor* localExecutor,
            TMaybe<TSize> approximateBlockSize = Nothing()
        ) const {
            SubsetIndexing->ParallelForEach(
                [src = this->Src, f = std::move(f)](TSize index, TSize srcIndex) {
                    f(index, (*(const TArrayLike*)src)[srcIndex]);
                },
                localExecutor,
                approximateBlockSize
            );
        };

        // might be used if fine-grained control is needed
        TArrayLike* GetSrc() noexcept {
            return Src;
        }
        const TArrayLike* GetSrc() const noexcept {
            return Src;
        }

        // might be used if fine-grained control is needed
        const TArraySubsetIndexing<TSize>* GetSubsetIndexing() const noexcept {
            return SubsetIndexing;
        }

    private:
        TArrayLike* Src;
        const TArraySubsetIndexing<TSize>* SubsetIndexing;
    };


    template <class T, class TArrayLike, class TSize=size_t>
    bool Equal(TConstArrayRef<T> lhs, const TArraySubset<TArrayLike, TSize>& rhs) {
        if (lhs.size() != rhs.Size()) {
            return false;
        }

        return !rhs.Find([&](TSize idx, T element) { return element != lhs[idx]; });
    }

    template <class T, class TArrayLike, class TSize=size_t>
    bool EqualWithNans(TConstArrayRef<T> lhs, const TArraySubset<TArrayLike, TSize>& rhs) {
        if (lhs.size() != rhs.Size()) {
            return false;
        }

        return !rhs.Find([&](TSize idx, T element) { return !EqualWithNans(element, lhs[idx]); });
    }

    template <class TDst, class TSrcArrayLike, class TSize=size_t>
    inline TVector<TDst> GetSubset(
        const TSrcArrayLike& srcArrayLike,
        const TArraySubsetIndexing<TSize>& subsetIndexing,
        TMaybe<NPar::TLocalExecutor*> localExecutor = Nothing(), // use parallel implementation if defined
        TMaybe<TSize> approximateBlockSize = Nothing() // for parallel version
    ) {
        TVector<TDst> dst;
        dst.yresize(subsetIndexing.Size());

        TArraySubset<const TSrcArrayLike, TSize> arraySubset(&srcArrayLike, &subsetIndexing);
        if (localExecutor.Defined()) {
            arraySubset.ParallelForEach(
                [&dst](TSize idx, TDst srcElement) { dst[idx] = srcElement; },
                *localExecutor,
                approximateBlockSize
            );
        } else {
            arraySubset.ForEach(
                [&dst](TSize idx, TDst srcElement) { dst[idx] = srcElement; }
            );
        }

        return dst;
    }

    template <class T, class TMaybePolicy, class TSize=size_t>
    inline TMaybe<TVector<T>, TMaybePolicy> GetSubsetOfMaybeEmpty(
        TMaybe<TConstArrayRef<T>, TMaybePolicy> src,
        const TArraySubsetIndexing<TSize>& subsetIndexing,
        TMaybe<NPar::TLocalExecutor*> localExecutor = Nothing(), // use parallel implementation if defined
        TMaybe<TSize> approximateBlockSize = Nothing() // for parallel version
    ) {
        if (!src) {
            return Nothing();
        } else {
            return GetSubset<T>(*src, subsetIndexing, localExecutor, approximateBlockSize);
        }
    }


    template <class T, class TSize=size_t>
    using TMaybeOwningArraySubset = TArraySubset<TMaybeOwningArrayHolder<T>, TSize>;

    template <class T, class TSize=size_t>
    using TMaybeOwningConstArraySubset = TArraySubset<const TMaybeOwningArrayHolder<const T>, TSize>;


    /* TArrayLike must have O(1) random-access operator[] and be lightweight copyable like
     *      T* or TArrayRef<T> or TMaybeOwningArrayHolder<T>
     *  it's elements type is not necessarily required to be equal to TDstValue, only convertible to it
     *
     *  SubsetIndexingIterator is a template parameter instead of IDynamicIteratorPtr to allow inlining
     *    for concrete iterator types to avoid double dynamic dispatch.
     */
    template <class TDstValue, class TArrayLike, class TSubsetIndexingIterator, class TTransformer>
    class TArraySubsetBlockIterator final : public IDynamicBlockIterator<TDstValue> {
    public:
        TArraySubsetBlockIterator(
            TArrayLike src,
            size_t subsetSize,
            TSubsetIndexingIterator&& subsetIndexingIterator,
            TTransformer&& transformer
        )
            : Src(std::move(src))
            , RemainingSize(subsetSize)
            , SubsetIndexingIterator(std::move(subsetIndexingIterator))
            , Transformer(std::move(transformer))
        {}

        TConstArrayRef<TDstValue> Next(size_t maxBlockSize = Max<size_t>()) override {
            const size_t dstBlockSize = Min(maxBlockSize, RemainingSize);
            Buffer.yresize(dstBlockSize);
            for (auto& dstElement : Buffer) {
                dstElement = Transformer(Src[*SubsetIndexingIterator.Next()]);
            }
            RemainingSize -= dstBlockSize;
            return Buffer;
        }

    private:
        TArrayLike Src;
        size_t RemainingSize;
        TSubsetIndexingIterator SubsetIndexingIterator;
        TVector<TDstValue> Buffer;
        TTransformer Transformer;
    };

    template <class TDstValue, class TArrayLike, class TTransformer>
    IDynamicBlockIteratorPtr<TDstValue> MakeTransformingArraySubsetBlockIterator(
        const TArraySubsetIndexing<ui32>* subsetIndexing,
        TArrayLike src,
        ui32 offset,
        TTransformer&& transformer
    ) {
        const ui32 size = subsetIndexing->Size();
        const ui32 remainingSize = size - offset;

        switch (subsetIndexing->index()) {
            case TVariantIndexV<TFullSubset<ui32>, TArraySubsetIndexing<ui32>::TBase>:
                return MakeHolder<
                        TArraySubsetBlockIterator<TDstValue, TArrayLike, TRangeIterator<ui32>, TTransformer>
                    >(
                        std::move(src),
                        remainingSize,
                        TRangeIterator<ui32>(TIndexRange<ui32>(offset, size)),
                        std::move(transformer)
                    );
            case TVariantIndexV<TRangesSubset<ui32>, TArraySubsetIndexing<ui32>::TBase>:
                return MakeHolder<
                        TArraySubsetBlockIterator<TDstValue, TArrayLike, TRangesSubsetIterator<ui32>, TTransformer>
                    >(
                        std::move(src),
                        remainingSize,
                        TRangesSubsetIterator<ui32>(subsetIndexing->Get<TRangesSubset<ui32>>(), offset),
                        std::move(transformer)
                    );
            case TVariantIndexV<TIndexedSubset<ui32>, TArraySubsetIndexing<ui32>::TBase>:
                {
                    using TIterator = TStaticIteratorRangeAsDynamic<const ui32*>;

                    const auto& indexedSubset = subsetIndexing->Get<TIndexedSubset<ui32>>();

                    return MakeHolder<TArraySubsetBlockIterator<TDstValue, TArrayLike, TIterator, TTransformer>>(
                        std::move(src),
                        remainingSize,
                        TIterator(indexedSubset.begin() + offset, indexedSubset.end()),
                        std::move(transformer)
                    );
                }
            default:
                Y_UNREACHABLE();
        }
        Y_UNREACHABLE();
    }

    template <class TDstValue, class TArrayLike>
    IDynamicBlockIteratorPtr<TDstValue> MakeArraySubsetBlockIterator(
        const TArraySubsetIndexing<ui32>* subsetIndexing,
        TArrayLike src,
        ui32 offset
    ) {
        return MakeTransformingArraySubsetBlockIterator<TDstValue, TArrayLike, TIdentity>(subsetIndexing, src, offset, TIdentity());
    }

    // index in dst data or NOT_PRESENT if not present in subset
    template <class TSize>
    class TInvertedIndexedSubset {
    public:
        constexpr static TSize NOT_PRESENT = Max<TSize>();

    public:
        TInvertedIndexedSubset(TSize size, TVector<TSize>&& mapping)
            : Size(size)
            , Mapping(std::move(mapping))
        {
            CB_ENSURE_INTERNAL(Size <= Mapping.size(), "Mapping size is smaller than subset size");
        }

        bool operator ==(const TInvertedIndexedSubset<TSize>& rhs) const {
            return (Size == rhs.Size) && (Mapping == rhs.Mapping);
        }

        TSize GetSize() const {
            return Size;
        }

        // srcIdx -> dstIdx or NOT_PRESENT
        TConstArrayRef<TSize> GetMapping() const {
            return Mapping;
        }

    private:
        TSize Size;
        TVector<TSize> Mapping;
    };


    template <class TSize>
    class TArraySubsetInvertedIndexing
        : public TVariant<TFullSubset<TSize>, TInvertedIndexedSubset<TSize>>
    {
    public:
        using TBase = TVariant<TFullSubset<TSize>, TInvertedIndexedSubset<TSize>>;

    public:
        explicit TArraySubsetInvertedIndexing(TFullSubset<TSize>&& subset)
            : TBase(std::move(subset))
        {}

        explicit TArraySubsetInvertedIndexing(TInvertedIndexedSubset<TSize>&& subset)
            : TBase(std::move(subset))
        {}

        friend bool operator ==(const TArraySubsetInvertedIndexing& a, const TArraySubsetInvertedIndexing& b) {
            return static_cast<const TBase&>(a) == static_cast<const TBase&>(b);
        }

        template <class T>
        decltype(auto) Get() const {
            return ::Get<T>((const TBase&)*this);
        }

        TSize GetSize() const {
            return Visit(
                [&] (const auto& variant) { return variant.GetSize(); },
                *this
            );
        }

        TSize GetSrcSize() const {
            switch (TBase::index()) {
                case TVariantIndexV<TFullSubset<TSize>, TBase>:
                    return Get<TFullSubset<TSize>>().GetSize();
                case TVariantIndexV<TInvertedIndexedSubset<TSize>, TBase>:
                    return Get<TInvertedIndexedSubset<TSize>>().GetMapping().size();
            }
            Y_UNREACHABLE();
        }
    };

    template <class TSize>
    TArraySubsetInvertedIndexing<TSize> GetInvertedIndexing(
        const TArraySubsetIndexing<TSize>& indexing,
        TSize srcSize,
        NPar::TLocalExecutor* localExecutor
    ) {
        if (indexing.index()
            == TVariantIndexV<TFullSubset<TSize>, typename TArraySubsetIndexing<TSize>::TBase>)
        {
            return TArraySubsetInvertedIndexing<TSize>(
                TFullSubset<TSize>(Get<TFullSubset<TSize>>(indexing))
            );
        }

        TVector<TSize> invertedIndices;
        invertedIndices.yresize(srcSize);
        ParallelFill(
            TInvertedIndexedSubset<TSize>::NOT_PRESENT,
            /*blockSize*/ Nothing(),
            localExecutor,
            MakeArrayRef(invertedIndices)
        );

        indexing.ParallelForEach(
            [&] (TSize dstIdx, TSize srcIdx) {
                // check duplicates
                Y_ASSERT(invertedIndices[srcIdx] == TInvertedIndexedSubset<TSize>::NOT_PRESENT);
                invertedIndices[srcIdx] = dstIdx;
            },
            localExecutor
        );

        return TArraySubsetInvertedIndexing<TSize>(
            TInvertedIndexedSubset<TSize>(indexing.Size(), std::move(invertedIndices))
        );
    }

    template <class TSize = size_t>
    TArraySubsetInvertedIndexing<TSize> Compose(
        const TArraySubsetInvertedIndexing<TSize>& src,
        TArraySubsetInvertedIndexing<TSize>&& srcSubset,
        NPar::TLocalExecutor* localExecutor
    ) {
        CB_ENSURE_INTERNAL(
            src.GetSize() >= srcSubset.GetSrcSize(),
            "srcSubset has source mapping size greater than src's size"
        );
        if (HoldsAlternative<TFullSubset<TSize>>(src)) {
            return std::move(srcSubset);
        }
        if (HoldsAlternative<TFullSubset<TSize>>(srcSubset)) {
            return src;
        }

        TConstArrayRef<TSize> srcIndexing = Get<TInvertedIndexedSubset<TSize>>(src).GetMapping();
        TConstArrayRef<TSize> srcSubsetIndexing = Get<TInvertedIndexedSubset<TSize>>(srcSubset).GetMapping();

        TVector<TSize> dstIndexing;
        dstIndexing.yresize(srcIndexing.size());

        TArrayRef<TSize> dstIndexingRef = dstIndexing;

        NPar::ParallelFor(
            *localExecutor,
            0,
            SafeIntegerCast<ui32>(srcIndexing.size()),
            [srcIndexing, srcSubsetIndexing, dstIndexingRef] (int srcIndex) {
                auto subsetIndex = srcIndexing[srcIndex];
                if (subsetIndex < srcSubsetIndexing.size()) {
                    dstIndexingRef[srcIndex] = srcSubsetIndexing[subsetIndex];
                } else {
                    dstIndexingRef[srcIndex] = TInvertedIndexedSubset<TSize>::NOT_PRESENT;
                }
            }
        );

        return TArraySubsetInvertedIndexing<TSize>(
            TInvertedIndexedSubset<TSize>(srcSubset.GetSize(), std::move(dstIndexing))
        );
    }

    template <class TS, class TSize>
    class TDumperArraySubsetInvertedIndexingVisitor {
    public:
        TDumperArraySubsetInvertedIndexingVisitor(TS& s)
            : S(s)
        {}

        void operator()(const TFullSubset<TSize>& fullSubset) const {
            S << "FullSubset(size=" << fullSubset.Size << ")\n";
        }

        void operator()(const TInvertedIndexedSubset<TSize>& invertedIndexedSubset) const {
            S << "InvertedIndexedSubset(Size=" << invertedIndexedSubset.GetSize() << ", Mapping=[\n";

            TConstArrayRef<TSize> mapping = invertedIndexedSubset.GetMapping();
            for (auto i : xrange(mapping.size())) {
                S << "\t" << i << " -> ";
                if (mapping[i] == TInvertedIndexedSubset<TSize>::NOT_PRESENT) {
                    S << "NOT_PRESENT";
                } else {
                    S << mapping[i];
                }
                S << Endl;
            }
            S << "])\n";
        }

    private:
        TS& S;
    };

}

template <class TSize>
struct TDumper<NCB::TArraySubsetInvertedIndexing<TSize>> {
    template <class S>
    static inline void Dump(S& s, const NCB::TArraySubsetInvertedIndexing<TSize>& invertedSubset) {
        Visit(NCB::TDumperArraySubsetInvertedIndexingVisitor<S, TSize>(s), invertedSubset);
    }
};

