#pragma once

#include "exception.h"
#include "index_range.h"
#include "maybe_owning_array_holder.h"

#include <library/threading/local_executor/local_executor.h>

#include <util/generic/array_ref.h>
#include <util/generic/maybe.h>
#include <util/generic/variant.h>
#include <util/generic/vector.h>
#include <util/generic/xrange.h>
#include <util/generic/ymath.h>

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
    };

    template <class TSize>
    class TArraySubsetIndexing
        : public TVariant<TFullSubset<TSize>, TRangesSubset<TSize>, TIndexedSubset<TSize>>
    {
        using TBase = TVariant<TFullSubset<TSize>, TRangesSubset<TSize>, TIndexedSubset<TSize>>;

    public:
        template <class T>
        explicit TArraySubsetIndexing(T&& subsetIndexingVariant)
            : TBase(std::move(subsetIndexingVariant))
        {}

        TSize Size() const {
            switch (TBase::Index()) {
                case TBase::template TagOf<TFullSubset<TSize>>():
                    return Get<TFullSubset<TSize>>().Size;
                case TBase::template TagOf<TRangesSubset<TSize>>():
                    return Get<TRangesSubset<TSize>>().Size;
                case TBase::template TagOf<TIndexedSubset<TSize>>():
                    return static_cast<TSize>(Get<TIndexedSubset<TSize>>().size());
            }
            return 0; // just to silence compiler warnings
        }

        // number of elements for TFullSubset or TIndexedSubset, number of ranges for TRangesSubset
        TSize GetParallelizableUnitsCount() const {
            switch (TBase::Index()) {
                case TBase::template TagOf<TFullSubset<TSize>>():
                    return Get<TFullSubset<TSize>>().Size;
                case TBase::template TagOf<TRangesSubset<TSize>>():
                    return Get<TRangesSubset<TSize>>().Blocks.size();
                case TBase::template TagOf<TIndexedSubset<TSize>>():
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
                    (TSize)std::llrint(
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
            switch (TBase::Index()) {
                case TBase::template TagOf<TFullSubset<TSize>>():
                    return unitRange;
                case TBase::template TagOf<TRangesSubset<TSize>>():
                    {
                        const auto& rangesSubset = Get<TRangesSubset<TSize>>();
                        const auto& blocks = rangesSubset.Blocks;
                        return NCB::TIndexRange<TSize>(
                            blocks[unitRange.Begin].DstBegin,
                            unitRange.End < blocks.size() ? blocks[unitRange.End].DstBegin : rangesSubset.Size
                        );
                    }
                    break;
                case TBase::template TagOf<TIndexedSubset<TSize>>():
                    return unitRange;
            }
            return NCB::TIndexRange<TSize>(0); // silence compiler warnings
        }

// need to be able to use 'break' or 'return' in iteration
#define LOOP_SUB_RANGE(unitSubRange, LOOP_BODY_MACRO) \
        switch (TBase::Index()) { \
            case TBase::template TagOf<TFullSubset<TSize>>(): \
                { \
                    for (TSize index : unitSubRange.Iter()) { \
                        LOOP_BODY_MACRO(index, index) \
                    } \
                } \
                break; \
            case TBase::template TagOf<TRangesSubset<TSize>>(): \
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
            case TBase::template TagOf<TIndexedSubset<TSize>>(): \
                { \
                    const auto& srcIndex = Get<TIndexedSubset<TSize>>(); \
                    for (TSize index : unitSubRange.Iter()) { \
                        LOOP_BODY_MACRO(index, srcIndex[index]) \
                    } \
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
            NPar::TLocalExecutor& localExecutor,
            F&& f,
            TMaybe<TSize> approximateBlockSize = Nothing()
        ) const {
            if (!approximateBlockSize.Defined()) {
                TSize localExecutorThreadsPlusCurrentCount = (TSize)localExecutor.GetThreadCount() + 1;
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

            localExecutor.ExecRangeWithThrow(
                [this, parallelUnitRanges, f = std::move(f)] (int id) {
                    ForEachInSubRange(parallelUnitRanges.GetRange(id), f);
                },
                0,
                (int)parallelUnitRanges.RangesCount(),
                NPar::TLocalExecutor::WAIT_COMPLETE
            );
        }
    };


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
            NPar::TLocalExecutor& localExecutor,
            F&& f,
            TMaybe<TSize> approximateBlockSize = Nothing()
        ) {
            SubsetIndexing->ParallelForEach(
                localExecutor,
                [src = this->Src, f = std::move(f)](TSize index, TSize srcIndex) {
                    f(index, (*src)[srcIndex]);
                },
                approximateBlockSize
            );
        };

        template <class F>
        void ParallelForEach(
            NPar::TLocalExecutor& localExecutor,
            F&& f,
            TMaybe<TSize> approximateBlockSize = Nothing()
        ) const {
            SubsetIndexing->ParallelForEach(
                localExecutor,
                [src = this->Src, f = std::move(f)](TSize index, TSize srcIndex) {
                    f(index, (*(const TArrayLike*)src)[srcIndex]);
                },
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


    template <class TDst, class TSrcArrayLike, class TSize=size_t>
    inline TVector<TDst> GetSubset(
        const TSrcArrayLike& srcArrayLike,
        const TArraySubsetIndexing<TSize>& subsetIndexing
    ) {
        TVector<TDst> dst;
        dst.yresize(subsetIndexing.Size());

        TArraySubset<const TSrcArrayLike, TSize> arraySubset(&srcArrayLike, &subsetIndexing);
        arraySubset.ForEach(
            [&dst](ui64 idx, TDst srcElement) { dst[idx] = srcElement; }
        );

        return dst;
    }

    // useful for optionally empty data
    template<class T, class TSize=size_t>
    inline TVector<T> GetSubsetOfMaybeEmpty(
        TConstArrayRef<T> src,
        const TArraySubsetIndexing<TSize>& subsetIndexing
    ) {
        if (src.empty()) {
            return TVector<T>();
        } else {
            return GetSubset<T>(src, subsetIndexing);
        }
    }


    template <class T, class TSize=size_t>
    using TMaybeOwningArraySubset = TArraySubset<TMaybeOwningArrayHolder<T>, TSize>;

    template <class T, class TSize=size_t>
    using TConstMaybeOwningArraySubset = TArraySubset<const TMaybeOwningArrayHolder<T>, TSize>;


    template <class TDst, class TSrcArrayLike, class TSize=size_t>
    inline TMaybeOwningArrayHolder<TDst> ParallelExtractValues(
        const TArraySubset<TSrcArrayLike, TSize>& arraySubset,
        TMaybe<NPar::TLocalExecutor*> localExecutor = Nothing()
    ) {
        TVector<TDst> dst;
        dst.yresize(arraySubset.Size());

        arraySubset.ParallelForEach(
            localExecutor.Defined() ? (NPar::TLocalExecutor&)**localExecutor : NPar::LocalExecutor(),
            [&dst](ui64 idx, TDst srcElement) { dst[idx] = srcElement; }
        );

        return TMaybeOwningArrayHolder<TDst>::CreateOwning(std::move(dst));
    }
}

