#pragma once

#include "array_subset.h"
#include "dynamic_iterator.h"
#include "exception.h"
#include "maybe_owning_array_holder.h"

#include <catboost/private/libs/index_range/index_range.h>

#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/generic/algorithm.h>
#include <util/generic/array_ref.h>
#include <util/generic/cast.h>
#include <util/generic/maybe.h>
#include <util/generic/ptr.h>
#include <util/generic/ymath.h>
#include <util/system/compiler.h>
#include <util/system/types.h>
#include <util/system/yassert.h>

#include <type_traits>


namespace NCB {

    template <class T>
    class ITypedArraySubset : public TThrRefBase {
    public:
        virtual ui32 GetSize() const = 0;

        virtual IDynamicBlockIteratorPtr<T> GetBlockIterator(ui32 offset = 0) const = 0;

        virtual TIntrusivePtr<ITypedArraySubset<T>> CloneWithNewSubsetIndexing(
            const TArraySubsetIndexing<ui32>* newSubsetIndexing
        ) const = 0;

        // f is a visitor function that will be repeatedly called with (index, srcIndex) arguments
        template <class F>
        void ForEach(F&& f) const {
            IDynamicBlockIteratorPtr<T> blockIterator = GetBlockIterator();

            ui32 idx = 0;
            while (auto block = blockIterator->Next()) {
                for (auto element : block) {
                    f(idx++, element);
                }
            }
        }

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
            NPar::ILocalExecutor* localExecutor,
            TMaybe<ui32> approximateBlockSize = Nothing()
        ) const {
            TVector<IDynamicBlockIteratorPtr<T>> subRangeIterators;
            TVector<ui32> subRangeStarts;

            CreateSubRangesIterators(
                *localExecutor,
                approximateBlockSize,
                &subRangeIterators,
                &subRangeStarts
            );

            localExecutor->ExecRangeWithThrow(
                [&] (int subRangeIdx) {
                    IDynamicBlockIteratorPtr<T> subRangeIterator = std::move(subRangeIterators[subRangeIdx]);
                    ui32 idx = subRangeStarts[subRangeIdx];

                    while (auto block = subRangeIterator->Next()) {
                        for (auto element : block) {
                            f(idx++, element);
                        }
                    }
                },
                0,
                SafeIntegerCast<int>(subRangeIterators.size()),
                NPar::TLocalExecutor::WAIT_COMPLETE
            );
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
            IDynamicBlockIteratorPtr<T> blockIterator = GetBlockIterator();

            ui32 idx = 0;
            while (auto block = blockIterator->Next()) {
                for (auto element : block) {
                    if (predicate(idx++, element)) {
                        return true;
                    }
                }
            }
            return false;
        }

    protected:
        virtual void CreateSubRangesIterators(
            const NPar::ILocalExecutor& localExecutor,
            TMaybe<ui32> approximateBlockSize,
            TVector<IDynamicBlockIteratorPtr<T>>* subRangeIterators,
            TVector<ui32>* subRangeStarts
        ) const = 0;
    };

    template <class T>
    using ITypedArraySubsetPtr = TIntrusivePtr<ITypedArraySubset<T>>;

    template <class TSrc, class TDst>
    struct TStaticCast {
        constexpr TDst operator()(const TSrc& x) const {
            return TDst(x);
        }
    };


    template <
        class TInterfaceValue,
        class TStoredValue,
        class TTransformer = TStaticCast<TStoredValue, TInterfaceValue>
    >
    class TTypeCastArraySubset final : public ITypedArraySubset<TInterfaceValue> {
    public:
        using TData = TMaybeOwningConstArrayHolder<TStoredValue>;

    public:
        TTypeCastArraySubset(
            TMaybeOwningConstArrayHolder<TStoredValue> data,
            const TArraySubsetIndexing<ui32>* subsetIndexing
        )
            : Data(std::move(data))
            , SubsetIndexing(subsetIndexing)
        {
            Y_ASSERT(SubsetIndexing);
        }

        ui32 GetSize() const override {
            return SubsetIndexing->Size();
        }

        IDynamicBlockIteratorPtr<TInterfaceValue> GetBlockIterator(ui32 offset = 0) const override {
            return MakeTransformingArraySubsetBlockIterator<TInterfaceValue>(
                SubsetIndexing,
                Data,
                offset,
                TTransformer()
            );
        }

        TIntrusivePtr<ITypedArraySubset<TInterfaceValue>> CloneWithNewSubsetIndexing(
            const TArraySubsetIndexing<ui32>* newSubsetIndexing
        ) const override {
            return MakeIntrusive<TTypeCastArraySubset<TInterfaceValue, TStoredValue, TTransformer>>(
                Data,
                newSubsetIndexing
            );
        }

    protected:
        void CreateSubRangesIterators(
            const TFullSubset<ui32>& fullSubset,
            ui32 approximateBlockSize,
            TVector<IDynamicBlockIteratorPtr<TInterfaceValue>>* subRangeIterators,
            TVector<ui32>* subRangeStarts
        ) const {
            TSimpleIndexRangesGenerator<ui32> indexRanges(
                TIndexRange<ui32>(fullSubset.Size),
                approximateBlockSize
            );
            subRangeIterators->reserve(indexRanges.RangesCount());
            subRangeStarts->reserve(indexRanges.RangesCount());

            for (auto subRangeIdx : xrange(indexRanges.RangesCount())) {
                auto subRange = indexRanges.GetRange(subRangeIdx);
                subRangeIterators->push_back(
                    MakeHolder<
                        TArraySubsetBlockIterator<TInterfaceValue, TData, TRangeIterator<ui32>, TTransformer>
                    >(
                        Data,
                        subRange.GetSize(),
                        TRangeIterator<ui32>(subRange),
                        TTransformer()
                    )
                );
                subRangeStarts->push_back(subRange.Begin);
            }
        }

        void CreateSubRangesIterators(
            const TRangesSubset<ui32>& rangesSubset,
            ui32 approximateBlockSize,
            TVector<IDynamicBlockIteratorPtr<TInterfaceValue>>* subRangeIterators,
            TVector<ui32>* subRangeStarts
        ) const {
            // TODO(akhropov): don't join small blocks (rare case in practice) for simplicity
            for (const auto& block : rangesSubset.Blocks) {
                TSimpleIndexRangesGenerator<ui32> indexRanges(
                    TIndexRange<ui32>(block.GetSize()),
                    approximateBlockSize
                );

                for (auto subRangeIdx : xrange(indexRanges.RangesCount())) {
                    auto subRange = indexRanges.GetRange(subRangeIdx);
                    subRangeIterators->push_back(
                        MakeHolder<
                            TArraySubsetBlockIterator<TInterfaceValue, TData, TRangesSubsetIterator<ui32>, TTransformer>
                        >(
                            Data,
                            subRange.GetSize(),
                            TRangesSubsetIterator<ui32>(
                                &block,
                                subRange.Begin,
                                &block + 1,
                                subRange.End
                            ),
                            TTransformer()
                        )
                    );
                    subRangeStarts->push_back(block.DstBegin + subRange.Begin);
                }
            }
        }

        void CreateSubRangesIterators(
            const TIndexedSubset<ui32>& indexedSubset,
            ui32 approximateBlockSize,
            TVector<IDynamicBlockIteratorPtr<TInterfaceValue>>* subRangeIterators,
            TVector<ui32>* subRangeStarts
        ) const {
            TSimpleIndexRangesGenerator<ui32> indexRanges(
                TIndexRange<ui32>(GetSize()),
                approximateBlockSize
            );
            subRangeIterators->reserve(indexRanges.RangesCount());
            subRangeStarts->reserve(indexRanges.RangesCount());

            using TIterator = TStaticIteratorRangeAsDynamic<const ui32*>;
            const ui32* indexedSubsetBegin = indexedSubset.data();

            for (auto subRangeIdx : xrange(indexRanges.RangesCount())) {
                auto subRange = indexRanges.GetRange(subRangeIdx);
                subRangeIterators->push_back(
                    MakeHolder<
                        TArraySubsetBlockIterator<TInterfaceValue, TData, TIterator, TTransformer>
                    >(
                        Data,
                        subRange.GetSize(),
                        TIterator(indexedSubsetBegin + subRange.Begin, indexedSubsetBegin + subRange.End),
                        TTransformer()
                    )
                );
                subRangeStarts->push_back(subRange.Begin);
            }
        }

        void CreateSubRangesIterators(
            const NPar::ILocalExecutor& localExecutor,
            TMaybe<ui32> approximateBlockSize,
            TVector<IDynamicBlockIteratorPtr<TInterfaceValue>>* subRangeIterators,
            TVector<ui32>* subRangeStarts
        ) const override {
            const ui32 size = GetSize();
            if (!size) {
                subRangeIterators->clear();
                subRangeStarts->clear();
                return;
            }

            ui32 definedApproximateBlockSize
                = approximateBlockSize ?
                    *approximateBlockSize
                    : CeilDiv(size, SafeIntegerCast<ui32>(localExecutor.GetThreadCount()) + 1);

            std::visit(
                [&] (const auto& subsetIndexingAlternative) {
                    CreateSubRangesIterators(
                        subsetIndexingAlternative,
                        definedApproximateBlockSize,
                        subRangeIterators,
                        subRangeStarts
                    );
                },
                *SubsetIndexing
            );
        }

        const TData& GetData() const {
            return Data;
        }

        const TArraySubsetIndexing<ui32>* GetSubsetIndexing() const {
            return SubsetIndexing;
        }

    private:
        TData Data;
        const TArraySubsetIndexing<ui32>* SubsetIndexing;
    };


    template <class T>
    class ITypedSequence : public TThrRefBase {
    public:
        virtual int operator&(IBinSaver& binSaver) = 0;

        // comparison is strict by default, useful for unit tests
        bool operator==(const ITypedSequence<T>& rhs) const {
            return EqualTo(rhs, /*strict*/ true);
        }

        // if strict is true compare bit-by-bit, else compare values
        virtual bool EqualTo(const ITypedSequence<T>& rhs, bool strict = true) const = 0;

        virtual ui32 GetSize() const = 0;

        virtual IDynamicBlockWithExactIteratorPtr<T> GetBlockIterator(TIndexRange<ui32> indexRange) const = 0;

        IDynamicBlockWithExactIteratorPtr<T> GetBlockIterator() const {
            return GetBlockIterator(TIndexRange<ui32>(GetSize()));
        }

        virtual TIntrusivePtr<ITypedArraySubset<T>> GetSubset(
            const TArraySubsetIndexing<ui32>* subsetIndexing
        ) const = 0;

        // f is a visitor function that will be repeatedly called with (element) argument
        template <class F>
        void ForEach(F&& f) const {
            IDynamicBlockIteratorPtr<T> blockIterator = GetBlockIterator();
            while (auto block = blockIterator->Next()) {
                for (auto element : block) {
                    f(element);
                }
            }
        }
    };

    template <class T>
    using ITypedSequencePtr = TIntrusivePtr<ITypedSequence<T>>;

    /* dst points to preallocated buffer, such interface allows to write data to external storage
     * like numpy.ndarray
     */
    template <class T>
    void ToArray(const ITypedSequence<T>& typedSequence, TArrayRef<T> dst) {
        CB_ENSURE_INTERNAL(
            (size_t)typedSequence.GetSize() == dst.size(),
            "ToArray for ITypedSequence: Wrong dst array size"
        );
        size_t i = 0;
        typedSequence.ForEach([&i, dst] (T value) { dst[i++] = value; });
    }

    template <class T>
    TVector<T> ToVector(const ITypedSequence<T>& typedSequence) {
        TVector<T> dst;
        dst.yresize(typedSequence.GetSize());
        ToArray<T>(typedSequence, dst);
        return dst;
    }


    template <
        class TInterfaceValue,
        class TStoredValue,
        class TTransformer = TStaticCast<TStoredValue, TInterfaceValue>
    >
    class TTypeCastArrayHolder final : public ITypedSequence<TInterfaceValue> {
    public:
        explicit TTypeCastArrayHolder(TMaybeOwningConstArrayHolder<TStoredValue> values)
            : Values(std::move(values))
        {}

        explicit TTypeCastArrayHolder(TVector<TStoredValue>&& values)
            : Values(TMaybeOwningConstArrayHolder<TStoredValue>::CreateOwning(std::move(values)))
        {}

        int operator&(IBinSaver& binSaver) override {
            binSaver.Add(0, &Values);
            return 0;
        }

        bool EqualTo(const ITypedSequence<TInterfaceValue>& rhs, bool strict = true) const override {
            if (strict) {
                if (const auto* rhsAsThisType
                        = dynamic_cast<const TTypeCastArrayHolder<TInterfaceValue, TStoredValue, TTransformer>*>(&rhs))
                {
                    return Values == rhsAsThisType->Values;
                } else {
                    return false;
                }
            } else {
                return AreBlockedSequencesEqual<TInterfaceValue, TInterfaceValue>(
                    ITypedSequence<TInterfaceValue>::GetBlockIterator(),
                    rhs.ITypedSequence<TInterfaceValue>::GetBlockIterator()
                );
            }
        }

        ui32 GetSize() const override {
            return SafeIntegerCast<ui32>(Values.GetSize());
        }

        IDynamicBlockWithExactIteratorPtr<TInterfaceValue> GetBlockIterator(
            TIndexRange<ui32> indexRange
        ) const override {
            if constexpr (std::is_same_v<TTransformer, TStaticCast<TInterfaceValue, TStoredValue>>) {
                TConstArrayRef<TStoredValue> subRangeArrayRef(
                    Values.begin() + indexRange.Begin,
                    Values.begin() + indexRange.End
                );
                if constexpr (std::is_same_v<TInterfaceValue, TStoredValue>) {
                    return MakeHolder<TArrayBlockIterator<TInterfaceValue>>(subRangeArrayRef);
                } else {
                    return MakeHolder<TTypeCastingArrayBlockIterator<TInterfaceValue, TStoredValue>>(
                        subRangeArrayRef
                    );
                }
            } else {
                return MakeHolder<
                    TArraySubsetBlockIterator<
                        TInterfaceValue,
                        TConstArrayRef<TStoredValue>,
                        TRangeIterator<ui32>,
                        TTransformer
                    >
                >(
                    *Values,
                    indexRange.GetSize(),
                    TRangeIterator<ui32>(indexRange),
                    TTransformer()
                );
            }
        }

        TIntrusivePtr<ITypedArraySubset<TInterfaceValue>> GetSubset(
            const TArraySubsetIndexing<ui32>* subsetIndexing
        ) const override {
            return MakeIntrusive<TTypeCastArraySubset<TInterfaceValue, TStoredValue, TTransformer>>(
                Values,
                subsetIndexing
            );
        }

    private:
        TMaybeOwningConstArrayHolder<TStoredValue> Values;
    };


    // for Cython where MakeIntrusive cannot be used
    template <class TInterfaceValue, class TStoredValue>
    ITypedSequencePtr<TInterfaceValue> MakeTypeCastArrayHolder(
        TMaybeOwningConstArrayHolder<TStoredValue> values
    ) {
        return MakeIntrusive<TTypeCastArrayHolder<TInterfaceValue, TStoredValue>>(std::move(values));
    }


    template <class TInterfaceValue, class TStoredValue>
    ITypedSequencePtr<TInterfaceValue> MakeNonOwningTypeCastArrayHolder(
        const TStoredValue* begin,
        const TStoredValue* end
    ) {
        return MakeIntrusive<TTypeCastArrayHolder<TInterfaceValue, TStoredValue>>(
            TMaybeOwningConstArrayHolder<TStoredValue>::CreateNonOwning(
                TConstArrayRef<TStoredValue>(begin, end)
            )
        );
    }

    // for Cython where MakeHolder cannot be used
    template <class TInterfaceValue, class TStoredValue>
    ITypedSequencePtr<TInterfaceValue> MakeTypeCastArrayHolderFromVector(TVector<TStoredValue>& values) {
        return MakeIntrusive<TTypeCastArrayHolder<TInterfaceValue, TStoredValue>>(
            TMaybeOwningConstArrayHolder<TStoredValue>::CreateOwning(std::move(values))
        );
    }

    template <class TSrc, class TDst>
    struct TMaybeOwningArrayHolderCast {
        constexpr TMaybeOwningConstArrayHolder<TDst> operator()(
            const TMaybeOwningConstArrayHolder<TSrc>& x
        ) const {
            TVector<TDst> result;
            result.yresize(x.GetSize());
            Copy(x.begin(), x.end(), result.begin());
            return TMaybeOwningConstArrayHolder<TDst>::CreateOwning(std::move(result));
        }
    };

    template <class TInterfaceValue, class TStoredValue>
    ITypedSequencePtr<TMaybeOwningConstArrayHolder<TInterfaceValue>> MakeTypeCastArraysHolderFromVector(
        TVector<TMaybeOwningConstArrayHolder<TStoredValue>>& values
    ) {
        if constexpr (std::is_same_v<TInterfaceValue, TStoredValue>) {
            return MakeTypeCastArrayHolderFromVector<TMaybeOwningConstArrayHolder<TInterfaceValue>>(values);
        } else {
            return MakeIntrusive<
                TTypeCastArrayHolder<
                    TMaybeOwningConstArrayHolder<TInterfaceValue>,
                    TMaybeOwningConstArrayHolder<TStoredValue>,
                    TMaybeOwningArrayHolderCast<TStoredValue, TInterfaceValue>
                >
            >(
                TMaybeOwningConstArrayHolder<TMaybeOwningConstArrayHolder<TStoredValue>>::CreateOwning(
                    std::move(values)
                )
            );
        }
    }
}
