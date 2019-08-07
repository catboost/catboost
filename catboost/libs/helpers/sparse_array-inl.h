#pragma once

#include "array_subset.h"
#include "double_array_iterator.h"
#include "exception.h"
#include "serialization.h"

#include <library/pop_count/popcount.h>

#include <util/generic/bitops.h>
#include <util/generic/cast.h>
#include <util/generic/xrange.h>

#include <util/system/compiler.h>
#include <util/system/yassert.h>

namespace NCB {

    template <class T>
    void CheckIsIncreasingIndicesArray(TConstArrayRef<T> data, TStringBuf arrayName, bool isInternalError) {
        static_assert(std::is_integral<T>::value);

        for (auto i : xrange(data.size())) {
            if constexpr(std::is_signed<T>::value) {
                CB_ENSURE(
                    data[i] >= 0,
                    (isInternalError ? INTERNAL_ERROR_MSG : "") << " " << arrayName
                    << " at pos " << i << " contains negative index " << data[i]);
            }
            if (i) {
                CB_ENSURE(
                    data[i] > data[i-1],
                    (isInternalError ? INTERNAL_ERROR_MSG : "") << " " << arrayName
                    << " is not increasing (broken at pos " << i << ')');
            }
        }
    }

    template <class TSize>
    void TSparseSubsetBlocks<TSize>::Check() const {
        auto blockCount = (*BlockStarts).size();
        CB_ENSURE(
            blockCount == (*BlockLengths).size(),
            "Sparse Block Starts and Block Lengths arrays have different sizes");

        if (!blockCount) {
            return;
        }

        TSize lastBlockEndIndex = 0;
        for (auto i : xrange(blockCount)) {
            if constexpr(std::is_signed<TSize>::value) {
                CB_ENSURE((*BlockStarts)[i] >= 0, "Sparse Block Start array at pos " << i
                    << " contains negative index " << (*BlockStarts)[i]);
                CB_ENSURE((*BlockLengths)[i] >= 0, "Sparse Block Lengths array at pos " << i
                    << " contains negative index " << (*BlockLengths)[i]);
            }

            CB_ENSURE((*BlockStarts)[i] >= lastBlockEndIndex, "Sparse Block Start array at pos " << i
                << " contains index " << (*BlockStarts)[i]
                << " that is not greater than the last block end index " << lastBlockEndIndex);

            lastBlockEndIndex = (*BlockStarts)[i] + (*BlockLengths)[i];
        }
    }


    template <class TSize>
    TSparseSubsetBlocksIterator<TSize>::TSparseSubsetBlocksIterator(
        const TSparseSubsetBlocks<TSize>& sparseSubsetBlocks)
        : BlockStartsCurrent((*sparseSubsetBlocks.BlockStarts).begin())
        , BlockStartsEnd((*sparseSubsetBlocks.BlockStarts).end())
        , BlockLengthsCurrent((*sparseSubsetBlocks.BlockLengths).begin())
        , InBlockIdx(TSize(0))
    {}

    template <class TSize>
    inline TMaybe<TSize> TSparseSubsetBlocksIterator<TSize>::Next() {
        if (BlockStartsCurrent == BlockStartsEnd) {
            return Nothing();
        }
        while (InBlockIdx == *BlockLengthsCurrent) {
            if (++BlockStartsCurrent == BlockStartsEnd) {
                return Nothing();
            }
            ++BlockLengthsCurrent;
            InBlockIdx = 0;
        }
        return (*BlockStartsCurrent) + InBlockIdx++;
    }

    template <class TSize>
    void TSparseSubsetHybridIndex<TSize>::Check() const {
        auto blockCount = BlockIndices.size();
        CB_ENSURE_INTERNAL(
            blockCount == BlockBitmaps.size(),
            "TSparseSubsetHybridIndex: BlockIndices and BlockBitmaps have different sizes");

        CheckIsIncreasingIndicesArray<TSize>(BlockIndices, "TSparseSubsetHybridIndex: BlockIndices", true);
    }

    template <class TSize>
    TSize TSparseSubsetHybridIndex<TSize>::GetSize() const {
        TSize result = 0;
        for (auto blockBitmap : BlockBitmaps) {
            result += PopCount(blockBitmap);
        }
        return result;
    }

    template <class TSize>
    TSize TSparseSubsetHybridIndex<TSize>::GetUpperBound() const {
        if (BlockIndices.empty()) {
            return 0;
        }
        return BlockIndices.back() * BLOCK_SIZE
            + (BlockBitmaps.back() ? (MostSignificantBit(BlockBitmaps.back()) + 1) : 0);
    }


    template <class TSize>
    TSparseSubsetHybridIndexIterator<TSize>::TSparseSubsetHybridIndexIterator(
        const TSparseSubsetHybridIndex<TSize>& sparseSubsetHybridIndex)
        : BlockIndicesCurrent(sparseSubsetHybridIndex.BlockIndices.begin())
        , BlockIndicesEnd(sparseSubsetHybridIndex.BlockIndices.end())
        , BlockBitmapsCurrent(sparseSubsetHybridIndex.BlockBitmaps.begin())
        , InBlockIdx(ui32(0))
    {}

    template <class TSize>
    inline TMaybe<TSize> TSparseSubsetHybridIndexIterator<TSize>::Next() {
        if (BlockIndicesCurrent == BlockIndicesEnd) {
            return Nothing();
        }
        while (! (((*BlockBitmapsCurrent) >> InBlockIdx) & 1)) {
            ++InBlockIdx;
        }

        TMaybe<TSize> result
            = TSparseSubsetHybridIndex<TSize>::BLOCK_SIZE * (*BlockIndicesCurrent) + InBlockIdx;

        if (! ((*BlockBitmapsCurrent) >> (InBlockIdx + 1))) {
            ++BlockIndicesCurrent;
            ++BlockBitmapsCurrent;
            InBlockIdx = 0;
        } else {
            ++InBlockIdx;
        }

        return result;
    }


    template <class TSize>
    TSparseArrayIndexing<TSize>::TSparseArrayIndexing(TImpl&& impl, TMaybe<TSize> size, bool skipCheck)
        : Impl(std::move(impl))
        , NonDefaultSize(0) // properly inited later
        , Size(0) // properly inited later
    {
        Visit(
            [&](const auto& impl) {
                NonDefaultSize = impl.GetSize();
                InitSize(size, impl.GetUpperBound());
                if (!skipCheck) {
                    impl.Check();
                }
            },
            Impl);
    }

    template <class TSize>
    bool TSparseArrayIndexing<TSize>::EqualTo(const TSparseArrayIndexing<TSize>& rhs, bool strict) const {
        if (strict) {
            return std::tie(NonDefaultSize, Size, Impl) == std::tie(rhs.NonDefaultSize, rhs.Size, rhs.Impl);
        } else {
            if (std::tie(NonDefaultSize, Size) != std::tie(rhs.NonDefaultSize, rhs.Size)) {
                return false;
            }
            return AreSequencesEqual<TSize, TMaybe<TSize>>(GetIterator(), rhs.GetIterator());
        }
    }

    template <class TSize>
    ESparseArrayIndexingType TSparseArrayIndexing<TSize>::GetType() const {
        switch (Impl.index()) {
            case TVariantIndexV<TSparseSubsetIndices<TSize>, TImpl>:
                return ESparseArrayIndexingType::Indices;
            case TVariantIndexV<TSparseSubsetBlocks<TSize>, TImpl>:
                return ESparseArrayIndexingType::Blocks;
            case TVariantIndexV<TSparseSubsetHybridIndex<TSize>, TImpl>:
                return ESparseArrayIndexingType::HybridIndex;
        }
        Y_UNREACHABLE();
    }

    template <class TSize>
    template <class F>
    inline void TSparseArrayIndexing<TSize>::ForEachNonDefault(F &&f) const {
        auto iterator = GetIterator();
        while (auto next = iterator->Next()) {
            f(*next);
        }
    }

    template <class TSize>
    template <class F>
    inline void TSparseArrayIndexing<TSize>::ForEach(F &&f) const {
        auto nonDefaultIterator = GetIterator();
        TSize i = 0;
        while (auto nextNonDefault = nonDefaultIterator->Next()) {
            auto nextNonDefaultIndex = *nextNonDefault;
            for (; i < nextNonDefaultIndex; ++i) {
                f(i, /*isDefault*/ true);
            }
            f(i++, /*isDefault*/ false);
        }
        for (; i != Size; ++i) {
            f(i, /*isDefault*/ true);
        }
    }

    template <class TSize>
    IDynamicIteratorPtr<TSize> TSparseArrayIndexing<TSize>::GetIterator() const {
        switch (Impl.index()) {
            case TVariantIndexV<TSparseSubsetIndices<TSize>, TImpl>:
                return MakeHolder<TStaticIteratorRangeAsDynamic<const TSize*>>(
                    Get<TSparseSubsetIndices<TSize>>(Impl));
            case TVariantIndexV<TSparseSubsetBlocks<TSize>, TImpl>:
                return MakeHolder<TSparseSubsetBlocksIterator<TSize>>(Get<TSparseSubsetBlocks<TSize>>(Impl));
            case TVariantIndexV<TSparseSubsetHybridIndex<TSize>, TImpl>:
                return MakeHolder<TSparseSubsetHybridIndexIterator<TSize>>(
                    Get<TSparseSubsetHybridIndex<TSize>>(Impl));
            default:
                Y_UNREACHABLE();
        }
        Y_UNREACHABLE();
    }

    template <class TSize>
    void TSparseArrayIndexing<TSize>::InitSize(TMaybe<TSize> sizeArg, TSize indicesUpperBound) {
        if (sizeArg) {
            CB_ENSURE_INTERNAL(*sizeArg >= indicesUpperBound, "Bad size for TSparseArrayIndexing");
            Size = *sizeArg;
        } else {
            Size = indicesUpperBound;
        }
    }

    template <class TSize>
    inline void TSparseSubsetIndicesBuilder<TSize>::AddOrdered(TSize i) {
        Y_ASSERT(NonOrdered || Indices.empty() || (i > Indices.back()));
        Indices.push_back(i);
    }

    template <class TSize>
    inline void TSparseSubsetIndicesBuilder<TSize>::AddNonOrdered(TSize i) {
        NonOrdered = true;
        Indices.push_back(i);
    }

    template <class TSize>
    TSparseArrayIndexing<TSize> TSparseSubsetIndicesBuilder<TSize>::Build() {
        if (NonOrdered) {
            Sort(Indices);
        }
        return TSparseArrayIndexing<TSize>(TSparseSubsetIndices<TSize>(std::move(Indices)));
    }


    template <class TSize>
    inline void TSparseSubsetBlocksBuilder<TSize>::AddOrdered(TSize i) {
        AddImpl(i);
    }

    template <class TSize>
    inline void TSparseSubsetBlocksBuilder<TSize>::AddNonOrdered(TSize i) {
        NonOrdered = true;
        AddImpl(i);
    }

    template <class TSize>
    TSparseArrayIndexing<TSize> TSparseSubsetBlocksBuilder<TSize>::Build() {
        if (NonOrdered && (BlockStarts.size() > 1)) {
            TDoubleArrayIterator<TSize, TSize> beginIter{BlockStarts.begin(), BlockLengths.begin()};
            TDoubleArrayIterator<TSize, TSize> endIter{BlockStarts.end(), BlockLengths.end()};

            Sort(beginIter, endIter, [](auto lhs, auto rhs) { return lhs.first < rhs.first; });

            // compress blocks
            const TSize srcSize = BlockStarts.size();
            TSize dstIndex = 0;
            for (TSize srcIndex = 1; srcIndex < srcSize; ++srcIndex) {
                if (BlockStarts[srcIndex] == (BlockStarts[dstIndex] + BlockLengths[dstIndex])) {
                    BlockLengths[dstIndex] += BlockLengths[srcIndex];
                } else {
                    ++dstIndex;
                    BlockStarts[dstIndex] = BlockStarts[srcIndex];
                    BlockLengths[dstIndex] = BlockLengths[srcIndex];
                }
            }
            BlockStarts.resize(dstIndex + 1);
            BlockStarts.shrink_to_fit();
            BlockLengths.resize(dstIndex + 1);
            BlockLengths.shrink_to_fit();
        }
        return TSparseArrayIndexing<TSize>(
            TSparseSubsetBlocks<TSize>{std::move(BlockStarts), std::move(BlockLengths)}
        );
    }

    template <class TSize>
    inline void TSparseSubsetBlocksBuilder<TSize>::AddImpl(TSize i) {
        if (BlockStarts.empty() || ((BlockStarts.back() + BlockLengths.back()) != i)) {
            BlockStarts.push_back(i);
            BlockLengths.push_back(1);
        } else {
            ++BlockLengths.back();
        }
    }

    template <class TSize>
    inline void TSparseSubsetHybridIndexBuilder<TSize>::AddOrdered(TSize i) {
        AddImpl(i);
    }

    template <class TSize>
    inline void TSparseSubsetHybridIndexBuilder<TSize>::AddNonOrdered(TSize i) {
        NonOrdered = true;
        AddImpl(i);
    }

    template <class TSize>
    TSparseArrayIndexing<TSize> TSparseSubsetHybridIndexBuilder<TSize>::Build() {
        if (NonOrdered && (BlockIndices.size() > 1)) {
            TDoubleArrayIterator<TSize, ui64> beginIter{BlockIndices.begin(), BlockBitmaps.begin()};
            TDoubleArrayIterator<TSize, ui64> endIter{BlockIndices.end(), BlockBitmaps.end()};

            Sort(beginIter, endIter, [](auto lhs, auto rhs) { return lhs.first < rhs.first; });

            // compress blocks
            const TSize srcSize = BlockIndices.size();
            TSize dstIndex = 0;
            for (TSize srcIndex = 1; srcIndex < srcSize; ++srcIndex) {
                if (BlockIndices[srcIndex] == BlockIndices[dstIndex]) {
                    BlockBitmaps[dstIndex] = BlockBitmaps[dstIndex] | BlockBitmaps[srcIndex];
                } else {
                    ++dstIndex;
                    BlockIndices[dstIndex] = BlockIndices[srcIndex];
                    BlockBitmaps[dstIndex] = BlockBitmaps[srcIndex];
                }
            }
            BlockIndices.resize(dstIndex + 1);
            BlockIndices.shrink_to_fit();
            BlockBitmaps.resize(dstIndex + 1);
            BlockBitmaps.shrink_to_fit();
        }
        return TSparseArrayIndexing<TSize>(
            TSparseSubsetHybridIndex<TSize>{std::move(BlockIndices), std::move(BlockBitmaps)}
        );
    }

    template <class TSize>
    inline void TSparseSubsetHybridIndexBuilder<TSize>::AddImpl(TSize i) {
        const TSize blockIdx = i / TSparseSubsetHybridIndex<TSize>::BLOCK_SIZE;
        const ui64 bitInBlock = ui64(1) << (i % TSparseSubsetHybridIndex<TSize>::BLOCK_SIZE);
        if (BlockIndices.empty() || (blockIdx != BlockIndices.back())) {
            BlockIndices.push_back(blockIdx);
            BlockBitmaps.push_back(bitInBlock);
        } else {
            BlockBitmaps.back() |= bitInBlock;
        }
    }


    template <class TSize>
    TSparseArrayIndexingBuilderPtr<TSize> CreateSparseArrayIndexingBuilder(ESparseArrayIndexingType type) {
        switch (type) {
            case ESparseArrayIndexingType::Blocks:
                return MakeHolder<TSparseSubsetBlocksBuilder<TSize>>();
            case ESparseArrayIndexingType::HybridIndex:
                return MakeHolder<TSparseSubsetHybridIndexBuilder<TSize>>();
            case ESparseArrayIndexingType::Indices:
            default:
                return MakeHolder<TSparseSubsetIndicesBuilder<TSize>>();
        }
    }


    template <class TValue, class TContainer, class TSize>
    TSparseArrayBaseIterator<TValue, TContainer, TSize>::TSparseArrayBaseIterator(
        IDynamicIteratorPtr<TSize> indexingIteratorPtr,
        const TContainer& nonDefaultValues)
        : IndexingIteratorPtr(std::move(indexingIteratorPtr))
        , NonDefaultIndex(0)
        , NonDefaultValues(nonDefaultValues)
    {}

    template <class TValue, class TContainer, class TSize>
    inline TMaybe<std::pair<TSize, TValue>> TSparseArrayBaseIterator<TValue, TContainer, TSize>::Next() {
        const TMaybe<TSize> indexingNext = IndexingIteratorPtr->Next();
        if (!indexingNext) {
            return Nothing();
        }
        return MakeMaybe(std::pair<TSize, TValue>(*indexingNext, NonDefaultValues[NonDefaultIndex++]));
    }


    template <class TValue, class TContainer, class TSize>
    TSparseArrayBase<TValue, TContainer, TSize>::TSparseArrayBase(
        TIndexingPtr indexing,
        TContainer&& nonDefaultValues,
        TValue&& defaultValue)
        : Indexing(std::move(indexing))
        , NonDefaultValues(std::move(nonDefaultValues))
        , DefaultValue(std::move(defaultValue))
    {
        CB_ENSURE_INTERNAL(
            SafeIntegerCast<size_t>(Indexing->GetNonDefaultSize()) == NonDefaultValues.GetSize(),
            "TSparseArray: Indexing size and nondefault array size differ");
    }

    template <class TValue, class TContainer, class TSize>
    template <class TIndexingArg>
    TSparseArrayBase<TValue, TContainer, TSize>::TSparseArrayBase(
        TIndexingArg&& indexing,
        TContainer&& nonDefaultValues,
        TValue&& defaultValue)
        : TSparseArrayBase(
            MakeIntrusive<TIndexing>(std::move(indexing)),
            std::move(nonDefaultValues),
            std::move(defaultValue))
    {}


    template <class TValue, class TContainer, class TSize>
    int TSparseArrayBase<TValue, TContainer, TSize>::operator&(IBinSaver& binSaver) {
        AddWithShared(&binSaver, &Indexing);
        binSaver.AddMulti(NonDefaultValues, DefaultValue);
        return 0;
    }

    template <class TValue, class TContainer, class TSize>
    bool TSparseArrayBase<TValue, TContainer, TSize>::EqualTo(
        const TSparseArrayBase& rhs,
        bool strict) const {

        if (!Indexing->EqualTo(*(rhs.Indexing), strict)) {
            return false;
        }
        return std::tie(DefaultValue, NonDefaultValues)
            == std::tie(rhs.DefaultValue, rhs.NonDefaultValues);
    }

    template <class TValue, class TContainer, class TSize>
    template <class F>
    inline void TSparseArrayBase<TValue, TContainer, TSize>::ForEachNonDefault(F&& f) const {
        TSize nonDefaultValuesIdx = 0;
        Indexing->ForEachNonDefault(
            [f=std::move(f), this, &nonDefaultValuesIdx](TSize i) {
                f(i, NonDefaultValues[nonDefaultValuesIdx++]);
            }
        );
    }

    template <class TValue, class TContainer, class TSize>
    template <class F>
    inline void TSparseArrayBase<TValue, TContainer, TSize>::ForEach(F&& f) const {
        TSize nonDefaultValuesIdx = 0;
        Indexing->ForEach(
            [f=std::move(f), this, &nonDefaultValuesIdx](TSize i, bool isDefault) {
                if (isDefault) {
                    f(i, DefaultValue);
                } else {
                    f(i, NonDefaultValues[nonDefaultValuesIdx++]);
                }
            }
        );
    }

    template <class TValue, class TContainer, class TSize>
    TVector<typename TSparseArrayBase<TValue, TContainer, TSize>::TNonConstValue>
        TSparseArrayBase<TValue, TContainer, TSize>::ExtractValues() const {
            using TNonConstValue = typename TSparseArrayBase<TValue, TContainer, TSize>::TNonConstValue;

            TVector<TNonConstValue> result;
            result.reserve(GetSize());
            ForEach(
                [&] (ui32 idx, TNonConstValue value) {
                    Y_UNUSED(idx);
                    result.push_back(std::move(value));
                }
            );
            return result;
        }

    template <class TValue, class TContainer, class TSize>
    TSparseArrayBaseIteratorPtr<TValue, TContainer, TSize>
        TSparseArrayBase<TValue, TContainer, TSize>::GetIterator() {
            return MakeHolder<TSparseArrayBaseIterator<TValue, TContainer, TSize>>(
               Indexing->GetIterator(),
               NonDefaultValues);
    }


    template <class TValue, class TSize>
    TMaybeOwningArrayHolder<TValue> CreateSubsetContainer(
        TVector<TValue>&& subsetNonDefaultValues,
        const TSparseArrayBase<TValue, TMaybeOwningArrayHolder<TValue>, TSize>& parent) {

        Y_UNUSED(parent);

        return TMaybeOwningArrayHolder<TValue>::CreateOwning(std::move(subsetNonDefaultValues));
    }

    // in addition to dstValues passed to CreateSubsetContainer
    template <class TValue, class TSize>
    size_t EstimateContainerCreationAdditionalCpuRamUsage(
        const TSparseArrayBase<TValue, TMaybeOwningArrayHolder<TValue>, TSize>& parent) {

        Y_UNUSED(parent);

        // dstValues is moved to result
        return 0;
    }

    template <class TValue, class TSize>
    TCompressedArray CreateSubsetContainer(
        TVector<TValue>&& subsetNonDefaultValues,
        const TSparseArrayBase<TValue, TCompressedArray, TSize>& parent) {

        const ui32 bitsPerKey = parent.GetNonDefaultValues().GetBitsPerKey();

        return TCompressedArray(
            subsetNonDefaultValues.size(),
            bitsPerKey,
            CompressVector<ui64>(subsetNonDefaultValues, bitsPerKey)
        );
    }

    // in addition to dstValues passed to CreateSubsetContainer
    template <class TValue, class TSize>
    size_t EstimateContainerCreationAdditionalCpuRamUsage(
        const TSparseArrayBase<TValue, TCompressedArray, TSize>& parent) {

        const TCompressedArray& nonDefaultValues = parent.GetNonDefaultValues();
        const TIndexHelper<ui64> indexHelper(nonDefaultValues.GetBitsPerKey());
        return indexHelper.CompressedSize(nonDefaultValues.GetSize()) * sizeof(ui64);
    }


    template <class TValue, class TContainer, class TSize>
    size_t TSparseArrayBase<TValue, TContainer, TSize>::EstimateGetSubsetCpuRamUsage(
        const TArraySubsetInvertedIndexing<TSize>& subsetInvertedIndexing,
        ESparseArrayIndexingType sparseArrayIndexingType) const {

        if (HoldsAlternative<TFullSubset<TSize>>(subsetInvertedIndexing)) {
            return 0;
        }

        ui64 ramUsedForDstIndexing;
        switch (sparseArrayIndexingType) {
            case ESparseArrayIndexingType::Indices:
                ramUsedForDstIndexing = sizeof(TSize) * GetNonDefaultSize();
                break;
            case ESparseArrayIndexingType::Blocks:
                ramUsedForDstIndexing = 2 * sizeof(TSize) * GetNonDefaultSize();
                break;
            case ESparseArrayIndexingType::HybridIndex:
                ramUsedForDstIndexing = (sizeof(TSize) + sizeof(ui64)) * GetNonDefaultSize();
                break;
            default:
                ;
        }

        const ui64 ramUsedForDstValues = sizeof(TValue) * GetNonDefaultSize();

        if (sparseArrayIndexingType == ESparseArrayIndexingType::Undefined) {
            sparseArrayIndexingType = Indexing->GetType();
        }

        ui64 ramUsedDuringBuilding = ramUsedForDstIndexing + ramUsedForDstValues;
        if (sparseArrayIndexingType != ESparseArrayIndexingType::Indices) {
            // for dstVectorIndexing
            ramUsedDuringBuilding += sizeof(TSize) * GetNonDefaultSize();
        }

        const ui64 ramUsedDuringResultCreation
            = ramUsedForDstIndexing
                + ramUsedForDstValues
                + EstimateContainerCreationAdditionalCpuRamUsage(*this);

        return Max(ramUsedDuringBuilding, ramUsedDuringResultCreation);
    }


    template <class TValue, class TContainer, class TSize>
    TSparseArrayBase<TValue, TContainer, TSize> TSparseArrayBase<TValue, TContainer, TSize>::GetSubset(
        const TArraySubsetInvertedIndexing<TSize>& subsetInvertedIndexing,
        ESparseArrayIndexingType sparseArrayIndexingType) const {

        if (HoldsAlternative<TFullSubset<TSize>>(subsetInvertedIndexing)) {
            return *this;
        }

        const TInvertedIndexedSubset<TSize>& invertedIndexedSubset
            = Get<TInvertedIndexedSubset<TSize>>(subsetInvertedIndexing);

        TConstArrayRef<TSize> invertedIndicesArray = invertedIndexedSubset.GetMapping();

        TVector<TSize> dstVectorIndexing;
        TVector<TValue> dstValues;

        TSize nonDefaultValuesIdx = 0;
        Indexing->ForEachNonDefault(
            [&](TSize srcIdx) {
                auto dstIdx = invertedIndicesArray[srcIdx];
                if (dstIdx != TInvertedIndexedSubset<TSize>::NOT_PRESENT) {
                    dstVectorIndexing.push_back(dstIdx);
                    dstValues.push_back(NonDefaultValues[nonDefaultValuesIdx]);
                }
                ++nonDefaultValuesIdx;
            }
        );

        TDoubleArrayIterator<TSize, TValue> dstBegin{dstVectorIndexing.begin(), dstValues.begin()};
        TDoubleArrayIterator<TSize, TValue> dstEnd{dstVectorIndexing.end(), dstValues.end()};

        Sort(dstBegin, dstEnd, [](auto lhs, auto rhs) { return lhs.first < rhs.first; });

        if (sparseArrayIndexingType == ESparseArrayIndexingType::Undefined) {
            sparseArrayIndexingType = Indexing->GetType();
        }

        TIndexingPtr dstIndexing;
        if (sparseArrayIndexingType == ESparseArrayIndexingType::Indices) {
            dstIndexing = MakeIntrusive<TIndexing>(TSparseSubsetIndices<TSize>(std::move(dstVectorIndexing)));
        } else {
            auto builder = CreateSparseArrayIndexingBuilder<TSize>(sparseArrayIndexingType);
            for (auto i : dstVectorIndexing) {
                builder->AddOrdered(i);
            }
            TVector<TSize>().swap(dstVectorIndexing); // force early CPU RAM release
            dstIndexing = MakeIntrusive<TIndexing>(builder->Build());
        }

        TValue defaultValueCopy = DefaultValue;

        return TSparseArrayBase<TValue, TContainer, TSize>(
            std::move(dstIndexing),
            CreateSubsetContainer(std::move(dstValues), *this),
            std::move(defaultValueCopy)
        );
    }

}
