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

#include <algorithm>


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
    TSparseSubsetBlocksIterator<TSize>::TSparseSubsetBlocksIterator(
        const TSize* blockStartsCurrent,
        const TSize* blockStartsEnd,
        const TSize* blockLengthsCurrent,
        TSize inBlockIdx)
        : BlockStartsCurrent(blockStartsCurrent)
        , BlockStartsEnd(blockStartsEnd)
        , BlockLengthsCurrent(blockLengthsCurrent)
        , InBlockIdx(inBlockIdx)
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
    TSparseSubsetHybridIndexIterator<TSize>::TSparseSubsetHybridIndexIterator(
        const TSize* blockIndicesCurrent,
        const TSize* blockIndicesEnd,
        const ui64* blockBitmapsCurrent,
        ui32 inBlockIdx)
        : BlockIndicesCurrent(blockIndicesCurrent)
        , BlockIndicesEnd(blockIndicesEnd)
        , BlockBitmapsCurrent(blockBitmapsCurrent)
        , InBlockIdx(inBlockIdx)
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
    void GetIteratorAndNonDefaultBeginImpl(
        const TSparseSubsetIndices<TSize>& sparseSubsetIndices,
        TSize begin,
        IDynamicIteratorPtr<TSize>* iterator,
        TSize* nonDefaultBegin) {

        auto lowerBound = LowerBound(sparseSubsetIndices.begin(), sparseSubsetIndices.end(), begin);
        *iterator = MakeHolder<TStaticIteratorRangeAsDynamic<const TSize*>>(
            lowerBound,
            sparseSubsetIndices.end());
        *nonDefaultBegin = lowerBound - sparseSubsetIndices.begin();
    }

    template <class TSize>
    void GetIteratorAndNonDefaultBeginImpl(
        const TSparseSubsetBlocks<TSize>& sparseSubsetBlocks,
        TSize begin,
        IDynamicIteratorPtr<TSize>* iterator,
        TSize* nonDefaultBegin) {

        TDoubleArrayIterator<const TSize, const TSize> blocksBegin{
            sparseSubsetBlocks.BlockStarts.begin(),
            sparseSubsetBlocks.BlockLengths.begin()};

        TDoubleArrayIterator<const TSize, const TSize> blocksEnd{
            sparseSubsetBlocks.BlockStarts.end(),
            sparseSubsetBlocks.BlockLengths.end()};

        TDoubleArrayIterator<const TSize, const TSize> blockIterator
            = LowerBound(
                blocksBegin,
                blocksEnd,
                begin,
                [&] (const std::pair<const TSize&, const TSize&>& block, const TSize begin) {
                    return (block.first + block.second) <= begin;
                });

        const TSize blockOffset = blockIterator - blocksBegin;
        const TSize* blockStartsCurrent = sparseSubsetBlocks.BlockStarts.begin() + blockOffset;
        const TSize* blockLengthsCurrent = sparseSubsetBlocks.BlockLengths.begin() + blockOffset;
        TSize inBlockIdx = 0;
        if (blockStartsCurrent != sparseSubsetBlocks.BlockStarts.end()) {
            inBlockIdx = (begin < *blockStartsCurrent) ? 0 : (begin - *blockStartsCurrent);
            *nonDefaultBegin
                = std::accumulate(sparseSubsetBlocks.BlockLengths.begin(), blockLengthsCurrent, inBlockIdx);
        } else {
            *nonDefaultBegin = 0; // not really used, set for consistency
        }

        *iterator = MakeHolder<TSparseSubsetBlocksIterator<TSize>>(
            blockStartsCurrent,
            sparseSubsetBlocks.BlockStarts.end(),
            blockLengthsCurrent,
            inBlockIdx);

    }

    template <class TSize>
    void GetIteratorAndNonDefaultBeginImpl(
        const TSparseSubsetHybridIndex<TSize>& sparseSubsetHybridIndex,
        TSize begin,
        IDynamicIteratorPtr<TSize>* iterator,
        TSize* nonDefaultBegin) {

        const TSize beginBlockIdx = begin / TSparseSubsetHybridIndex<TSize>::BLOCK_SIZE;

        const auto& blockIndices = sparseSubsetHybridIndex.BlockIndices;
        const auto& blockBitmaps = sparseSubsetHybridIndex.BlockBitmaps;

        const TSize* blockIndicesCurrent
            = LowerBound(blockIndices.begin(), blockIndices.end(), beginBlockIdx);

        const TSize blockOffset = blockIndicesCurrent - blockIndices.begin();
        const ui64* blockBitmapsCurrent = blockBitmaps.begin() + blockOffset;

        TSize inBlockIdx;
        TSize nonDefaultInBlockBeforeBegin;
        if ((blockIndicesCurrent != blockIndices.end()) && (beginBlockIdx == *blockIndicesCurrent)) {
            inBlockIdx = begin % TSparseSubsetHybridIndex<TSize>::BLOCK_SIZE;
            if (*blockBitmapsCurrent >> inBlockIdx) {
                nonDefaultInBlockBeforeBegin
                    = (TSize)PopCount((*blockBitmapsCurrent) & ((1ULL << inBlockIdx) - 1));
            } else {
                ++blockIndicesCurrent;
                ++blockBitmapsCurrent;
                inBlockIdx = 0;
                nonDefaultInBlockBeforeBegin = 0;
            }
        } else {
            inBlockIdx = 0;
            nonDefaultInBlockBeforeBegin = 0;
        }

        *nonDefaultBegin
            = std::accumulate(
                blockBitmaps.begin(),
                blockBitmapsCurrent,
                nonDefaultInBlockBeforeBegin,
                [] (TSize sum, ui64 element) { return sum + (TSize)PopCount(element); });

        *iterator = MakeHolder<TSparseSubsetHybridIndexIterator<TSize>>(
            blockIndicesCurrent,
            blockIndices.end(),
            blockBitmapsCurrent,
            inBlockIdx);
    }

    template <class TSize>
    void TSparseArrayIndexing<TSize>::GetIteratorAndNonDefaultBegin(
        TSize begin,
        IDynamicIteratorPtr<TSize>* iterator,
        TSize* nonDefaultBegin) const {

        Visit(
            [&](const auto& impl) {
                GetIteratorAndNonDefaultBeginImpl(impl, begin, iterator, nonDefaultBegin);
            },
            Impl);
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
    TSparseArrayIndexing<TSize> TSparseSubsetIndicesBuilder<TSize>::Build(TMaybe<TSize> size) {
        if (NonOrdered) {
            Sort(Indices);
        }
        return TSparseArrayIndexing<TSize>(TSparseSubsetIndices<TSize>(std::move(Indices)), size);
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
    TSparseArrayIndexing<TSize> TSparseSubsetBlocksBuilder<TSize>::Build(TMaybe<TSize> size) {
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
            TSparseSubsetBlocks<TSize>{std::move(BlockStarts), std::move(BlockLengths)},
            size
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
    TSparseArrayIndexing<TSize> TSparseSubsetHybridIndexBuilder<TSize>::Build(TMaybe<TSize> size) {
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
            TSparseSubsetHybridIndex<TSize>{std::move(BlockIndices), std::move(BlockBitmaps)},
            size
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
        const TContainer& nonDefaultValues,
        TSize nonDefaultOffset)
        : IndexingIteratorPtr(std::move(indexingIteratorPtr))
        , NonDefaultIndex(nonDefaultOffset)
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
    TSparseArrayBaseBlockIterator<TValue, TContainer, TSize>::TSparseArrayBaseBlockIterator(
        TSize size,
        IDynamicIteratorPtr<TSize> indexingIteratorPtr,
        TValue defaultValue,
        const TContainer& nonDefaultValues,
        TSize offset,
        TSize nonDefaultOffset)
        : Index(offset)
        , Size(size)
        , IndexingIteratorPtr(std::move(indexingIteratorPtr))
        , NextNonDefault(IndexingIteratorPtr->Next())
        , DefaultValue(defaultValue)
        , NonDefaultIndex(nonDefaultOffset)
        , NonDefaultValues(nonDefaultValues)
    {}

    template <class TValue, class TContainer, class TSize>
    inline
    TConstArrayRef<TValue>
        TSparseArrayBaseBlockIterator<TValue, TContainer, TSize>::Next(size_t maxBlockSize) {
            const TSize blockSize = Min(Size - Index, (TSize)Min(size_t(Max<TSize>()), maxBlockSize));
            Buffer.yresize(blockSize);
            Fill(Buffer.begin(), Buffer.end(), DefaultValue);
            const TSize blockEnd = Index + blockSize;

            while (NextNonDefault && (*NextNonDefault < blockEnd)) {
                Buffer[*NextNonDefault - Index] = NonDefaultValues[NonDefaultIndex++];
                NextNonDefault = IndexingIteratorPtr->Next();
            }
            Index = blockEnd;
            return Buffer;
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
    typename TSparseArrayBase<TValue, TContainer, TSize>::TIterator
        TSparseArrayBase<TValue, TContainer, TSize>::GetIterator(TSize offset) const {
            IDynamicIteratorPtr<TSize> indexingIterator;
            TSize nonDefaultOffset;
            Indexing->GetIteratorAndNonDefaultBegin(offset, &indexingIterator, &nonDefaultOffset);

            return TSparseArrayBaseIterator<TNonConstValue, TContainer, TSize>(
               std::move(indexingIterator),
               NonDefaultValues,
               nonDefaultOffset
            );
    }


    template <class TValue, class TContainer, class TSize>
    typename TSparseArrayBase<TValue, TContainer, TSize>::TBlockIterator
        TSparseArrayBase<TValue, TContainer, TSize>::GetBlockIterator(TSize offset) const {
            IDynamicIteratorPtr<TSize> indexingIterator;
            TSize nonDefaultOffset;
            Indexing->GetIteratorAndNonDefaultBegin(offset, &indexingIterator, &nonDefaultOffset);

            return TSparseArrayBaseBlockIterator<TNonConstValue, TContainer, TSize>(
                GetSize(),
                std::move(indexingIterator),
                GetDefaultValue(),
                NonDefaultValues,
                offset,
                nonDefaultOffset
            );
    }


    template <class TDstValue, class TSrcValue, class TSize>
    TMaybeOwningArrayHolder<TDstValue> CreateSubsetContainer(
        TVector<TSrcValue>&& subsetNonDefaultValues,
        const TSparseArrayBase<TDstValue, TMaybeOwningArrayHolder<TDstValue>, TSize>& parent) {

        Y_UNUSED(parent);

        return TMaybeOwningArrayHolder<TDstValue>::CreateOwning(std::move(subsetNonDefaultValues));
    }

    // in addition to dstValues passed to CreateSubsetContainer
    template <class TValue, class TSize>
    size_t EstimateContainerCreationAdditionalCpuRamUsage(
        const TSparseArrayBase<TValue, TMaybeOwningArrayHolder<TValue>, TSize>& parent) {

        Y_UNUSED(parent);

        // dstValues is moved to result
        return 0;
    }

    template <class TDstValue, class TSrcValue, class TSize>
    TCompressedArray CreateSubsetContainer(
        TVector<TSrcValue>&& subsetNonDefaultValues,
        const TSparseArrayBase<TDstValue, TCompressedArray, TSize>& parent) {

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

        if (sparseArrayIndexingType == ESparseArrayIndexingType::Undefined) {
            sparseArrayIndexingType = Indexing->GetType();
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
                Y_UNREACHABLE();
        }

        const ui64 ramUsedForDstValues = sizeof(TValue) * GetNonDefaultSize();

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

        using TNonConstValue = typename std::remove_const<TValue>::type;

        TVector<TSize> dstVectorIndexing;
        TVector<TNonConstValue> dstValues;

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

        if (sparseArrayIndexingType == ESparseArrayIndexingType::Undefined) {
            sparseArrayIndexingType = Indexing->GetType();
        }

        std::function<TContainer(TVector<TNonConstValue>&&)> createNonDefaultValuesContainer
            = [&] (TVector<TNonConstValue>&& dstValues) {
                return CreateSubsetContainer(std::move(dstValues), *this);
            };

        TValue defaultValueCopy = DefaultValue;

        return MakeSparseArrayBase<TValue, TContainer, TSize>(
            invertedIndexedSubset.GetSize(),
            std::move(dstVectorIndexing),
            std::move(dstValues),
            std::move(createNonDefaultValuesContainer),
            sparseArrayIndexingType,
            /*ordered*/ false,
            std::move(defaultValueCopy)
        );
    }

    template <class TValue, class TContainer, class TSize, class TSrcValue>
    TSparseArrayBase<TValue, TContainer, TSize> MakeSparseArrayBase(
        TSize size,
        TVector<TSize>&& indexing,
        TVector<TSrcValue>&& nonDefaultValues,
        std::function<TContainer(TVector<TSrcValue>&&)>&& createNonDefaultValuesContainer,
        ESparseArrayIndexingType sparseArrayIndexingType,
        bool ordered,
        TValue&& defaultValue) {

        if (sparseArrayIndexingType == ESparseArrayIndexingType::Undefined) {
            sparseArrayIndexingType = ESparseArrayIndexingType::Indices;
        }

        if (!ordered) {
            TDoubleArrayIterator<TSize, TSrcValue> dstBegin{indexing.begin(), nonDefaultValues.begin()};
            TDoubleArrayIterator<TSize, TSrcValue> dstEnd{indexing.end(), nonDefaultValues.end()};

            Sort(dstBegin, dstEnd, [](auto lhs, auto rhs) { return lhs.first < rhs.first; });
        }

        using TIndexing = TSparseArrayIndexing<TSize>;

        TIntrusivePtr<TIndexing> dstIndexing;
        if (sparseArrayIndexingType == ESparseArrayIndexingType::Indices) {
            dstIndexing = MakeIntrusive<TIndexing>(TSparseSubsetIndices<TSize>(std::move(indexing)), size);
        } else {
            auto builder = CreateSparseArrayIndexingBuilder<TSize>(sparseArrayIndexingType);
            for (auto i : indexing) {
                builder->AddOrdered(i);
            }
            TVector<TSize>().swap(indexing); // force early CPU RAM release
            dstIndexing = MakeIntrusive<TIndexing>(builder->Build(size));
        }

        return TSparseArrayBase<TValue, TContainer, TSize>(
            std::move(dstIndexing),
            createNonDefaultValuesContainer(std::move(nonDefaultValues)),
            std::move(defaultValue)
        );
    }

    template <class TSize>
    TSparseArrayIndexingPtr<TSize> MakeSparseArrayIndexing(
        TSize size,
        TMaybeOwningConstArrayHolder<TSize> indices) {

        return MakeIntrusive<TSparseArrayIndexing<TSize>>(
            TSparseSubsetIndices<TSize>(std::move(indices)),
            size
        );
    }

    template <class TSize>
    TSparseArrayIndexingPtr<TSize> MakeSparseBlockIndexing(
        TSize size,
        TMaybeOwningConstArrayHolder<TSize> blockStarts, // already ordered
        TMaybeOwningConstArrayHolder<TSize> blockLengths) {

        return MakeIntrusive<TSparseArrayIndexing<TSize>>(
            TSparseSubsetBlocks<TSize>(std::move(blockStarts), std::move(blockLengths)),
            size
        );
    }

    template <class TValue, class TSize>
    TConstSparseArray<TValue, TSize> MakeConstSparseArray(
        TSparseArrayIndexingPtr<TSize> indexing,
        TMaybeOwningConstArrayHolder<TValue> nonDefaultValues,
        TValue defaultValue) {

        return TConstSparseArray<TValue, TSize>(
            std::move(indexing),
            std::move(nonDefaultValues),
            std::move(defaultValue)
        );
    }


    template <class TValue, class TSize>
    TConstSparseArray<TValue, TSize> MakeConstSparseArrayWithArrayIndex(
        TSize size,
        TMaybeOwningConstArrayHolder<TSize> indexing, // alrady ordered
        TMaybeOwningConstArrayHolder<TValue> nonDefaultValues,
        bool ordered,
        TValue defaultValue) {

        if (ordered) {
            return TConstSparseArray<TValue, TSize>(
                MakeIntrusive<TSparseArrayIndexing<TSize>>(
                    TSparseSubsetIndices<TSize>(std::move(indexing)),
                    size
                ),
                std::move(nonDefaultValues),
                std::move(defaultValue)
            );
        } else {
            using TNonConstValue = std::remove_const_t<TValue>;

            TVector<TSize> indexingCopy(indexing.begin(), indexing.end());
            TVector<TNonConstValue> nonDefaultValuesCopy(nonDefaultValues.begin(), nonDefaultValues.end());

            std::function<TMaybeOwningConstArrayHolder<TValue>(TVector<TNonConstValue>&&)>
                createNonDefaultValues
                    = [&] (TVector<TNonConstValue>&& values) {
                        return TMaybeOwningConstArrayHolder<TValue>::CreateOwning(std::move(values));
                    };

            return MakeSparseArrayBase<const TValue, TMaybeOwningConstArrayHolder<TValue>>(
                size,
                std::move(indexingCopy),
                std::move(nonDefaultValuesCopy),
                std::move(createNonDefaultValues),
                ESparseArrayIndexingType::Indices,
                /*ordered*/ false,
                std::move(defaultValue)
            );
        }
    }

}
