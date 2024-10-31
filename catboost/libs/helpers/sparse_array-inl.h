#pragma once

#include "array_subset.h"
#include "double_array_iterator.h"
#include "exception.h"
#include "serialization.h"


#include <util/generic/bitops.h>
#include <util/generic/cast.h>
#include <util/generic/overloaded.h>
#include <util/generic/xrange.h>

#include <util/system/compiler.h>
#include <util/system/yassert.h>

#include <algorithm>
#include <bit>


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
    TSparseSubsetIndicesBlockIterator<TSize>::TSparseSubsetIndicesBlockIterator(
        const TSize* current,
        const TSize* end)
        : Current(current)
        , End(end)
    {}

    template <class TSize>
    TConstArrayRef<TSize> TSparseSubsetIndicesBlockIterator<TSize>::Next(size_t maxBlockSize) {
        const size_t blockSize = Min(maxBlockSize, size_t(End - Current));
        TConstArrayRef<TSize> result(Current, Current + blockSize);
        Current += blockSize;
        return result;
    }

    template <class TSize>
    TConstArrayRef<TSize> TSparseSubsetIndicesBlockIterator<TSize>::NextUpToBound(TSize upperBound) {
        const TSize* blockEnd = Current;
        while ((blockEnd < End) && (*blockEnd < upperBound)) {
            ++blockEnd;
        }
        TConstArrayRef<TSize> result(Current, blockEnd);
        Current = blockEnd;
        return result;
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
        : Data{
            (*sparseSubsetBlocks.BlockStarts).begin(),
            (*sparseSubsetBlocks.BlockStarts).end(),
            (*sparseSubsetBlocks.BlockLengths).begin(),
            TSize(0)
        }
    {}

    template <class TSize>
    TSparseSubsetBlocksIterator<TSize>::TSparseSubsetBlocksIterator(
        TSparseSubsetBlocksIteratorData<TSize>&& data)
        : Data(std::move(data))
    {}

    template <class TSize>
    inline bool TSparseSubsetBlocksIterator<TSize>::Next(TSize* value) {
        if (Data.BlockStartsCurrent == Data.BlockStartsEnd) {
            return false;
        }
        while (Data.InBlockIdx == *Data.BlockLengthsCurrent) {
            if (++Data.BlockStartsCurrent == Data.BlockStartsEnd) {
                return false;
            }
            ++Data.BlockLengthsCurrent;
            Data.InBlockIdx = 0;
        }
        *value = (*Data.BlockStartsCurrent) + Data.InBlockIdx++;
        return true;
    }

    template <class TSize>
    TSparseSubsetBlocksBlockIterator<TSize>::TSparseSubsetBlocksBlockIterator(
        TSparseSubsetBlocksIteratorData<TSize>&& data)
        : Data(std::move(data))
    {}

    template <class TSize>
    TConstArrayRef<TSize> TSparseSubsetBlocksBlockIterator<TSize>::Next(size_t maxBlockSize) {
        if (Data.BlockStartsCurrent == Data.BlockStartsEnd) {
            return TConstArrayRef<TSize>();
        }
        while (Data.InBlockIdx == *Data.BlockLengthsCurrent) {
            if (++Data.BlockStartsCurrent == Data.BlockStartsEnd) {
                return TConstArrayRef<TSize>();
            }
            ++Data.BlockLengthsCurrent;
            Data.InBlockIdx = 0;
        }
        Buffer.yresize(Min(maxBlockSize, size_t(*Data.BlockLengthsCurrent - Data.InBlockIdx)));
        Iota(Buffer.begin(), Buffer.end(), (*Data.BlockStartsCurrent) + Data.InBlockIdx);

        if (Data.InBlockIdx + Buffer.size() == *Data.BlockLengthsCurrent) {
            ++Data.BlockStartsCurrent;
            ++Data.BlockLengthsCurrent;
            Data.InBlockIdx = 0;
        } else {
            Data.InBlockIdx += Buffer.size();
        }

        return Buffer;
    }

    template <class TSize>
    TConstArrayRef<TSize> TSparseSubsetBlocksBlockIterator<TSize>::NextUpToBound(TSize upperBound) {
        if (Data.BlockStartsCurrent == Data.BlockStartsEnd) {
            return TConstArrayRef<TSize>();
        }
        Buffer.clear();

        while (true) {
            const auto lowerBoundInBlock = (*Data.BlockStartsCurrent) + Data.InBlockIdx;
            if (lowerBoundInBlock >= upperBound) {
                return Buffer;
            }
            const auto blockEnd = (*Data.BlockStartsCurrent) + (*Data.BlockLengthsCurrent);
            const auto upperBoundInBlock = Min(upperBound, blockEnd);

            const auto dstBlockSize = upperBoundInBlock - lowerBoundInBlock;
            Buffer.yresize(Buffer.size() + dstBlockSize);
            Iota(Buffer.end() - dstBlockSize, Buffer.end(), lowerBoundInBlock);

            if (upperBoundInBlock < blockEnd) {
                Data.InBlockIdx += dstBlockSize;
                return Buffer;
            }
            ++Data.BlockStartsCurrent;
            if (Data.BlockStartsCurrent == Data.BlockStartsEnd) {
                return Buffer;
            }

            ++Data.BlockLengthsCurrent;
            Data.InBlockIdx = 0;
        }
        Y_UNREACHABLE();
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
            result += std::popcount(blockBitmap);
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
        : Data{
            sparseSubsetHybridIndex.BlockIndices.begin(),
            sparseSubsetHybridIndex.BlockIndices.end(),
            sparseSubsetHybridIndex.BlockBitmaps.begin(),
            ui32(0)
        }
    {}

    template <class TSize>
    TSparseSubsetHybridIndexIterator<TSize>::TSparseSubsetHybridIndexIterator(
        TSparseSubsetHybridIndexIteratorData<TSize>&& data)
        : Data(std::move(data))
    {}

    template <class TSize>
    inline bool TSparseSubsetHybridIndexIterator<TSize>::Next(TSize* value) {
        if (Data.BlockIndicesCurrent == Data.BlockIndicesEnd) {
            return false;
        }
        while (! (((*Data.BlockBitmapsCurrent) >> Data.InBlockIdx) & 1)) {
            ++Data.InBlockIdx;
        }

        *value
            = TSparseSubsetHybridIndex<TSize>::BLOCK_SIZE * (*Data.BlockIndicesCurrent) + Data.InBlockIdx;

        if (! ((*Data.BlockBitmapsCurrent) >> (Data.InBlockIdx + 1))) {
            ++Data.BlockIndicesCurrent;
            ++Data.BlockBitmapsCurrent;
            Data.InBlockIdx = 0;
        } else {
            ++Data.InBlockIdx;
        }

        return true;
    }


    template <class TSize>
    TSparseSubsetHybridIndexBlockIterator<TSize>::TSparseSubsetHybridIndexBlockIterator(
        TSparseSubsetHybridIndexIteratorData<TSize>&& data)
        : Data(std::move(data))
    {}

    template <class TSize>
    TConstArrayRef<TSize> TSparseSubsetHybridIndexBlockIterator<TSize>::Next(size_t maxBlockSize) {
        if (Data.BlockIndicesCurrent == Data.BlockIndicesEnd) {
            return TConstArrayRef<TSize>();
        }

        size_t inBlockSize = (size_t)std::popcount((*Data.BlockBitmapsCurrent) >> Data.InBlockIdx);
        if (inBlockSize == 0) {
            ++Data.BlockIndicesCurrent;
            if (Data.BlockIndicesCurrent == Data.BlockIndicesEnd) {
                return TConstArrayRef<TSize>();
            }
            ++Data.BlockBitmapsCurrent;
            Data.InBlockIdx = 0;
            inBlockSize = (size_t)std::popcount(*Data.BlockBitmapsCurrent);
        }

        const auto dstBlockSize = Min(maxBlockSize, inBlockSize);

        Buffer.yresize(dstBlockSize);

        TSize srcBlockStart = (*Data.BlockIndicesCurrent) * TSparseSubsetHybridIndex<TSize>::BLOCK_SIZE;
        size_t blockIdx = 0;
        while (true) {
            if (((*Data.BlockBitmapsCurrent) >> (Data.InBlockIdx)) & 1) {
                Buffer[blockIdx] = srcBlockStart + Data.InBlockIdx;
                if (++blockIdx == dstBlockSize) {
                    ++Data.InBlockIdx;
                    break;
                }
            }
            ++Data.InBlockIdx;
        }
        if (dstBlockSize == inBlockSize) {
            ++Data.BlockIndicesCurrent;
            ++Data.BlockBitmapsCurrent;
            Data.InBlockIdx = 0;
        }

        return Buffer;
    }

    template <class TSize>
    TConstArrayRef<TSize> TSparseSubsetHybridIndexBlockIterator<TSize>::NextUpToBound(TSize upperBound) {
        if (Data.BlockIndicesCurrent == Data.BlockIndicesEnd) {
            return TConstArrayRef<TSize>();
        }
        Buffer.clear();

        while (true) {
            const auto srcBlockStart
                = (*Data.BlockIndicesCurrent) * TSparseSubsetHybridIndex<TSize>::BLOCK_SIZE;

            if (upperBound <= srcBlockStart) {
                return Buffer;
            }
            const auto srcBlockEnd = srcBlockStart + TSparseSubsetHybridIndex<TSize>::BLOCK_SIZE;
            const auto upperBoundInBlock = Min(srcBlockEnd, upperBound) - srcBlockStart;
            ui32 dstBlockSize = 0;
            if (upperBoundInBlock == TSparseSubsetHybridIndex<TSize>::BLOCK_SIZE) {
                dstBlockSize = std::popcount((*Data.BlockBitmapsCurrent) >> Data.InBlockIdx);
            } else {
                dstBlockSize = std::popcount(
                    (*Data.BlockBitmapsCurrent & ((1ULL << upperBoundInBlock) - 1)) >> Data.InBlockIdx
                );
            }

            if (dstBlockSize) {
                auto dstIdx = Buffer.size();
                Buffer.yresize(Buffer.size() + dstBlockSize);

                while (true) {
                    if (((*Data.BlockBitmapsCurrent) >> Data.InBlockIdx) & 1) {
                        Buffer[dstIdx] = srcBlockStart + Data.InBlockIdx;
                        if (++dstIdx == Buffer.size()) {
                            break;
                        }
                    }
                    ++Data.InBlockIdx;
                }
                ++Data.InBlockIdx;
            }

            if (upperBound <= srcBlockEnd) {
                return Buffer;
            }

            ++Data.BlockIndicesCurrent;
            if (Data.BlockIndicesCurrent == Data.BlockIndicesEnd) {
                return Buffer;
            }
            ++Data.BlockBitmapsCurrent;
            Data.InBlockIdx = 0;
        }
        Y_UNREACHABLE();
    }


    template <class TSize>
    TSparseArrayIndexing<TSize>::TSparseArrayIndexing(TImpl&& impl, TMaybe<TSize> size, bool skipCheck)
        : Impl(std::move(impl))
        , NonDefaultSize(0) // properly inited later
        , Size(0) // properly inited later
    {
        std::visit(
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
            return AreSequencesEqual<TSize>(GetIterator(), rhs.GetIterator());
        }
    }

    template <class TSize>
    ESparseArrayIndexingType TSparseArrayIndexing<TSize>::GetType() const {
        return std::visit(TOverloaded{
            [](const TSparseSubsetIndices<TSize>&) { return ESparseArrayIndexingType::Indices; },
            [](const TSparseSubsetBlocks<TSize>&) { return ESparseArrayIndexingType::Blocks; },
            [](const TSparseSubsetHybridIndex<TSize>&) { return ESparseArrayIndexingType::HybridIndex; }
        }, Impl);
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
        return std::visit(TOverloaded{
            [](const TSparseSubsetIndices<TSize>& indices) -> IDynamicIteratorPtr<TSize> {
                return MakeHolder<TStaticIteratorRangeAsDynamic<const TSize*>>(indices);
            },
            [](const TSparseSubsetBlocks<TSize>& blocks) -> IDynamicIteratorPtr<TSize> {
                return MakeHolder<TSparseSubsetBlocksIterator<TSize>>(blocks);
            },
            [](const TSparseSubsetHybridIndex<TSize>& hybrid) -> IDynamicIteratorPtr<TSize> {
                return MakeHolder<TSparseSubsetHybridIndexIterator<TSize>>(hybrid);
            }
        }, Impl);
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
    void GetBlockIteratorAndNonDefaultBeginImpl(
        const TSparseSubsetIndices<TSize>& sparseSubsetIndices,
        TSize begin,
        ISparseArrayIndexingBlockIteratorPtr<TSize>* iterator,
        TSize* nonDefaultBegin) {

        auto lowerBound = LowerBound(sparseSubsetIndices.begin(), sparseSubsetIndices.end(), begin);
        *iterator = MakeHolder<TSparseSubsetIndicesBlockIterator<TSize>>(
            lowerBound,
            sparseSubsetIndices.end());
        *nonDefaultBegin = lowerBound - sparseSubsetIndices.begin();
    }

    template <class TSize>
    void GetSparseSubsetBlocksIteratorDataAndNonDefaultBegin(
        const TSparseSubsetBlocks<TSize>& sparseSubsetBlocks,
        TSize begin,
        TSparseSubsetBlocksIteratorData<TSize>* iteratorData,
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

        iteratorData->BlockStartsCurrent = blockStartsCurrent;
        iteratorData->BlockStartsEnd = sparseSubsetBlocks.BlockStarts.end();
        iteratorData->BlockLengthsCurrent = blockLengthsCurrent;
        iteratorData->InBlockIdx = 0;

        if (blockStartsCurrent != sparseSubsetBlocks.BlockStarts.end()) {
            iteratorData->InBlockIdx = (begin < *blockStartsCurrent) ? 0 : (begin - *blockStartsCurrent);
            *nonDefaultBegin
                = std::accumulate(
                    sparseSubsetBlocks.BlockLengths.begin(),
                    blockLengthsCurrent,
                    iteratorData->InBlockIdx);
        } else {
            iteratorData->InBlockIdx = 0;
            *nonDefaultBegin = 0; // not really used, set for consistency
        }
    }

    template <class TSize>
    void GetIteratorAndNonDefaultBeginImpl(
        const TSparseSubsetBlocks<TSize>& sparseSubsetBlocks,
        TSize begin,
        IDynamicIteratorPtr<TSize>* iterator,
        TSize* nonDefaultBegin) {

        TSparseSubsetBlocksIteratorData<TSize> iteratorData;

        GetSparseSubsetBlocksIteratorDataAndNonDefaultBegin(
            sparseSubsetBlocks,
            begin,
            &iteratorData,
            nonDefaultBegin);

        *iterator = MakeHolder<TSparseSubsetBlocksIterator<TSize>>(std::move(iteratorData));
    }

    template <class TSize>
    void GetBlockIteratorAndNonDefaultBeginImpl(
        const TSparseSubsetBlocks<TSize>& sparseSubsetBlocks,
        TSize begin,
        ISparseArrayIndexingBlockIteratorPtr<TSize>* iterator,
        TSize* nonDefaultBegin) {

        TSparseSubsetBlocksIteratorData<TSize> iteratorData;

        GetSparseSubsetBlocksIteratorDataAndNonDefaultBegin(
            sparseSubsetBlocks,
            begin,
            &iteratorData,
            nonDefaultBegin);

        *iterator = MakeHolder<TSparseSubsetBlocksBlockIterator<TSize>>(std::move(iteratorData));
    }

    template <class TSize>
    void GetSparseSubsetHybridIndexIteratorDataAndNonDefaultBegin(
        const TSparseSubsetHybridIndex<TSize>& sparseSubsetHybridIndex,
        TSize begin,
        TSparseSubsetHybridIndexIteratorData<TSize>* iteratorData,
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
                    = (TSize)std::popcount((*blockBitmapsCurrent) & ((1ULL << inBlockIdx) - 1));
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
                [] (TSize sum, ui64 element) { return sum + (TSize)std::popcount(element); });

        iteratorData->BlockIndicesCurrent = blockIndicesCurrent;
        iteratorData->BlockIndicesEnd = blockIndices.end();
        iteratorData->BlockBitmapsCurrent = blockBitmapsCurrent;
        iteratorData->InBlockIdx = inBlockIdx;
    }

    template <class TSize>
    void GetIteratorAndNonDefaultBeginImpl(
        const TSparseSubsetHybridIndex<TSize>& sparseSubsetHybridIndex,
        TSize begin,
        IDynamicIteratorPtr<TSize>* iterator,
        TSize* nonDefaultBegin) {

        TSparseSubsetHybridIndexIteratorData<TSize> iteratorData;

        GetSparseSubsetHybridIndexIteratorDataAndNonDefaultBegin(
            sparseSubsetHybridIndex,
            begin,
            &iteratorData,
            nonDefaultBegin);

        *iterator = MakeHolder<TSparseSubsetHybridIndexIterator<TSize>>(std::move(iteratorData));
    }

    template <class TSize>
    void GetBlockIteratorAndNonDefaultBeginImpl(
        const TSparseSubsetHybridIndex<TSize>& sparseSubsetHybridIndex,
        TSize begin,
        ISparseArrayIndexingBlockIteratorPtr<TSize>* iterator,
        TSize* nonDefaultBegin) {

        TSparseSubsetHybridIndexIteratorData<TSize> iteratorData;

        GetSparseSubsetHybridIndexIteratorDataAndNonDefaultBegin(
            sparseSubsetHybridIndex,
            begin,
            &iteratorData,
            nonDefaultBegin);

        *iterator = MakeHolder<TSparseSubsetHybridIndexBlockIterator<TSize>>(std::move(iteratorData));
    }

    template <class TSize>
    void TSparseArrayIndexing<TSize>::GetIteratorAndNonDefaultBegin(
        TSize begin,
        IDynamicIteratorPtr<TSize>* iterator,
        TSize* nonDefaultBegin) const {

        std::visit(
            [&](const auto& impl) {
                GetIteratorAndNonDefaultBeginImpl(impl, begin, iterator, nonDefaultBegin);
            },
            Impl);
    }

    template <class TSize>
    void TSparseArrayIndexing<TSize>::GetBlockIteratorAndNonDefaultBegin(
        TSize begin,
        ISparseArrayIndexingBlockIteratorPtr<TSize>* iterator,
        TSize* nonDefaultBegin) const {

        std::visit(
            [&](const auto& impl) {
                GetBlockIteratorAndNonDefaultBeginImpl(impl, begin, iterator, nonDefaultBegin);
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

    /*
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
    */

    template <class TValue>
    class TNonDefaultValuesBlockIterator<std::remove_const_t<TValue>, TMaybeOwningArrayHolder<TValue>> final
        : public IDynamicExactBlockIterator<std::remove_const_t<TValue>>
    {
    public:
        using TNonConstValue = std::remove_const_t<TValue>;

    public:
        explicit TNonDefaultValuesBlockIterator(
            const TMaybeOwningArrayHolder<TValue>& container,
            size_t offset = 0
        )
            : Impl(TConstArrayRef<TNonConstValue>(container.begin() + offset, container.end()))
        {}

        inline TConstArrayRef<TNonConstValue> NextExact(size_t exactBlockSize) override {
            return Impl.NextExact(exactBlockSize);
        }

    private:
        TArrayBlockIterator<TNonConstValue> Impl;
    };

    template <class TValue>
    class TNonDefaultValuesBlockIterator<TValue, TCompressedArray> final
        : public IDynamicExactBlockIterator<TValue>
    {
    public:
        explicit TNonDefaultValuesBlockIterator(const TCompressedArray& container, size_t offset = 0)
            : Impl(container.GetTypedBlockIterator<TValue>(offset))
        {}

        inline TConstArrayRef<TValue> NextExact(size_t exactBlockSize) override {
            return Impl->NextExact(exactBlockSize);
        }

    private:
        IDynamicExactBlockIteratorPtr<TValue> Impl;
    };

    template <class TValue>
    class TNonDefaultValuesBlockIterator<TValue, TTypedSequenceContainer<TValue>>
        : public IDynamicExactBlockIterator<TValue>
    {
    public:
        explicit TNonDefaultValuesBlockIterator(
            const TTypedSequenceContainer<TValue>& container,
            size_t offset = 0
        )
            : Impl(container.GetImpl().GetBlockIterator(TIndexRange<ui32>(offset, container.GetSize())))
        {}

        inline TConstArrayRef<TValue> NextExact(size_t exactBlockSize) override {
            return Impl->NextExact(exactBlockSize);
        }

    private:
        IDynamicExactBlockIteratorPtr<TValue> Impl;
    };


    template <class TBaseValue, class TInterfaceValue, class TContainer, class TSize, class TTransformer>
    TSparseArrayBaseBlockIterator<TBaseValue, TInterfaceValue, TContainer, TSize, TTransformer>::TSparseArrayBaseBlockIterator(
        TSize size,
        ISparseArrayIndexingBlockIteratorPtr<TSize> indexingBlockIteratorPtr,
        TBaseValue defaultValue,
        TNonDefaultValuesBlockIterator<TBaseValue, TContainer>&& nonDefaultValuesBlockIterator,
        TSize offset,
        TTransformer&& transformer)
        : Index(offset)
        , Size(size)
        , IndexingBlockIteratorPtr(std::move(indexingBlockIteratorPtr))
        , NonDefaultValuesBlockIterator(std::move(nonDefaultValuesBlockIterator))
        , TransformedDefaultValue(transformer(defaultValue))
        , Transformer(std::move(transformer))
    {}

    template <class TBaseValue, class TInterfaceValue, class TContainer, class TSize, class TTransformer>
    inline
    TConstArrayRef<TInterfaceValue>
        TSparseArrayBaseBlockIterator<TBaseValue, TInterfaceValue, TContainer, TSize, TTransformer>::Next(size_t maxBlockSize) {
            const TSize blockSize = Min(Size - Index, (TSize)Min(size_t(Max<TSize>()), maxBlockSize));
            Buffer.yresize(blockSize);
            Fill(Buffer.begin(), Buffer.end(), TransformedDefaultValue);
            const TSize blockEnd = Index + blockSize;

            TConstArrayRef<TSize> nonDefaultIndices = IndexingBlockIteratorPtr->NextUpToBound(blockEnd);

            TConstArrayRef<TBaseValue> nonDefaultValues = NonDefaultValuesBlockIterator.NextExact(
                nonDefaultIndices.size()
            );
            auto nonDefaultValuesIterator = nonDefaultValues.begin();

            for (auto nonDefaultIdx : nonDefaultIndices) {
                Buffer[nonDefaultIdx - Index] = Transformer(*nonDefaultValuesIterator++);
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
        if (DefaultValue != rhs.DefaultValue) {
            return false;
        }
        return NonDefaultValues.EqualTo(rhs.NonDefaultValues, strict);
    }

    template <class TValue, class TContainer, class TSize>
    template <class F>
    inline void TSparseArrayBase<TValue, TContainer, TSize>::ForEachNonDefault(
        F&& f,
        TSize maxBlockSize
    ) const {
        ForBlockNonDefault(
            [f] (auto indexingBlock, auto valuesBlock) {
                for (auto i : xrange(indexingBlock.size())) {
                    f(indexingBlock[i], valuesBlock[i]);
                }
            },
            maxBlockSize
        );
    }

    template <class TValue, class TContainer, class TSize>
    template <class F>
    inline void TSparseArrayBase<TValue, TContainer, TSize>::ForBlockNonDefault(
        F&& f,
        TSize maxBlockSize
    ) const {
        ISparseArrayIndexingBlockIteratorPtr<TSize> indexingBlockIterator;
        TSize nonDefaultBegin = 0;
        Indexing->GetBlockIteratorAndNonDefaultBegin(
            /*begin*/ 0,
            &indexingBlockIterator,
            &nonDefaultBegin
        );

        TNonDefaultValuesBlockIterator<TNonConstValue, TContainer> nonDefaultValuesBlockIterator(
            NonDefaultValues,
            /*offset*/ 0
        );

        while (auto indexingBlock = indexingBlockIterator->Next(maxBlockSize)) {
            TConstArrayRef<TValue> valuesBlock = nonDefaultValuesBlockIterator.NextExact(
                indexingBlock.size()
            );
            f(indexingBlock, valuesBlock);
        }
    }

    /*
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
    */

    template <class TValue, class TContainer, class TSize>
    TVector<typename TSparseArrayBase<TValue, TContainer, TSize>::TNonConstValue>
        TSparseArrayBase<TValue, TContainer, TSize>::ExtractValues() const {
            using TNonConstValue = typename TSparseArrayBase<TValue, TContainer, TSize>::TNonConstValue;

            TVector<TNonConstValue> result;
            result.yresize(GetSize());
            Fill(result.begin(), result.end(), GetDefaultValue());

            TArrayRef<TNonConstValue> resultRef = result;

            ForEachNonDefault(
                [=] (ui32 idx, TNonConstValue value) {
                    resultRef[idx] = value;
                }
            );
            return result;
        }

    /*
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
    */


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

    template <class TDstValue, class TSrcValue, class TSize>
    TTypedSequenceContainer<TDstValue> CreateSubsetContainer(
        TVector<TSrcValue>&& subsetNonDefaultValues,
        const TSparseArrayBase<const TDstValue, TTypedSequenceContainer<TDstValue>, TSize>& parent) {

        Y_UNUSED(parent);

        return TTypedSequenceContainer<TDstValue>(
            MakeIntrusive<TTypeCastArrayHolder<TDstValue, TDstValue>>(
                TMaybeOwningConstArrayHolder<TDstValue>::CreateOwning(std::move(subsetNonDefaultValues))
            )
        );
    }

    // in addition to dstValues passed to CreateSubsetContainer
    template <class TValue, class TSize>
    size_t EstimateContainerCreationAdditionalCpuRamUsage(
        const TSparseArrayBase<const TValue, TTypedSequenceContainer<TValue>, TSize>& parent) {

        Y_UNUSED(parent);

        return 0;
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

        if (std::holds_alternative<TFullSubset<TSize>>(subsetInvertedIndexing)) {
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
                CB_ENSURE(false, "Unexpected sparse array indexing type");
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

        if (std::holds_alternative<TFullSubset<TSize>>(subsetInvertedIndexing)) {
            return *this;
        }

        const TInvertedIndexedSubset<TSize>& invertedIndexedSubset
            = std::get<TInvertedIndexedSubset<TSize>>(subsetInvertedIndexing);

        TConstArrayRef<TSize> invertedIndicesArray = invertedIndexedSubset.GetMapping();

        using TNonConstValue = typename std::remove_const<TValue>::type;

        TVector<TSize> dstVectorIndexing;
        TVector<TNonConstValue> dstValues;

        ForEachNonDefault(
            [&](TSize srcIdx, TNonConstValue value) {
                auto dstIdx = invertedIndicesArray[srcIdx];
                if (dstIdx != TInvertedIndexedSubset<TSize>::NOT_PRESENT) {
                    dstVectorIndexing.push_back(dstIdx);
                    dstValues.push_back(value);
                }
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


    template <class TDstValue, class TSize>
    TConstPolymorphicValuesSparseArray<TDstValue, TSize> MakeConstPolymorphicValuesSparseArrayGeneric(
        TSparseArrayIndexingPtr<TSize> indexing,
        ITypedSequencePtr<TDstValue> nonDefaultValues,
        TDstValue defaultValue) {

        return TConstPolymorphicValuesSparseArray<TDstValue, TSize>(
            std::move(indexing),
            TTypedSequenceContainer<TDstValue>(std::move(nonDefaultValues)),
            std::move(defaultValue)
        );
    }

    template <class TDstValue, class TSrcValue, class TSize>
    TConstPolymorphicValuesSparseArray<TDstValue, TSize> MakeConstPolymorphicValuesSparseArray(
        TSparseArrayIndexingPtr<TSize> indexing,
        TMaybeOwningConstArrayHolder<TSrcValue> nonDefaultValues,
        TDstValue defaultValue) {

        return TConstPolymorphicValuesSparseArray<TDstValue, TSize>(
            std::move(indexing),
            TTypedSequenceContainer<TDstValue>(
                MakeIntrusive<TTypeCastArrayHolder<TDstValue, TSrcValue>>(std::move(nonDefaultValues))
            ),
            std::move(defaultValue)
        );
    }

    template <class TDstValue, class TSize>
    TConstPolymorphicValuesSparseArray<TDstValue, TSize> MakeConstPolymorphicValuesSparseArrayWithArrayIndexGeneric(
        TSize size,
        TMaybeOwningConstArrayHolder<TSize> indices,
        ITypedSequencePtr<TDstValue> nonDefaultValues,
        bool ordered,
        TDstValue defaultValue) {

        if (ordered) {
            return MakeConstPolymorphicValuesSparseArrayGeneric(
                MakeIntrusive<TSparseArrayIndexing<TSize>>(
                    TSparseSubsetIndices<TSize>(std::move(indices)),
                    size
                ),
                std::move(nonDefaultValues),
                std::move(defaultValue)
            );
        } else {
            TVector<TSize> indexingCopy(indices.begin(), indices.end());
            TVector<TDstValue> nonDefaultValuesCopy;
            nonDefaultValuesCopy.yresize(nonDefaultValues->GetSize());
            TDstValue* dstIterator = nonDefaultValuesCopy.data();

            nonDefaultValues->ForEach(
                [&dstIterator] (TDstValue value) {
                    *dstIterator++ = value;
                }
            );

            std::function<TTypedSequenceContainer<TDstValue>(TVector<TDstValue>&&)>
                createNonDefaultValues
                    = [&] (TVector<TDstValue>&& values) {
                        return TTypedSequenceContainer<TDstValue>(
                            MakeIntrusive<TTypeCastArrayHolder<TDstValue, TDstValue>>(
                                TMaybeOwningConstArrayHolder<TDstValue>::CreateOwning(std::move(values))
                            )
                        );
                    };

            return MakeSparseArrayBase<const TDstValue, TTypedSequenceContainer<TDstValue>>(
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


    template <class TDstValue, class TSrcValue, class TSize>
    TConstPolymorphicValuesSparseArray<TDstValue, TSize> MakeConstPolymorphicValuesSparseArrayWithArrayIndex(
        TSize size,
        TMaybeOwningConstArrayHolder<TSize> indices,
        TMaybeOwningConstArrayHolder<TSrcValue> nonDefaultValues,
        bool ordered,
        TDstValue defaultValue) {

        return MakeConstPolymorphicValuesSparseArrayWithArrayIndexGeneric<TDstValue>(
            size,
            std::move(indices),
            MakeIntrusive<TTypeCastArrayHolder<TDstValue, TSrcValue>>(std::move(nonDefaultValues)),
            ordered,
            defaultValue
        );
    }

}
