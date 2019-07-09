#pragma once

#include "compression.h"
#include "dynamic_iterator.h"
#include "maybe_owning_array_holder.h"

#include <library/binsaver/bin_saver.h>

#include <util/generic/algorithm.h>
#include <util/generic/array_ref.h>
#include <util/generic/maybe.h>
#include <util/generic/ptr.h>
#include <util/generic/strbuf.h>
#include <util/generic/variant.h>
#include <util/generic/vector.h>
#include <util/system/types.h>

#include <climits>
#include <tuple>
#include <type_traits>


namespace NCB {

    template <class TSize>
    class TArraySubsetInvertedIndexing;


    static_assert(CHAR_BIT == 8, "CatBoost requires CHAR_BIT == 8");

    template <class T>
    void CheckIsIncreasingIndicesArray(TConstArrayRef<T> data, TStringBuf arrayName, bool isInternalError);


    template <class TSize>
    struct TSparseSubsetIndices : public TMaybeOwningArrayHolder<TSize> {
        static_assert(std::is_integral<TSize>::value);

    public:
        // needed because of IBinSaver
        TSparseSubsetIndices() = default;

        explicit TSparseSubsetIndices(TMaybeOwningArrayHolder<TSize>&& base)
            : TMaybeOwningArrayHolder<TSize>(std::move(base))
        {}

        explicit TSparseSubsetIndices(TVector<TSize>&& indices)
            : TMaybeOwningArrayHolder<TSize>(TMaybeOwningArrayHolder<TSize>::CreateOwning(std::move(indices)))
        {}

        void Check() const {
            CheckIsIncreasingIndicesArray(**this, "Sparse array indices array", false);
        }

        TSize GetSize() const {
            return (TSize)(**this).size();
        }

        TSize GetUpperBound() const {
            return (**this).empty() ? 0 : ((**this).back() + 1);
        }
    };


    template <class TSize>
    struct TSparseSubsetBlocks {
        static_assert(std::is_integral<TSize>::value);

    public:
        TMaybeOwningArrayHolder<TSize> BlockStarts;
        TMaybeOwningArrayHolder<TSize> BlockLengths;

    public:
        // needed because of IBinSaver
        TSparseSubsetBlocks() = default;

        TSparseSubsetBlocks(
            TMaybeOwningArrayHolder<TSize>&& blockStarts,
            TMaybeOwningArrayHolder<TSize>&& blockLengths)
            : BlockStarts(std::move(blockStarts))
            , BlockLengths(std::move(blockLengths))
        {}

        TSparseSubsetBlocks(TVector<TSize>&& blockStarts, TVector<TSize>&& blockLengths)
            : BlockStarts(TMaybeOwningArrayHolder<TSize>::CreateOwning(std::move(blockStarts)))
            , BlockLengths(TMaybeOwningArrayHolder<TSize>::CreateOwning(std::move(blockLengths)))
        {}

        SAVELOAD(BlockStarts, BlockLengths)

        bool operator==(const TSparseSubsetBlocks& rhs) const {
            return std::tie(BlockStarts, BlockLengths) == std::tie(rhs.BlockStarts, rhs.BlockLengths);
        }

        void Check() const;

        TSize GetSize() const {
            return Accumulate(*BlockLengths, TSize(0));
        }

        TSize GetUpperBound() const {
            return (*BlockStarts).empty() ? TSize(0) : ((*BlockStarts).back() + (*BlockLengths).back());
        }
    };

    template <class TSize>
    class TSparseSubsetBlocksIterator final : public IDynamicIterator<TSize> {
    public:
        TSparseSubsetBlocksIterator(const TSparseSubsetBlocks<TSize>& sparseSubsetBlocks);

        inline TMaybe<TSize> Next() override;

    private:
        const TSize* BlockStartsCurrent;
        const TSize* BlockStartsEnd;
        const TSize* BlockLengthsCurrent;
        TSize InBlockIdx;
    };


    /* value with index I is present in the sparse array if Block with value ceil(I / 64)
     * is present in BlockIndices and bit NonEmptyBlockBitmaps[blockIdx][I % 64] is set
     */
    template <class TSize>
    struct TSparseSubsetHybridIndex {
        static_assert(std::is_integral<TSize>::value);

    public:
        constexpr static TSize BLOCK_SIZE = sizeof(ui64) * CHAR_BIT;

    public:
        // non-empty 64-value blocks
        TVector<TSize> BlockIndices;

        // bit mask for values in nonempty blocks
        TVector<ui64> BlockBitmaps;

    public:
        SAVELOAD(BlockIndices, BlockBitmaps)

        bool operator==(const TSparseSubsetHybridIndex& rhs) const {
            return std::tie(BlockIndices, BlockBitmaps) == std::tie(rhs.BlockIndices, rhs.BlockBitmaps);
        }

        void Check() const;

        TSize GetSize() const;

        TSize GetUpperBound() const;
    };


    template <class TSize>
    class TSparseSubsetHybridIndexIterator final : public IDynamicIterator<TSize> {
    public:
        TSparseSubsetHybridIndexIterator(const TSparseSubsetHybridIndex<TSize>& sparseSubsetHybridIndex);

        inline TMaybe<TSize> Next() override;

    private:
        const TSize* BlockIndicesCurrent;
        const TSize* BlockIndicesEnd;
        const ui64* BlockBitmapsCurrent;
        ui32 InBlockIdx;
    };


    // useful to specify desired index type when creating
    enum class ESparseArrayIndexingType {
        Indices,
        Blocks,
        HybridIndex,
        Undefined
    };


    /*
     * Derived from TThrRefBase because it might be useful to share the same TSparseArrayIndexing
     *  for different arrays.
     * For example, for the raw and quantized values arrays for the single feature or if several
     *  sparse features have non-default values for the same dataset subset
     */
    template <class TSize>
    class TSparseArrayIndexing final : public TThrRefBase {
        static_assert(std::is_integral<TSize>::value);

    public:
        using TImpl =
            TVariant<TSparseSubsetIndices<TSize>, TSparseSubsetBlocks<TSize>, TSparseSubsetHybridIndex<TSize>>;

    public:
        // needed for IBinSaver serialization
        TSparseArrayIndexing() = default;

        explicit TSparseArrayIndexing(TImpl&& impl, TMaybe<TSize> size = Nothing(), bool skipCheck = false);

        SAVELOAD(NonDefaultSize, Size, Impl);

        // comparison is strict by default, useful for unit tests
        bool operator==(const TSparseArrayIndexing& rhs) const {
            return EqualTo(rhs, true);
        }

        // if strict is true compare bit-by-bit, else compare values
        bool EqualTo(const TSparseArrayIndexing& rhs, bool strict = true) const;

        TSize GetNonDefaultSize() const {
            return NonDefaultSize;
        }

        TSize GetSize() const {
            return Size;
        }

        ESparseArrayIndexingType GetType() const;

        const TImpl& GetImpl() const {
            return Impl;
        }

        // f is a visitor function that will be repeatedly called with (index) argument
        template <class F>
        inline void ForEachNonDefault(F&& f) const;

        // f is a visitor function that will be repeatedly called with (index, isDefault) arguments
        template <class F>
        inline void ForEach(F&& f) const;

        IDynamicIteratorPtr<TSize> GetIterator() const;

    private:
        void InitSize(TMaybe<TSize> sizeArg, TSize indicesUpperBound);

    private:
        TImpl Impl;
        TSize NonDefaultSize;
        TSize Size;
    };


    template <class TSize>
    struct ISparseArrayIndexingBuilder {
        virtual ~ISparseArrayIndexingBuilder() = default;

        /* By default, it is assumed that AddOrdered() is called with sequentially increasing indices
         *  if order of indices is undefined, use AddNonOrdered instead
         */
        virtual void AddOrdered(TSize i) = 0;
        virtual void AddNonOrdered(TSize i) = 0;

        // this method is intended to be called only once at the end
        virtual TSparseArrayIndexing<TSize> Build() = 0;
    };

    template <class TSize>
    using TSparseArrayIndexingBuilderPtr = THolder<ISparseArrayIndexingBuilder<TSize>>;


    template <class TSize>
    class TSparseSubsetIndicesBuilder final : public ISparseArrayIndexingBuilder<TSize> {
    public:
        inline void AddOrdered(TSize i) override;

        inline void AddNonOrdered(TSize i) override;

        TSparseArrayIndexing<TSize> Build() override;

    private:
        bool NonOrdered = false;
        TVector<TSize> Indices;
    };

    template <class TSize>
    class TSparseSubsetBlocksBuilder final : public ISparseArrayIndexingBuilder<TSize> {
    public:
        inline void AddOrdered(TSize i) override;

        inline void AddNonOrdered(TSize i) override;

        TSparseArrayIndexing<TSize> Build() override;

    private:
        inline void AddImpl(TSize i);

    private:
        bool NonOrdered = false;
        TVector<TSize> BlockStarts;
        TVector<TSize> BlockLengths;
    };


    template <class TSize>
    class TSparseSubsetHybridIndexBuilder final : public ISparseArrayIndexingBuilder<TSize> {
    public:
        inline void AddOrdered(TSize i) override;

        inline void AddNonOrdered(TSize i) override;

        TSparseArrayIndexing<TSize> Build() override;

    private:
        inline void AddImpl(TSize i);

    private:
        bool NonOrdered = false;
        TVector<TSize> BlockIndices;
        TVector<ui64> BlockBitmaps;
    };


    template <class TSize>
    TSparseArrayIndexingBuilderPtr<TSize> CreateSparseArrayIndexingBuilder(ESparseArrayIndexingType type);


    template <class TValue, class TContainer, class TSize = size_t>
    class TSparseArrayBaseIterator final : public IDynamicSparseIterator<TValue, TSize> {
    public:
        TSparseArrayBaseIterator(
            IDynamicIteratorPtr<TSize> indexingIteratorPtr,
            const TContainer& nonDefaultValues);

        inline TMaybe<std::pair<TSize, TValue>> Next() override;

    private:
        IDynamicIteratorPtr<TSize> IndexingIteratorPtr;
        TSize NonDefaultIndex;
        const TContainer& NonDefaultValues;
    };

    template <class TValue, class TContainer, class TSize = size_t>
    using TSparseArrayBaseIteratorPtr = THolder<TSparseArrayBaseIterator<TValue, TContainer, TSize>>;

    /*
     * TContainer must implement GetSize and 'operator[]'
     */
    template <class TValue, class TContainer, class TSize = size_t>
    class TSparseArrayBase final : public TThrRefBase {
        static_assert(std::is_integral<TSize>::value);

    public:
        using TIndexing = TSparseArrayIndexing<TSize>;
        using TIndexingPtr = TIntrusivePtr<TSparseArrayIndexing<TSize>>;
        using TIndexingImpl = typename TIndexing::TImpl;

    public:
        // needed because of IBinSaver
        TSparseArrayBase() = default;

        TSparseArrayBase(
            TIndexingPtr indexing,
            TContainer&& nonDefaultValues,
            TValue&& defaultValue = TValue(0));

        template <class TIndexingArg>
        TSparseArrayBase(
            TIndexingArg&& indexing,
            TContainer&& nonDefaultValues,
            TValue&& defaultValue = TValue(0));

        int operator&(IBinSaver& binSaver);

        // comparison is strict by default, useful for unit tests
        bool operator==(const TSparseArrayBase& rhs) const {
            return EqualTo(rhs, true);
        }

        // if strict is true compare bit-by-bit, else compare values
        bool EqualTo(const TSparseArrayBase& rhs, bool strict = true) const;

        TSize GetNonDefaultSize() const {
            return Indexing->GetNonDefaultSize();
        }

        TSize GetSize() const {
            return Indexing->GetSize();
        }

        const TValue& GetDefaultValue() const {
            return DefaultValue;
        }

        // f is a visitor function that will be repeatedly called with (index, value) arguments
        template <class F>
        inline void ForEachNonDefault(F&& f) const;

        // f is a visitor function that will be repeatedly called with (index, value) arguments
        template <class F>
        inline void ForEach(F&& f) const;

        TVector<TValue> ExtractValues() const;

        TSparseArrayBaseIteratorPtr<TValue, TContainer, TSize> GetIterator();

        size_t EstimateGetSubsetCpuRamUsage(
            const TArraySubsetInvertedIndexing<TSize>& subsetInvertedIndexing,
            ESparseArrayIndexingType sparseArrayIndexingType = ESparseArrayIndexingType::Undefined) const;

        TSparseArrayBase GetSubset(
            const TArraySubsetInvertedIndexing<TSize>& subsetInvertedIndexing,

            // if undefined - use the same indexing type as *this
            ESparseArrayIndexingType sparseArrayIndexingType = ESparseArrayIndexingType::Undefined) const;

        // could be used in low-level code

        TIndexingPtr GetIndexing() const {
            return Indexing;
        }

        const TContainer& GetNonDefaultValues() const {
            return NonDefaultValues;
        }

    private:
        TIndexingPtr Indexing;
        TContainer NonDefaultValues;
        TValue DefaultValue;
    };

    template <class TValue, class TSize>
    using TSparseArray = TSparseArrayBase<TValue, TMaybeOwningArrayHolder<TValue>, TSize>;

    template <class TValue, class TSize>
    using TSparseCompressedArray = TSparseArrayBase<TValue, TCompressedArray, TSize>;
}

#include "sparse_array-inl.h"

