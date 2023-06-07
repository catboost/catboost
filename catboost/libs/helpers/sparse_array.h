#pragma once

#include "compression.h"
#include "dynamic_iterator.h"
#include "maybe_owning_array_holder.h"
#include "polymorphic_type_containers.h"

#include <library/cpp/binsaver/bin_saver.h>

#include <util/generic/algorithm.h>
#include <util/generic/array_ref.h>
#include <util/generic/maybe.h>
#include <util/generic/ptr.h>
#include <util/generic/strbuf.h>
#include <util/generic/variant.h>
#include <util/generic/vector.h>
#include <util/system/types.h>

#include <climits>
#include <functional>
#include <tuple>
#include <type_traits>


namespace NCB {

    template <class TSize>
    class TArraySubsetInvertedIndexing;


    static_assert(CHAR_BIT == 8, "CatBoost requires CHAR_BIT == 8");

    template <class T>
    void CheckIsIncreasingIndicesArray(TConstArrayRef<T> data, TStringBuf arrayName, bool isInternalError);


    template <class TSize>
    class ISparseArrayIndexingBlockIterator : public IDynamicBlockIterator<TSize> {
    public:
        virtual TConstArrayRef<TSize> NextUpToBound(TSize upperBound) = 0;
    };

    template <class TSize>
    using ISparseArrayIndexingBlockIteratorPtr = THolder<ISparseArrayIndexingBlockIterator<TSize>>;


    template <class TSize>
    struct TSparseSubsetIndices : public TMaybeOwningConstArrayHolder<TSize> {
        static_assert(std::is_integral<TSize>::value);

    public:
        // needed because of IBinSaver
        TSparseSubsetIndices() = default;

        explicit TSparseSubsetIndices(TMaybeOwningConstArrayHolder<TSize>&& base)
            : TMaybeOwningConstArrayHolder<TSize>(std::move(base))
        {}

        explicit TSparseSubsetIndices(TVector<TSize>&& indices)
            : TMaybeOwningConstArrayHolder<TSize>(
                TMaybeOwningConstArrayHolder<TSize>::CreateOwning(std::move(indices))
            )
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
    class TSparseSubsetIndicesBlockIterator final : public ISparseArrayIndexingBlockIterator<TSize> {
    public:
        TSparseSubsetIndicesBlockIterator(const TSize* current, const TSize* end);

        TConstArrayRef<TSize> Next(size_t maxBlockSize = Max<size_t>()) override;
        TConstArrayRef<TSize> NextUpToBound(TSize upperBound) override;

    private:
        const TSize* Current;
        const TSize* const End;
    };


    template <class TSize>
    struct TSparseSubsetBlocks {
        static_assert(std::is_integral<TSize>::value);

    public:
        TMaybeOwningConstArrayHolder<TSize> BlockStarts;
        TMaybeOwningConstArrayHolder<TSize> BlockLengths;

    public:
        // needed because of IBinSaver
        TSparseSubsetBlocks() = default;

        TSparseSubsetBlocks(
            TMaybeOwningConstArrayHolder<TSize>&& blockStarts,
            TMaybeOwningConstArrayHolder<TSize>&& blockLengths)
            : BlockStarts(std::move(blockStarts))
            , BlockLengths(std::move(blockLengths))
        {}

        TSparseSubsetBlocks(TVector<TSize>&& blockStarts, TVector<TSize>&& blockLengths)
            : BlockStarts(TMaybeOwningConstArrayHolder<TSize>::CreateOwning(std::move(blockStarts)))
            , BlockLengths(TMaybeOwningConstArrayHolder<TSize>::CreateOwning(std::move(blockLengths)))
        {}

        SAVELOAD(BlockStarts, BlockLengths);

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
    struct TSparseSubsetBlocksIteratorData {
        const TSize* BlockStartsCurrent = nullptr;
        const TSize* BlockStartsEnd = nullptr;
        const TSize* BlockLengthsCurrent = nullptr;
        TSize InBlockIdx = 0;
    };

    template <class TSize>
    class TSparseSubsetBlocksIterator final : public IDynamicIterator<TSize> {
    public:
        TSparseSubsetBlocksIterator(const TSparseSubsetBlocks<TSize>& sparseSubsetBlocks);

        TSparseSubsetBlocksIterator(TSparseSubsetBlocksIteratorData<TSize>&& data);

        inline bool Next(TSize* ) override;

    private:
        TSparseSubsetBlocksIteratorData<TSize> Data;
    };

    template <class TSize>
    class TSparseSubsetBlocksBlockIterator final : public ISparseArrayIndexingBlockIterator<TSize> {
    public:
        TSparseSubsetBlocksBlockIterator(TSparseSubsetBlocksIteratorData<TSize>&& data);

        TConstArrayRef<TSize> Next(size_t maxBlockSize = Max<size_t>()) override;
        TConstArrayRef<TSize> NextUpToBound(TSize upperBound) override;

    private:
        TSparseSubsetBlocksIteratorData<TSize> Data;
        TVector<TSize> Buffer;
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
        SAVELOAD(BlockIndices, BlockBitmaps);

        bool operator==(const TSparseSubsetHybridIndex& rhs) const {
            return std::tie(BlockIndices, BlockBitmaps) == std::tie(rhs.BlockIndices, rhs.BlockBitmaps);
        }

        void Check() const;

        TSize GetSize() const;

        TSize GetUpperBound() const;
    };

    template <class TSize>
    struct TSparseSubsetHybridIndexIteratorData {
        const TSize* BlockIndicesCurrent = nullptr;
        const TSize* BlockIndicesEnd = nullptr;
        const ui64* BlockBitmapsCurrent = nullptr;
        ui32 InBlockIdx = 0;
    };


    template <class TSize>
    class TSparseSubsetHybridIndexIterator final : public IDynamicIterator<TSize> {
    public:
        TSparseSubsetHybridIndexIterator(const TSparseSubsetHybridIndex<TSize>& sparseSubsetHybridIndex);

        TSparseSubsetHybridIndexIterator(TSparseSubsetHybridIndexIteratorData<TSize>&& data);

        inline bool Next(TSize* value) override;

    private:
        TSparseSubsetHybridIndexIteratorData<TSize> Data;
    };

    template <class TSize>
    class TSparseSubsetHybridIndexBlockIterator final : public ISparseArrayIndexingBlockIterator<TSize> {
    public:
        TSparseSubsetHybridIndexBlockIterator(TSparseSubsetHybridIndexIteratorData<TSize>&& data);

        TConstArrayRef<TSize> Next(size_t maxBlockSize = Max<size_t>()) override;
        TConstArrayRef<TSize> NextUpToBound(TSize upperBound) override;

    private:
        TSparseSubsetHybridIndexIteratorData<TSize> Data;
        TVector<TSize> Buffer;
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
            std::variant<TSparseSubsetIndices<TSize>, TSparseSubsetBlocks<TSize>, TSparseSubsetHybridIndex<TSize>>;

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

        TSize GetDefaultSize() const {
            return Size - NonDefaultSize;
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

        // get iterator and nonDefaultBegin index for subset starting at 'begin'
        void GetIteratorAndNonDefaultBegin(
            TSize begin,
            IDynamicIteratorPtr<TSize>* iterator,
            TSize* nonDefaultBegin) const;

        // get block iterator and nonDefaultBegin index for subset starting at 'begin'
        void GetBlockIteratorAndNonDefaultBegin(
            TSize begin,
            ISparseArrayIndexingBlockIteratorPtr<TSize>* iterator,
            TSize* nonDefaultBegin) const;

    private:
        void InitSize(TMaybe<TSize> sizeArg, TSize indicesUpperBound);

    private:
        TImpl Impl;
        TSize NonDefaultSize;
        TSize Size;
    };

    template <class TSize>
    using TSparseArrayIndexingPtr = TIntrusivePtr<TSparseArrayIndexing<TSize>>;


    template <class TSize>
    struct ISparseArrayIndexingBuilder {
        virtual ~ISparseArrayIndexingBuilder() = default;

        /* By default, it is assumed that AddOrdered() is called with sequentially increasing indices
         *  if order of indices is undefined, use AddNonOrdered instead
         */
        virtual void AddOrdered(TSize i) = 0;
        virtual void AddNonOrdered(TSize i) = 0;

        // this method is intended to be called only once at the end
        virtual TSparseArrayIndexing<TSize> Build(TMaybe<TSize> size = Nothing()) = 0;
    };

    template <class TSize>
    using TSparseArrayIndexingBuilderPtr = THolder<ISparseArrayIndexingBuilder<TSize>>;


    template <class TSize>
    class TSparseSubsetIndicesBuilder final : public ISparseArrayIndexingBuilder<TSize> {
    public:
        inline void AddOrdered(TSize i) override;

        inline void AddNonOrdered(TSize i) override;

        TSparseArrayIndexing<TSize> Build(TMaybe<TSize> size = Nothing()) override;

    private:
        bool NonOrdered = false;
        TVector<TSize> Indices;
    };

    template <class TSize>
    class TSparseSubsetBlocksBuilder final : public ISparseArrayIndexingBuilder<TSize> {
    public:
        inline void AddOrdered(TSize i) override;

        inline void AddNonOrdered(TSize i) override;

        TSparseArrayIndexing<TSize> Build(TMaybe<TSize> size = Nothing()) override;

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

        TSparseArrayIndexing<TSize> Build(TMaybe<TSize> size = Nothing()) override;

    private:
        inline void AddImpl(TSize i);

    private:
        bool NonOrdered = false;
        TVector<TSize> BlockIndices;
        TVector<ui64> BlockBitmaps;
    };


    template <class TSize>
    TSparseArrayIndexingBuilderPtr<TSize> CreateSparseArrayIndexingBuilder(ESparseArrayIndexingType type);


    template <class TValue, class TContainer>
    class TNonDefaultValuesBlockIterator final : public IDynamicExactBlockIterator<TValue> {
    public:
        explicit TNonDefaultValuesBlockIterator(const TContainer& container, size_t offset = 0);
        inline TConstArrayRef<TValue> NextExact(size_t exactBlockSize) override;
    };

    template <class TBaseValue, class TInterfaceValue, class TContainer, class TSize, class TTransformer>
    class TSparseArrayBaseBlockIterator final : public IDynamicBlockIterator<TInterfaceValue> {
    public:
        TSparseArrayBaseBlockIterator(
            TSize size,
            ISparseArrayIndexingBlockIteratorPtr<TSize> indexingBlockIteratorPtr,
            TBaseValue defaultValue,
            TNonDefaultValuesBlockIterator<TBaseValue, TContainer>&& nonDefaultValuesBlockIterator,
            TSize offset,
            TTransformer&& transformer);

        inline TConstArrayRef<TInterfaceValue> Next(size_t maxBlockSize = Max<size_t>()) override;

    public:
        TSize Index;
        TSize Size;
        ISparseArrayIndexingBlockIteratorPtr<TSize> IndexingBlockIteratorPtr;
        TNonDefaultValuesBlockIterator<TBaseValue, TContainer> NonDefaultValuesBlockIterator;

        TInterfaceValue TransformedDefaultValue;
        TVector<TInterfaceValue> Buffer;
        TTransformer Transformer;
    };


    /*
     * TContainer must implement operator&(IBinSaver&), GetSize, EqualTo(rhs, strict)
     *  and TNonDefaultValuesBlockIterator<TContainer> must be defined
     */
    template <class TValue, class TContainer, class TSize = size_t>
    class TSparseArrayBase final : public TThrRefBase {
        static_assert(std::is_integral<TSize>::value);

    public:
        using TIndexing = TSparseArrayIndexing<TSize>;
        using TIndexingPtr = TIntrusivePtr<TSparseArrayIndexing<TSize>>;
        using TIndexingImpl = typename TIndexing::TImpl;
        using TNonConstValue = typename std::remove_const<TValue>::type;

    public:
        // needed because of IBinSaver
        TSparseArrayBase()
            : NonDefaultValues(TContainer())
            , DefaultValue(TValue())
        {}

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

        TSize GetDefaultSize() const {
            return Indexing->GetDefaultSize();
        }

        TSize GetSize() const {
            return Indexing->GetSize();
        }

        const TValue& GetDefaultValue() const {
            return DefaultValue;
        }

        // f is a visitor function that will be repeatedly called with (index, value) arguments
        template <class F>
        inline void ForEachNonDefault(F&& f, TSize maxBlockSize = TSize(128)) const;

        // f is a visitor function that will be repeatedly called with (indexBlock, valueBlock) arguments
        template <class F>
        inline void ForBlockNonDefault(F&& f, TSize maxBlockSize = TSize(128)) const;

        TVector<TNonConstValue> ExtractValues() const;

        template <typename TInterfaceValue, class TTransformer>
        IDynamicBlockIteratorPtr<TInterfaceValue> GetTransformingBlockIterator(TTransformer&& transformer, TSize offset = 0) const {
            ISparseArrayIndexingBlockIteratorPtr<TSize> indexingBlockIterator;
            TSize nonDefaultOffset;
            Indexing->GetBlockIteratorAndNonDefaultBegin(offset, &indexingBlockIterator, &nonDefaultOffset);

            return MakeHolder<TSparseArrayBaseBlockIterator<TNonConstValue, TInterfaceValue, TContainer, TSize, TTransformer>>(
                GetSize(),
                std::move(indexingBlockIterator),
                GetDefaultValue(),
                TNonDefaultValuesBlockIterator<TNonConstValue, TContainer>(NonDefaultValues, nonDefaultOffset),
                offset,
                std::move(transformer)
            );
        }

        IDynamicBlockIteratorPtr<TNonConstValue> GetBlockIterator(TSize offset = 0) const {
            return GetTransformingBlockIterator<TNonConstValue>(TIdentity(), offset);
        }

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
        TNonConstValue DefaultValue;
    };

    template <class TValue, class TContainer, class TSize, class TSrcValue>
    TSparseArrayBase<TValue, TContainer, TSize> MakeSparseArrayBase(
        TSize size,
        TVector<TSize>&& indexing,
        TVector<TSrcValue>&& nonDefaultValues,
        std::function<TContainer(TVector<TSrcValue>&&)>&& createNonDefaultValuesContainer,
        ESparseArrayIndexingType sparseArrayIndexingType = ESparseArrayIndexingType::Indices,
        bool ordered = false,
        TValue&& defaultValue = TValue(0));

    template <class TValue, class TSize>
    using TSparseArray = TSparseArrayBase<TValue, TMaybeOwningArrayHolder<TValue>, TSize>;

    template <class TValue, class TSize>
    using TConstSparseArray = TSparseArray<const TValue, TSize>;


    // TSparseArrayBase TContainer - compatible wrapper for ITypedSequencePtr<TValue>
    template <class T>
    class TTypedSequenceContainer {
    public:
        // needed for BinSaver and Cython
        TTypedSequenceContainer() = default;

        explicit TTypedSequenceContainer(ITypedSequencePtr<T>&& impl)
            : Impl(std::move(impl))
        {}

        int operator&(IBinSaver& binSaver) {
            return Impl & binSaver;
        }

        // if strict is true compare bit-by-bit, else compare values
        bool EqualTo(const TTypedSequenceContainer<T>& rhs, bool strict = true) const {
            return Impl->EqualTo(*(rhs->Impl), strict);
        }

        ui32 GetSize() const {
            return Impl->GetSize();
        }

        const ITypedSequence<T>& GetImpl() const {
            return *Impl;
        }

    private:
        ITypedSequencePtr<T> Impl;
    };


    template <class TValue, class TSize>
    using TConstPolymorphicValuesSparseArray
        = TSparseArrayBase<const TValue, TTypedSequenceContainer<TValue>, TSize>;


    // for Cython
    template <class TSize>
    TSparseArrayIndexingPtr<TSize> MakeSparseArrayIndexing(
        TSize size,
        TMaybeOwningConstArrayHolder<TSize> indices); // already ordered

    template <class TSize>
    TSparseArrayIndexingPtr<TSize> MakeSparseBlockIndexing(
        TSize size,
        TMaybeOwningConstArrayHolder<TSize> blockStarts, // already ordered
        TMaybeOwningConstArrayHolder<TSize> blockLengths); // already ordered

    template <class TDstValue, class TSize>
    TConstPolymorphicValuesSparseArray<TDstValue, TSize> MakeConstPolymorphicValuesSparseArrayGeneric(
        TSparseArrayIndexingPtr<TSize> indexing,
        ITypedSequencePtr<TDstValue> nonDefaultValues,
        TDstValue defaultValue = TDstValue());

    template <class TDstValue, class TSrcValue, class TSize>
    TConstPolymorphicValuesSparseArray<TDstValue, TSize> MakeConstPolymorphicValuesSparseArray(
        TSparseArrayIndexingPtr<TSize> indexing,
        TMaybeOwningConstArrayHolder<TSrcValue> nonDefaultValues,
        TDstValue defaultValue = TDstValue());

    template <class TDstValue, class TSize>
    TConstPolymorphicValuesSparseArray<TDstValue, TSize> MakeConstPolymorphicValuesSparseArrayWithArrayIndexGeneric(
        TSize size,
        TMaybeOwningConstArrayHolder<TSize> indices,
        ITypedSequencePtr<TDstValue> nonDefaultValues,
        bool ordered = false,
        TDstValue defaultValue = TDstValue());

    template <class TDstValue, class TSrcValue, class TSize>
    TConstPolymorphicValuesSparseArray<TDstValue, TSize> MakeConstPolymorphicValuesSparseArrayWithArrayIndex(
        TSize size,
        TMaybeOwningConstArrayHolder<TSize> indices,
        TMaybeOwningConstArrayHolder<TSrcValue> nonDefaultValues,
        bool ordered = false,
        TDstValue defaultValue = TDstValue());


    template <class TValue, class TSize>
    using TSparseCompressedArray = TSparseArrayBase<TValue, TCompressedArray, TSize>;
}

#include "sparse_array-inl.h"

