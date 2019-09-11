#pragma once

#include <catboost/libs/index_range/index_range.h>

#include <util/generic/algorithm.h>
#include <util/generic/array_ref.h>
#include <util/generic/maybe.h>
#include <util/generic/ptr.h>
#include <util/generic/utility.h>
#include <util/generic/ylimits.h>
#include <util/system/compiler.h>
#include <util/system/yassert.h>

#include <iterator>
#include <type_traits>
#include <utility>


namespace NCB {

    // helper class for specifyng End VALUE constant
    template <class TReturnValue>
    struct IDynamicIteratorEnd;

    template <class TValue>
    struct IDynamicIteratorEnd<TMaybe<TValue>> {
        constexpr static TNothing VALUE = Nothing();
    };

    template <class TValue>
    struct IDynamicIteratorEnd<TValue*> {
        constexpr static TValue* VALUE = nullptr;
    };

    template <class TValue>
    struct IDynamicIteratorEnd<TConstArrayRef<TValue>> {
        constexpr static TConstArrayRef<TValue> VALUE = TConstArrayRef<TValue>();
    };


    /*
     * Use TReturnValue = 'TMaybe<TValue>' for const iteration over light or dynamically generated objects
     * Use TReturnValue = 'const TValue*'/'TValue*' for iteration objects stored somewhere
     *  Use if TValue is either heavy for copying or mutability of objects iterated over is desired
     *
     *  Other useful case is TConstArrayRef<T> for both TValue and TReturnValue, empty array will indicate
     *    the end of iteration
     *
     * Both 'TMaybe<TValue>' and 'const TValue*' support
     *  checking for end of iteration with 'if (returnedValue)' and
     *  getting value by '*returnedValue' so they will work with generic code
     *
     *  Is not intended to be shared, so use with THolder
     *   but STL LegacyIterator requires CopyConstructible and CopyAssignable - use with TIntrusivePtr
     *   in this case.
     */
    template <class TValue, class TReturnValue = TMaybe<TValue>>
    struct IDynamicIterator : public TThrRefBase {
    public:
        using value_type = TValue;
        using return_value_type = TReturnValue;

    public:
        constexpr static auto END_VALUE = IDynamicIteratorEnd<TReturnValue>::VALUE;

    public:
        virtual ~IDynamicIterator() = default;

        // returns END_VALUE if exhausted
        virtual TReturnValue Next() = 0;
    };

    template <class TValue, class TReturnValue = TMaybe<TValue>>
    using IDynamicIteratorPtr = THolder<IDynamicIterator<TValue, TReturnValue>>;


    template <class TValue, class TReturnValue = TMaybe<TValue>>
    bool AreSequencesEqual(
        IDynamicIteratorPtr<TValue, TReturnValue> lhs,
        IDynamicIteratorPtr<TValue, TReturnValue> rhs) {

        while (true) {
            auto lNext = lhs->Next();
            auto rNext = rhs->Next();

            if (!lNext) {
                return !rNext;
            }
            if (!rNext) {
                return false;
            }
            if (*lNext != *rNext) {
                return false;
            }
        }
        Y_UNREACHABLE();
    }


    template <class TValue, class TReturnValue = TMaybe<TValue>>
    class TDynamicIteratorAsStatic {
    public:
        using iterator_category = std::input_iterator_tag;
        using value_type = TValue;
        using difference_type = size_t;
        using pointer = TReturnValue;
        using reference = TValue&;

        using IBaseIterator = IDynamicIterator<TValue, TReturnValue>;

    public:
        // use as end() sentinel
        TDynamicIteratorAsStatic() = default;

        // baseIteratorPtr must be non-nullptr
        explicit TDynamicIteratorAsStatic(THolder<IBaseIterator> baseIteratorPtr)
            : Current(baseIteratorPtr->Next())
            , BaseIteratorPtr(baseIteratorPtr.Release())
        {}

        reference operator*() {
            return *Current;
        }

        TDynamicIteratorAsStatic& operator++() {
            Current = BaseIteratorPtr->Next();
            return *this;
        }

        // this iterator is non-copyable so return TReturnValue to make '*it++' idiom work
        TReturnValue operator++(int) {
            TReturnValue result = std::move(Current);
            Current = BaseIteratorPtr->Next();
            return result;
        }

        // the only valid comparison is comparison with end() sentinel
        bool operator==(const TDynamicIteratorAsStatic& rhs) const {
            Y_ASSERT(!rhs.BaseIteratorPtr);
            return Current == IBaseIterator::END_VALUE;
        }

        // the only valid comparison is comparison with end() sentinel
        bool operator!=(const TDynamicIteratorAsStatic& rhs) const {
            Y_ASSERT(!rhs.BaseIteratorPtr);
            return Current != IBaseIterator::END_VALUE;
        }

    private:
        TReturnValue Current;
        TIntrusivePtr<IBaseIterator> BaseIteratorPtr; // TIntrusivePtr for copyability
    };


    template <
        class TBaseIterator,
        class TReturnValue = TMaybe<typename std::iterator_traits<TBaseIterator>::value_type>>
    class TStaticIteratorRangeAsDynamic final
        : public IDynamicIterator<typename std::iterator_traits<TBaseIterator>::value_type, TReturnValue>
    {
        using IBase = IDynamicIterator<typename std::iterator_traits<TBaseIterator>::value_type, TReturnValue>;

    public:
        TStaticIteratorRangeAsDynamic(TBaseIterator begin, TBaseIterator end)
            : Current(std::move(begin))
            , End(std::move(end))
        {}

        template <class TContainer>
        explicit TStaticIteratorRangeAsDynamic(TContainer& container)
            : TStaticIteratorRangeAsDynamic(container.begin(), container.end())
        {}

        TReturnValue Next() override {
            if (Current == End) {
                return IBase::END_VALUE;
            }
            if constexpr(std::is_same<TReturnValue, TMaybe<typename IBase::value_type>>()) {
                return *Current++;
            } else {
                return &*(Current++);
            }
        }

    private:
        TBaseIterator Current;
        TBaseIterator End;
    };


    template <class TSize>
    class TRangeIterator final : public IDynamicIterator<TSize> {
    public:
        explicit TRangeIterator(TIndexRange<TSize> range)
            : Current(range.Begin)
            , End(range.End)
        {}

        TMaybe<TSize> Next() override {
            if (Current == End) {
                return IDynamicIterator<TSize>::END_VALUE;
            }
            return Current++;
        }

    private:
        TSize Current;
        const TSize End;
    };

    /*
     * Iterates over (possibly sparse) array returning only non-default (index, value) pairs
     * Mutability of values is not supported yet, but could be implemented
     */
    template <class TReturnValue, class TIndex = size_t>
    using IDynamicSparseIterator = IDynamicIterator<std::pair<TIndex, TReturnValue>>;

    template <class TReturnValue, class TIndex = size_t>
    using IDynamicSparseIteratorPtr = THolder<IDynamicSparseIterator<TReturnValue, TIndex>>;


    template <class TValue, class TIndex = size_t>
    class TDynamicIteratorAsSparseDynamic final
        : public IDynamicSparseIterator<TValue, TIndex> {
    public:
        using IBase = IDynamicSparseIterator<TValue, TIndex>;

    public:
        explicit TDynamicIteratorAsSparseDynamic(IDynamicIteratorPtr<TValue> valueIterator, TIndex offset = 0)
            : ValueIterator(std::move(valueIterator))
            , Index(offset)
        {}

        TMaybe<std::pair<TIndex, TValue>> Next() override {
            const TMaybe<TValue> nextValue = ValueIterator->Next();
            if (nextValue) {
                return MakeMaybe(std::pair<TIndex, TValue>(Index++, *nextValue));
            } else {
                return IBase::END_VALUE;
            }
        }

    private:
        IDynamicIteratorPtr<TValue> ValueIterator;
        TIndex Index;
    };


    template <class TBaseIterator, class TIndex = size_t>
    class TStaticIteratorRangeAsSparseDynamic final
        : public IDynamicSparseIterator<typename std::iterator_traits<TBaseIterator>::value_type, TIndex> {
    public:
        using TValue = typename std::iterator_traits<TBaseIterator>::value_type;
        using IBase = IDynamicSparseIterator<TValue, TIndex>;

    public:
        TStaticIteratorRangeAsSparseDynamic(TBaseIterator begin, TBaseIterator end)
            : Begin(std::move(begin))
            , End(std::move(end))
            , Index(0)
        {}

        template <class TContainer>
        explicit TStaticIteratorRangeAsSparseDynamic(const TContainer& container)
            : TStaticIteratorRangeAsSparseDynamic(container.begin(), container.end())
        {}

        TMaybe<std::pair<TIndex, TValue>> Next() override {
            if (Begin == End) {
                return IBase::END_VALUE;
            }
            return MakeMaybe(std::pair<TIndex, TValue>(Index++, *Begin++));
        }

    private:
        TBaseIterator Begin;
        TBaseIterator End;
        TIndex Index;
    };


    template <class T>
    struct IDynamicBlockIterator {
    public:
        using value_type = T;

    public:
        virtual ~IDynamicBlockIterator() = default;

        /*
         * returns array with size <= maxBlockSize
         * returns empty array if exhausted
         *
         * array contents are guaranteed to exist only until the next call to Next()
         */
        virtual TConstArrayRef<T> Next(size_t maxBlockSize = Max<size_t>()) = 0;
    };


    template <class TValue>
    using IDynamicBlockIteratorPtr = THolder<IDynamicBlockIterator<TValue>>;

    template <class TLeft, class TRight, class TComparer = std::equal_to<TLeft>>
    bool AreBlockedSequencesEqual(
        IDynamicBlockIteratorPtr<TLeft> lhs,
        IDynamicBlockIteratorPtr<TRight> rhs,
        TComparer&& comparer = TComparer(), // returns true if values are equal
        size_t maxBlockSize = Max<size_t>()) {

        TConstArrayRef<TLeft> lBlock = lhs->Next(maxBlockSize);
        TConstArrayRef<TRight> rBlock = rhs->Next(maxBlockSize);

        while (true) {
            size_t blockIntersectionSize = Min(lBlock.size(), rBlock.size());
            if (!blockIntersectionSize) {
                return lBlock.empty() && rBlock.empty();
            }

            if (!::Equal(lBlock.begin(), lBlock.begin() + blockIntersectionSize, rBlock.begin(), comparer)) {
                return false;
            }
            if (lBlock.size() == blockIntersectionSize) {
                lBlock = lhs->Next(maxBlockSize);
            } else {
                lBlock = lBlock.Slice(blockIntersectionSize);
            }
            if (rBlock.size() == blockIntersectionSize) {
                rBlock = rhs->Next(maxBlockSize);
            } else {
                rBlock = rBlock.Slice(blockIntersectionSize);
            }
        }
        Y_UNREACHABLE();
    }


    template <class T>
    struct IDynamicExactBlockIterator {
    public:
        using value_type = T;

    public:
        virtual ~IDynamicExactBlockIterator() = default;

        /*
         * returns array with size == exactBlockSize
         * caller must know that at least exactBlockSize elements are still available
         * array contents are guaranteed to exist only until the next call to Next()
         */
        virtual TConstArrayRef<T> NextExact(size_t exactBlockSize) = 0;
    };

    template <class TValue>
    using IDynamicExactBlockIteratorPtr = THolder<IDynamicExactBlockIterator<TValue>>;


    template <class T>
    struct IDynamicBlockWithExactIterator
        : public IDynamicBlockIterator<T>
        , public IDynamicExactBlockIterator<T>
    {
    public:
        using value_type = T;
    };

    template <class TValue>
    using IDynamicBlockWithExactIteratorPtr = THolder<IDynamicBlockWithExactIterator<TValue>>;


    template <class T>
    class TArrayBlockIterator final : public IDynamicBlockWithExactIterator<T> {
    public:
        TArrayBlockIterator(TConstArrayRef<T> array)
            : Current(array.data())
            , End(array.data() + array.size())
        {}

        TConstArrayRef<T> Next(size_t maxBlockSize = Max<size_t>()) override {
            return NextExact(Min(maxBlockSize, size_t(End - Current)));
        }

        TConstArrayRef<T> NextExact(size_t exactBlockSize) override {
            TConstArrayRef<T> result(Current, Current + exactBlockSize);
            Current += exactBlockSize;
            return result;
        }
    private:
        const T* Current;
        const T* const End;
    };

    template <class TDst, class TSrc>
    class TTypeCastingArrayBlockIterator final : public IDynamicBlockWithExactIterator<TDst> {
    public:
        TTypeCastingArrayBlockIterator(TConstArrayRef<TSrc> array)
            : Current(array.data())
            , End(array.data() + array.size())
        {}

        TConstArrayRef<TDst> Next(size_t maxBlockSize = Max<size_t>()) override {
            return NextExact(Min(maxBlockSize, size_t(End - Current)));
        }

        TConstArrayRef<TDst> NextExact(size_t exactBlockSize) override {
            DstBuffer.assign(Current, Current + exactBlockSize);
            Current += exactBlockSize;
            return DstBuffer;
        }
    private:
        const TSrc* Current;
        const TSrc* const End;

        TVector<TDst> DstBuffer;
    };

    template <class TDst, class TSrc, class TTransformer>
    class TTransformArrayBlockIterator final : public IDynamicBlockWithExactIterator<TDst> {
    public:
        TTransformArrayBlockIterator(TConstArrayRef<TSrc> array, TTransformer&& transformer)
            : Current(array.data())
            , End(array.data() + array.size())
            , Transformer(std::move(transformer))
        {}

        TConstArrayRef<TDst> Next(size_t maxBlockSize = Max<size_t>()) override {
            return NextExact(Min(maxBlockSize, size_t(End - Current)));
        }

        TConstArrayRef<TDst> NextExact(size_t exactBlockSize) override {
            DstBuffer.yresize(exactBlockSize);
            Transform(Current, Current + exactBlockSize, DstBuffer.begin(), Transformer);
            Current += exactBlockSize;
            return DstBuffer;
        }
    private:
        const TSrc* Current;
        const TSrc* const End;

        TVector<TDst> DstBuffer;

        TTransformer Transformer;
    };

}
