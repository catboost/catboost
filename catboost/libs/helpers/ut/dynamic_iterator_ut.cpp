#include <catboost/libs/helpers/dynamic_iterator.h>

#include <catboost/libs/helpers/vector_helpers.h>

#include <util/generic/algorithm.h>
#include <util/generic/array_size.h>
#include <util/generic/xrange.h>
#include <util/system/types.h>

#include <library/unittest/registar.h>


using namespace NCB;

namespace {
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

}


Y_UNIT_TEST_SUITE(DynamicIterator) {
    Y_UNIT_TEST(TStaticIteratorRangeAsDynamic) {
        {
            using TIterator = TStaticIteratorRangeAsDynamic<const int*>;

            {
                TIterator iterator(nullptr, nullptr);
                UNIT_ASSERT(!iterator.Next());
            }
            {
                int data[] = {0, 5, 10, 7};

                TIterator iterator(data, data + Y_ARRAY_SIZE(data));
                for (auto element : data) {
                    auto next = iterator.Next();
                    UNIT_ASSERT(next);
                    UNIT_ASSERT_VALUES_EQUAL(element, *next);
                }
                UNIT_ASSERT(!iterator.Next());
            }
        }
        {
            // iteration with element mutation
            using TIterator = TStaticIteratorRangeAsDynamic<int*, int*>;

            {
                TVector<int> data = {0, 5, 10, 7};

                TIterator iterator(data);
                while (auto next = iterator.Next()) {
                    *next += 2;
                }

                TVector<int> expectedData = {2, 7, 12, 9};
                UNIT_ASSERT_VALUES_EQUAL(data, expectedData);
            }
        }
    }

    Y_UNIT_TEST(AreSequencesEqual) {
        using TIterator = TStaticIteratorRangeAsDynamic<const TString*>;

        TVector<TVector<TString>> data = {
            {}, {"a", "bb", "ccc"}, {"a", "bb", "ccc", "d"}, {"a", "bb", "xxx"}
        };

        for (auto i : xrange(data.size())) {
            for (auto j : xrange(i, data.size())) {
                const bool result = AreSequencesEqual<TString, TMaybe<TString>>(
                    MakeHolder<TIterator>(data[i]),
                    MakeHolder<TIterator>(data[j]));
                UNIT_ASSERT_EQUAL(result, (i == j));
            }
        }

    }

    Y_UNIT_TEST(TDynamicIteratorAsStatic) {
        {
            using TDynamicIterator = TStaticIteratorRangeAsDynamic<const ui32*>;
            using TIterator = TDynamicIteratorAsStatic<ui32>;

            TVector<TVector<ui32>> dataSamples = {{}, {0, 3, 5}};

            for (const auto& data : dataSamples) {
                UNIT_ASSERT(
                    Equal(
                        data.begin(),
                        data.end(),
                        TIterator(MakeHolder<TDynamicIterator>(data)),
                        TIterator()));
            }
        }
        {
            // iteration with element mutation
            using TDynamicIterator = TStaticIteratorRangeAsDynamic<ui32*, ui32*>;
            using TIterator = TDynamicIteratorAsStatic<ui32, ui32*>;

            TVector<ui32> data = {0, 5, 10, 7};

            ForEach(
                TIterator(MakeHolder<TDynamicIterator>(data)),
                TIterator(),
                [] (auto& element) { element += 2; });

            TVector<ui32> expectedData = {2, 7, 12, 9};
            UNIT_ASSERT_VALUES_EQUAL(data, expectedData);
        }
    }

    Y_UNIT_TEST(TStaticIteratorRangeAsSparseDynamic) {
        using TIterator = TStaticIteratorRangeAsSparseDynamic<const int*>;

        {
            TIterator iterator(nullptr, nullptr);
            UNIT_ASSERT(!iterator.Next());
        }
        {
            TVector<int> data = {0, 5, 10, 7};
            TIterator iterator(data);

            for (auto i : xrange(data.size())) {
                auto next = iterator.Next();
                UNIT_ASSERT(next);
                UNIT_ASSERT_VALUES_EQUAL(i, next->first);
                UNIT_ASSERT_VALUES_EQUAL(data[i], next->second);
            }
            UNIT_ASSERT(!iterator.Next());
        }
    }

    Y_UNIT_TEST(TArrayBlockIterator) {
        using TIterator = TArrayBlockIterator<int>;

        {
            TIterator iterator{TConstArrayRef<int>()};
            UNIT_ASSERT(!iterator.Next());
        }

        {
            TVector<int> v = {0, 12, 5, 10, 11};
            TIterator iterator(v);
            UNIT_ASSERT(Equal(iterator.Next(), v));
            UNIT_ASSERT(!iterator.Next());
        }
        {
            TVector<int> v = {0, 12, 5, 10, 11, 3, 7, 18, 2, 1};
            TIterator iterator(v);
            UNIT_ASSERT(Equal(iterator.Next(3), TVector<int>{0, 12, 5}));
            UNIT_ASSERT(Equal(iterator.Next(3), TVector<int>{10, 11, 3}));
            UNIT_ASSERT(Equal(iterator.Next(3), TVector<int>{7, 18, 2}));
            UNIT_ASSERT(Equal(iterator.Next(3), TVector<int>{1}));
            UNIT_ASSERT(!iterator.Next());
        }
        {
            TVector<int> v = {0, 12, 5, 10, 11, 3, 7, 18, 2, 1};
            TIterator iterator(v);
            UNIT_ASSERT(Equal(iterator.Next(4), TVector<int>{0, 12, 5, 10}));
            UNIT_ASSERT(Equal(iterator.Next(1), TVector<int>{11}));
            UNIT_ASSERT(Equal(iterator.Next(100), TVector<int>{3, 7, 18, 2, 1}));
            UNIT_ASSERT(!iterator.Next());
        }
    }
}
