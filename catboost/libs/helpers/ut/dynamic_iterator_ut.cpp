#include <catboost/libs/helpers/dynamic_iterator.h>

#include <catboost/libs/helpers/vector_helpers.h>

#include <util/generic/algorithm.h>
#include <util/generic/array_size.h>
#include <util/generic/xrange.h>
#include <util/system/types.h>

#include <library/cpp/testing/unittest/registar.h>


using namespace NCB;

namespace {
    template <class TValue>
    class TDynamicIteratorAsStatic {
    public:
        using iterator_category = std::input_iterator_tag;
        using value_type = TValue;
        using difference_type = size_t;
        using pointer = const TValue*;
        using reference = TValue&;

        using IBaseIterator = IDynamicIterator<TValue>;

    public:
        // use as end() sentinel
        TDynamicIteratorAsStatic() = default;

        // baseIteratorPtr must be non-nullptr
        explicit TDynamicIteratorAsStatic(THolder<IBaseIterator> baseIteratorPtr)
            : Current{}
            , GotCurrent(baseIteratorPtr->Next(&Current))
            , BaseIteratorPtr(baseIteratorPtr.Release())
        {}

        reference operator*() {
            return Current;
        }

        TDynamicIteratorAsStatic& operator++() {
            GotCurrent = BaseIteratorPtr->Next(&Current);
            return *this;
        }

        // this iterator is non-copyable so return TReturnValue to make '*it++' idiom work
        TValue operator++(int) {
            TValue result = std::move(Current);
            GotCurrent = BaseIteratorPtr->Next(&Current);
            return result;
        }

        // the only valid comparison is comparison with end() sentinel
        bool operator==(const TDynamicIteratorAsStatic& rhs) const {
            Y_ASSERT(!rhs.BaseIteratorPtr);
            return GotCurrent;
        }

        // the only valid comparison is comparison with end() sentinel
        bool operator!=(const TDynamicIteratorAsStatic& rhs) const {
            Y_ASSERT(!rhs.BaseIteratorPtr);
            return !GotCurrent;
        }

    private:
        TValue Current;
        bool GotCurrent;
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

        bool Next(std::pair<TIndex, TValue>* value) override {
            if (Begin == End) {
                return false;
            }
            *value = std::pair<TIndex, TValue>(Index++, *Begin++);
            return true;
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
                UNIT_ASSERT(!iterator.Next(nullptr));
            }
            {
                int data[] = {0, 5, 10, 7};

                TIterator iterator(data, data + Y_ARRAY_SIZE(data));
                for (auto element : data) {
                    int next;
                    UNIT_ASSERT(iterator.Next(&next));
                    UNIT_ASSERT_VALUES_EQUAL(element, next);
                }
                UNIT_ASSERT(!iterator.Next(nullptr));
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
                const bool result = AreSequencesEqual<TString>(
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
                        TIterator(MakeHolder<TDynamicIterator>(data))
                    ));
            }
        }
    }

    Y_UNIT_TEST(TStaticIteratorRangeAsSparseDynamic) {
        using TIterator = TStaticIteratorRangeAsSparseDynamic<const int*>;

        {
            TIterator iterator(nullptr, nullptr);
            UNIT_ASSERT(!iterator.Next(nullptr));
        }
        {
            TVector<int> data = {0, 5, 10, 7};
            TIterator iterator(data);
            typename TIterator::value_type next;
            for (auto i : xrange(data.size())) {
                UNIT_ASSERT(iterator.Next(&next));
                UNIT_ASSERT_VALUES_EQUAL(i, next.first);
                UNIT_ASSERT_VALUES_EQUAL(data[i], next.second);
            }
            UNIT_ASSERT(!iterator.Next(nullptr));
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
