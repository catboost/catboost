#include <catboost/libs/helpers/dynamic_iterator.h>

#include <util/generic/algorithm.h>
#include <util/generic/array_size.h>
#include <util/generic/xrange.h>
#include <util/system/types.h>

#include <library/unittest/registar.h>


using namespace NCB;


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
}
