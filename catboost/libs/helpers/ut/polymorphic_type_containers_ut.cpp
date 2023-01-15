#include <catboost/libs/helpers/polymorphic_type_containers.h>

#include <catboost/libs/helpers/array_subset.h>

#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/generic/algorithm.h>
#include <util/generic/vector.h>
#include <util/generic/xrange.h>
#include <util/system/types.h>

#include <library/cpp/testing/unittest/registar.h>


using namespace NCB;


Y_UNIT_TEST_SUITE(TTypeCastArraySubset) {
    template <class TInterfaceValue, class TStoredValue>
    void TestOneCase(
        const TVector<TStoredValue>& v,
        const TArraySubsetIndexing<ui32>& arraySubsetIndexing,
        const TVector<TInterfaceValue>& expectedSubset,
        TInterfaceValue notPresentValue
    ) {
        TTypeCastArraySubset<TInterfaceValue, TStoredValue> typeCastArraySubset(
            TMaybeOwningConstArrayHolder<TStoredValue>::CreateNonOwning(v),
            &arraySubsetIndexing
        );

        const ITypedArraySubset<TInterfaceValue>& typedArraySubset = typeCastArraySubset;

        UNIT_ASSERT_EQUAL(typedArraySubset.GetSize(), expectedSubset.size());

        // ForEach
        {
            size_t expectedIndex = 0;
            typedArraySubset.ForEach(
                [&](ui32 index, TInterfaceValue value) {
                    UNIT_ASSERT_VALUES_EQUAL(expectedIndex, index);
                    UNIT_ASSERT_VALUES_EQUAL(expectedSubset[index], value);
                    ++expectedIndex;
                }
            );
            UNIT_ASSERT_VALUES_EQUAL(expectedIndex, expectedSubset.size());
        }

        // ParallelForEach
        {
            NPar::TLocalExecutor localExecutor;
            localExecutor.RunAdditionalThreads(3);

            for (ui32 approximateBlockSize : xrange(0, 12)) { // use 0 as undefined
                TVector<bool> indicesIterated(expectedSubset.size(), false);
                typedArraySubset.ParallelForEach(
                    [&](ui32 index, TInterfaceValue value) {
                        UNIT_ASSERT_VALUES_EQUAL(expectedSubset[index], value);

                        auto& indexIterated = indicesIterated.at(index);
                        UNIT_ASSERT(!indexIterated); // each index must be visited only once
                        indexIterated = true;
                    },
                    &localExecutor,
                    approximateBlockSize != 0 ? TMaybe<ui32>(approximateBlockSize) : Nothing()
                );
                UNIT_ASSERT(!IsIn(indicesIterated, false)); // each index was visited
            }
        }

        // Find
        {
            if (expectedSubset.size()) {
                TInterfaceValue presentValue = expectedSubset[expectedSubset.size() / 2];
                UNIT_ASSERT(
                    typedArraySubset.Find(
                        [=](ui32 /*idx*/, TInterfaceValue value) { return value == presentValue; }
                    )
                );
            }

            UNIT_ASSERT(
                !typedArraySubset.Find(
                    [=](ui32 /*idx*/, TInterfaceValue value) { return value == notPresentValue; }
                )
            );
        }
    }


    Y_UNIT_TEST(TFullSubset) {
        TArraySubsetIndexing<ui32> arraySubsetIndexing( TFullSubset<ui32>{10} );

        TVector<float> expectedSubset = {
            10.0f,
            11.0f,
            12.0f,
            13.0f,
            14.0f,
            15.0f,
            16.0f,
            17.0f,
            18.0f,
            19.0f
        };

        {
            TVector<float> v = expectedSubset;
            TestOneCase(v, arraySubsetIndexing, expectedSubset, 2.0f);
        }
        {
            TVector<int> v = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
            TestOneCase(v, arraySubsetIndexing, expectedSubset, 2.0f);
        }
        {
            TVector<double> v = {10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0};
            TestOneCase(v, arraySubsetIndexing, expectedSubset, 2.0f);
        }
    }

    Y_UNIT_TEST(TRangesSubset) {
        TVector<TIndexRange<ui32>> indexRanges{{7, 10}, {2, 3}, {4, 6}};

        TSavedIndexRanges<ui32> savedIndexRanges(std::move(indexRanges));

        TArraySubsetIndexing<ui32> arraySubsetIndexing( (NCB::TRangesSubset<ui32>(savedIndexRanges)) );

        TVector<ui32> expectedSubset = {17, 18, 19, 12, 14, 15};

        {
            TVector<float> v = {10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f};
            TestOneCase(v, arraySubsetIndexing, expectedSubset, ui32(1));
        }
        {
            TVector<int> v = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
            TestOneCase(v, arraySubsetIndexing, expectedSubset, ui32(1));
        }
        {
            TVector<double> v = {10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0};
            TestOneCase(v, arraySubsetIndexing, expectedSubset, ui32(1));
        }
    }

    Y_UNIT_TEST(TIndexedSubset) {
        TArraySubsetIndexing<ui32> arraySubsetIndexing( TIndexedSubset<ui32>{6, 5, 2, 0, 1} );

        TVector<ui8> expectedSubset = {16, 15, 12, 10, 11};

        {
            TVector<float> v = {10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f};
            TestOneCase(v, arraySubsetIndexing, expectedSubset, ui8(22));
        }
        {
            TVector<i32> v = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
            TestOneCase(v, arraySubsetIndexing, expectedSubset, ui8(22));
        }
        {
            TVector<double> v = {10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0};
            TestOneCase(v, arraySubsetIndexing, expectedSubset, ui8(22));
        }
    }
}

Y_UNIT_TEST_SUITE(TTypeCastArrayHolder) {
    template <class TInterfaceValue, class TStoredValue>
    void TestOneCase(
        const TVector<TStoredValue>& v,
        const TVector<TInterfaceValue>& expectedV,
        const TArraySubsetIndexing<ui32>& arraySubsetIndexing,
        const TVector<TInterfaceValue>& expectedSubset
    ) {
        ITypedSequencePtr<TInterfaceValue> typedSequencePtr
            = MakeNonOwningTypeCastArrayHolder<TInterfaceValue>(v.begin(), v.end());

        UNIT_ASSERT_VALUES_EQUAL((size_t)typedSequencePtr->GetSize(), expectedV.size());

        for (auto offset : xrange(v.size())) {
            IDynamicBlockIteratorPtr<TInterfaceValue> blockIterator = typedSequencePtr->GetBlockIterator(
                TIndexRange<ui32>(offset, typedSequencePtr->GetSize())
            );
            size_t expectedI = offset;
            while (auto block = blockIterator->Next()) {
                for (auto element : block) {
                    UNIT_ASSERT_VALUES_EQUAL(element, expectedV[expectedI++]);
                }
            }
            UNIT_ASSERT_VALUES_EQUAL(v.size(), expectedI);
        }

        // ToArray
        {
            TVector<TInterfaceValue> extractedValues;
            extractedValues.yresize(typedSequencePtr->GetSize());
            ToArray<TInterfaceValue>(*typedSequencePtr, extractedValues);
            UNIT_ASSERT_VALUES_EQUAL(extractedValues, expectedV);
        }

        UNIT_ASSERT_VALUES_EQUAL(ToVector(*typedSequencePtr), expectedV);


        TIntrusivePtr<ITypedArraySubset<TInterfaceValue>> typedArraySubsetHolder = typedSequencePtr->GetSubset(
            &arraySubsetIndexing
        );
        const ITypedArraySubset<TInterfaceValue>& typedArraySubset = *typedArraySubsetHolder;

        UNIT_ASSERT_EQUAL(typedArraySubset.GetSize(), expectedSubset.size());

        // ForEach
        {
            size_t expectedIndex = 0;
            typedArraySubset.ForEach(
                [&](ui32 index, TInterfaceValue value) {
                    UNIT_ASSERT_VALUES_EQUAL(expectedIndex, index);
                    UNIT_ASSERT_VALUES_EQUAL(expectedSubset[index], value);
                    ++expectedIndex;
                }
            );
            UNIT_ASSERT_VALUES_EQUAL(expectedIndex, expectedSubset.size());
        }
    }

    Y_UNIT_TEST(TFullSubset) {
        TArraySubsetIndexing<ui32> arraySubsetIndexing( TFullSubset<ui32>{10} );

        TVector<float> expectedV = {10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f};
        TVector<float> expectedSubset = expectedV;

        {
            TVector<float> v(expectedV.begin(), expectedV.end());
            TestOneCase(v, expectedV, arraySubsetIndexing, expectedSubset);
        }
        {
            TVector<int> v(expectedV.begin(), expectedV.end());
            TestOneCase(v, expectedV, arraySubsetIndexing, expectedSubset);
        }
        {
            TVector<double> v(expectedV.begin(), expectedV.end());
            TestOneCase(v, expectedV, arraySubsetIndexing, expectedSubset);
        }
    }

    Y_UNIT_TEST(TRangesSubset) {
        TVector<ui32> expectedV = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};

        TVector<TIndexRange<ui32>> indexRanges{{7, 10}, {2, 3}, {4, 6}};

        TSavedIndexRanges<ui32> savedIndexRanges(std::move(indexRanges));

        TArraySubsetIndexing<ui32> arraySubsetIndexing( (NCB::TRangesSubset<ui32>(savedIndexRanges)) );

        TVector<ui32> expectedSubset = {17, 18, 19, 12, 14, 15};

        {
            TVector<float> v(expectedV.begin(), expectedV.end());
            TestOneCase(v, expectedV, arraySubsetIndexing, expectedSubset);
        }
        {
            TVector<int> v(expectedV.begin(), expectedV.end());
            TestOneCase(v, expectedV, arraySubsetIndexing, expectedSubset);
        }
        {
            TVector<double> v(expectedV.begin(), expectedV.end());
            TestOneCase(v, expectedV, arraySubsetIndexing, expectedSubset);
        }
    }

    Y_UNIT_TEST(TIndexedSubset) {
        TVector<ui8> expectedV = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};

        TArraySubsetIndexing<ui32> arraySubsetIndexing( TIndexedSubset<ui32>{6, 5, 2, 0, 1} );

        TVector<ui8> expectedSubset = {16, 15, 12, 10, 11};

        {
            TVector<float> v(expectedV.begin(), expectedV.end());
            TestOneCase(v, expectedV, arraySubsetIndexing, expectedSubset);
        }
        {
            TVector<i32> v(expectedV.begin(), expectedV.end());
            TestOneCase(v, expectedV, arraySubsetIndexing, expectedSubset);
        }
        {
            TVector<double> v(expectedV.begin(), expectedV.end());
            TestOneCase(v, expectedV, arraySubsetIndexing, expectedSubset);
        }
    }
}
