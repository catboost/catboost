#include <catboost/libs/data_new/columns.h>

#include <util/generic/is_in.h>

#include <library/unittest/registar.h>


using namespace NCB;


Y_UNIT_TEST_SUITE(Columns) {
    Y_UNIT_TEST(TFloatValuesHolder) {
        TVector<float> v = {10.0f, 11.1f, 12.2f, 13.3f, 14.4f, 15.5f, 16.6f, 17.7f, 18.8f, 19.9f};
        TVector<float> vCopy = v;

        NCB::TArraySubsetIndexing<size_t> vSubsetIndexing( NCB::TFullSubset<size_t>{v.size()} );

        TFloatValuesHolder floatValuesHolder(
            10,
            TMaybeOwningArrayHolder<float>::CreateOwning(std::move(v)),
            &vSubsetIndexing
        );

        UNIT_ASSERT_EQUAL(floatValuesHolder.GetType(), EFeatureValuesType::Float);
        UNIT_ASSERT_EQUAL(floatValuesHolder.GetSize(), vCopy.size());
        UNIT_ASSERT_EQUAL(floatValuesHolder.GetId(), 10);

        TVector<bool> visitedIndices(vCopy.size(), false);
        floatValuesHolder.GetArrayData().ForEach(
            [&](ui64 idx, float value) {
                UNIT_ASSERT_EQUAL(vCopy[idx], value);
                UNIT_ASSERT(!visitedIndices[idx]);
                visitedIndices[idx] = true;
            }
        );

        UNIT_ASSERT(!IsIn(visitedIndices, false));
    }

    Y_UNIT_TEST(TQuantizedFloatValuesHolder) {
        TVector<ui8> src = {
            0xDE, 0xAD, 0xBE, 0xEF, 0xAB, 0xCD, 0xEF, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07
        };

        TVector<ui64> rawData = CompressVector<ui64>(src, 8);
        auto storage = NCB::TMaybeOwningArrayHolder<ui64>::CreateOwning(std::move(rawData));

        TCompressedArray data(10, 8, storage);

        TFeaturesArraySubsetIndexing subsetIndexing( TIndexedSubset<size_t>{6, 5, 2, 0, 12} );

        TQuantizedFloatValuesHolder quantizedFloatValuesHolder(9, data, &subsetIndexing);

        UNIT_ASSERT_EQUAL(quantizedFloatValuesHolder.GetType(), EFeatureValuesType::QuantizedFloat);
        UNIT_ASSERT_EQUAL(quantizedFloatValuesHolder.GetSize(), 5);
        UNIT_ASSERT_EQUAL(quantizedFloatValuesHolder.GetId(), 9);


        TVector<ui8> expectedSubset{0xEF, 0xCD, 0xBE, 0xDE, 0x06};
        TVector<bool> visitedIndices(subsetIndexing.Size(), false);

        quantizedFloatValuesHolder.GetArrayData().ForEach(
            [&](ui64 idx, ui8 value) {
                UNIT_ASSERT_EQUAL(expectedSubset[idx], value);
                UNIT_ASSERT(!visitedIndices[idx]);
                visitedIndices[idx] = true;
            }
        );

        UNIT_ASSERT(!IsIn(visitedIndices, false));
    }

}
