#include <catboost/libs/data/columns.h>
#include <catboost/libs/data/composite_columns.h>
#include <catboost/libs/helpers/vector_helpers.h>

#include <util/generic/is_in.h>
#include <util/generic/xrange.h>

#include <library/cpp/testing/unittest/registar.h>


using namespace NCB;


Y_UNIT_TEST_SUITE(Columns) {
    Y_UNIT_TEST(TFloatArrayValuesHolder) {
        TVector<float> v = {10.0f, 11.1f, 12.2f, 13.3f, 14.4f, 15.5f, 16.6f, 17.7f, 18.8f, 19.9f};
        TVector<float> vCopy = v;

        NCB::TArraySubsetIndexing<ui32> vSubsetIndexing( NCB::TFullSubset<ui32>{(ui32)v.size()} );

        TFloatArrayValuesHolder floatValuesHolder(
            10,
            TMaybeOwningConstArrayHolder<float>::CreateOwning(std::move(v)),
            &vSubsetIndexing
        );

        UNIT_ASSERT_EQUAL(floatValuesHolder.GetType(), EFeatureValuesType::Float);
        UNIT_ASSERT_EQUAL(floatValuesHolder.GetSize(), vCopy.size());
        UNIT_ASSERT_EQUAL(floatValuesHolder.GetId(), 10);

        TVector<bool> visitedIndices(vCopy.size(), false);
        floatValuesHolder.GetData()->ForEach(
            [&](ui32 idx, float value) {
                UNIT_ASSERT_EQUAL(vCopy[idx], value);
                UNIT_ASSERT(!visitedIndices[idx]);
                visitedIndices[idx] = true;
            }
        );

        UNIT_ASSERT(!IsIn(visitedIndices, false));
    }

    Y_UNIT_TEST(TQuantizedFloatValuesHolder) {
        const TVector<ui8> src = {
            0xDE, 0xAD, 0xBE, 0xEF, 0xAB, 0xCD, 0xEF, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07
        };

        TVector<ui64> rawData = CompressVector<ui64>(src, 8);
        auto storage = NCB::TMaybeOwningArrayHolder<ui64>::CreateOwning(std::move(rawData));

        TCompressedArray data(src.size(), 8, storage);

        TFeaturesArraySubsetIndexing subsetIndexing( TIndexedSubset<ui32>{6, 5, 2, 0, 12} );

        TQuantizedFloatValuesHolder quantizedFloatValuesHolder(9, data, &subsetIndexing);

        UNIT_ASSERT_EQUAL(quantizedFloatValuesHolder.GetType(), EFeatureValuesType::QuantizedFloat);
        UNIT_ASSERT_EQUAL(quantizedFloatValuesHolder.GetSize(), 5);
        UNIT_ASSERT_EQUAL(quantizedFloatValuesHolder.GetId(), 9);


        TVector<ui8> expectedSubset{0xEF, 0xCD, 0xBE, 0xDE, 0x06};
        TVector<bool> visitedIndices(subsetIndexing.Size(), false);

        quantizedFloatValuesHolder.ForEachBlock(
            [&](ui32 blockStartIdx, auto valuesBlock) {
                for (auto i : xrange(valuesBlock.size())) {
                    UNIT_ASSERT_EQUAL(expectedSubset[blockStartIdx + i], valuesBlock[i]);
                    UNIT_ASSERT(!visitedIndices[blockStartIdx + i]);
                    visitedIndices[blockStartIdx + i] = true;
                }
            }
        );

        UNIT_ASSERT(!IsIn(visitedIndices, false));
    }


    Y_UNIT_TEST(TQuantizedFloatPackedBinaryValuesHolder) {
        TVector<NCB::TBinaryFeaturesPack> src = {
            0b00010001, // 0
            0b01001111, // 1
            0b11101010, // 2
            0b11001110, // 3
            0b10101010, // 4
            0b11111111, // 5
            0b00000000, // 6
            0b11001011  // 7
        };

        const ui32 bitsPerKey = sizeof(TBinaryFeaturesPack) * CHAR_BIT;

        TFeaturesArraySubsetIndexing subsetIndexing( TIndexedSubset<ui32>{6, 5, 0, 2} );

        auto binaryPackColumn = MakeHolder<TBinaryPacksArrayHolder>(
            0,
            TCompressedArray(
                src.size(),
                bitsPerKey,
                CompressVector<ui64>(src, bitsPerKey)
            ),
            &subsetIndexing
        );

        NPar::TLocalExecutor localExecutor;
        localExecutor.RunAdditionalThreads(2);

        TVector<TVector<ui8>> expectedFeatureValues = {
            TVector<ui8>{0, 1, 1, 0}, // 0
            TVector<ui8>{0, 1, 0, 1}, // 1
            TVector<ui8>{0, 1, 0, 0}, // 2
            TVector<ui8>{0, 1, 0, 1}, // 3
            TVector<ui8>{0, 1, 1, 0}, // 4
            TVector<ui8>{0, 1, 0, 1}, // 5
            TVector<ui8>{0, 1, 0, 1}, // 6
            TVector<ui8>{0, 1, 0, 1}  // 7
        };


        for (auto bitIdx : xrange(ui8(8))) {
            TQuantizedFloatPackedBinaryValuesHolder valuesHolder(bitIdx, binaryPackColumn.Get(), bitIdx);

            auto values = valuesHolder.ExtractValues<ui8>(&localExecutor);
            UNIT_ASSERT(Equal<ui8>(values, expectedFeatureValues[bitIdx]));
        }
    }
}
