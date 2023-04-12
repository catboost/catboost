#include <catboost/libs/helpers/compression.h>

#include <library/cpp/binsaver/util_stream_io.h>

#include <util/generic/algorithm.h>
#include <util/generic/maybe.h>
#include <util/random/fast.h>
#include <util/stream/buffer.h>
#include <util/system/yassert.h>

#include <library/cpp/testing/unittest/registar.h>


using namespace NCB;


Y_UNIT_TEST_SUITE(Compression) {
    template <class T>
    TVector<T> GenerateRandomVector(ui32 size, ui32 bitsPerKey) {
        CB_ENSURE((bitsPerKey > 0) && (bitsPerKey <= CHAR_BIT * sizeof(T)));

        const ui64 upperBound = ui64(1) << bitsPerKey;

        TFastRng64 randGen(0);

        TVector<T> result;
        result.yresize(size);
        Generate(result.begin(), result.end(), [&] () { return (T)randGen.Uniform(upperBound); });
        return result;
    }

    template <class T>
    TCompressedArray CreateCompressedArray(TConstArrayRef<T> data, ui32 bitsPerKey) {
        return TCompressedArray(
            data.size(),
            bitsPerKey,
            TMaybeOwningArrayHolder<ui64>::CreateOwning(
                CompressVector<ui64>(data.data(), data.size(), bitsPerKey)
            )
        );
    }

    template <class T>
    void TestEqualityComparisonOnGenerated() {
        TVector<ui32> bitsPerKeyForTest = {1, 2, 4, 8, 12, 16, 32};

        TVector<TCompressedArray> compressedArraysForDifferentSizes;
        for (const auto size : {0, 5, 100, 316}) {
            TMaybe<TCompressedArray> compressedArrayForThisSize;

            for (auto i : xrange(bitsPerKeyForTest.size())) {
                if (bitsPerKeyForTest[i] > CHAR_BIT * sizeof(T)) {
                    continue;
                }
                const TVector<T> generatedData = GenerateRandomVector<T>(size, bitsPerKeyForTest[i]);

                TVector<TCompressedArray> compressedArraysForDifferentBitsPerKey;

                for (auto j : xrange(i, bitsPerKeyForTest.size())) {
                    TCompressedArray compressedArray
                        = CreateCompressedArray<T>(generatedData, bitsPerKeyForTest[j]);

                    for (const auto& prevCompressedArray : compressedArraysForDifferentBitsPerKey) {
                        UNIT_ASSERT(prevCompressedArray.EqualTo(compressedArray, false));
                        UNIT_ASSERT(compressedArray.EqualTo(prevCompressedArray, false));
                    }

                    for (const auto& prevCompressedArray : compressedArraysForDifferentSizes) {
                        UNIT_ASSERT(!prevCompressedArray.EqualTo(compressedArray, false));
                        UNIT_ASSERT(!compressedArray.EqualTo(prevCompressedArray, false));
                    }

                    UNIT_ASSERT(compressedArray.EqualTo(compressedArray, true));
                    UNIT_ASSERT_EQUAL(compressedArray, compressedArray);
                    UNIT_ASSERT_EQUAL(compressedArray, TConstArrayRef<T>(generatedData));

                    compressedArraysForDifferentBitsPerKey.push_back(compressedArray);
                }
                if (!compressedArrayForThisSize) {
                    compressedArrayForThisSize = std::move(compressedArraysForDifferentBitsPerKey.back());
                }
            }

            compressedArraysForDifferentSizes.push_back(*compressedArrayForThisSize);
        }
    };

    Y_UNIT_TEST(EqualityComparison) {
        TestEqualityComparisonOnGenerated<ui8>();
        TestEqualityComparisonOnGenerated<ui16>();
        TestEqualityComparisonOnGenerated<ui32>();
    }

    void TestSaveAndLoad(TCompressedArray&& data) {
        TBuffer buffer;

        {
            TBufferOutput out(buffer);
            SerializeToArcadiaStream(out, data);
        }

        TCompressedArray loadedData;

        {
            TBufferInput in(buffer);
            SerializeFromStream(in, loadedData);
        }

        UNIT_ASSERT_EQUAL(data, loadedData);
    }

    Y_UNIT_TEST(TestBinSaverSerialization) {
        TVector<ui32> bitsPerKeyForTest = {1, 2, 4, 8, 12, 16, 32};
        for (const auto size : {0, 5, 100, 316}) {
            for (auto bitsPerKey : bitsPerKeyForTest) {
                const TVector<ui32> generatedData = GenerateRandomVector<ui32>(size, bitsPerKey);
                TestSaveAndLoad(CreateCompressedArray<ui32>(generatedData, bitsPerKey));
            }
        }
    }
}
