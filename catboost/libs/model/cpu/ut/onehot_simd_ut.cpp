#include <catboost/libs/model/cpu/onehot_simd.h>
#include <catboost/libs/model/model.h>
#include <library/cpp/testing/unittest/registar.h>

using namespace NCB::NModelEvaluation::NDetail;

namespace {
    void CheckOneHot(TOneHotFunction kernel) {
        if (!kernel) {
            Cerr << "One-hot kernel unavailable on this compiler/CPU/OS" << Endl;
            return;
        }
        for (size_t count : {0, 1, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129}) {
            TVector<ui32> hashes(count + 1);
            for (size_t doc = 0; doc < count; ++doc) {
                hashes[doc + 1] = ui32(int((doc * 73) % 257) - 128);
            }
            for (size_t valueCount : {0, 1, 2, 16, 128, 254, 255, 256, 508, 509}) {
                TVector<int> values(valueCount);
                for (size_t value = 0; value < valueCount; ++value) {
                    values[value] = int((value * 37) % 257) - 128;
                }
                const size_t blocks = (valueCount + MAX_VALUES_PER_BIN - 1) / MAX_VALUES_PER_BIN;
                TVector<ui8> actual(count * blocks + 2, 37);
                auto expected = actual;
                for (size_t block = 0; block < blocks; ++block) {
                    for (size_t doc = 0; doc < count; ++doc) {
                        for (size_t value = block * MAX_VALUES_PER_BIN; value < Min((block + 1) * MAX_VALUES_PER_BIN, valueCount); ++value) {
                            expected[1 + block * count + doc] |= ui8(hashes[doc + 1] == ui32(values[value])) * (value - block * MAX_VALUES_PER_BIN + 1);
                        }
                    }
                }
                kernel(hashes.data() + 1, count, values.data(), valueCount, actual.data() + 1);
                UNIT_ASSERT_VALUES_EQUAL(actual, expected);
            }
        }
    }
} // namespace
Y_UNIT_TEST_SUITE(TOneHotSimd) {
    Y_UNIT_TEST(Avx2MatchesScalarIncludingDuplicateValues) {
        CheckOneHot(GetOneHotAvx2());
    }
    Y_UNIT_TEST(Avx512MatchesScalarIncludingDuplicateValues) {
        CheckOneHot(GetOneHotAvx512());
    }
} // Y_UNIT_TEST_SUITE(TOneHotSimd)
