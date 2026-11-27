#include <catboost/libs/model/cpu/leaf_add_simd.h>
#include <library/cpp/testing/unittest/registar.h>

#include <util/generic/vector.h>
#include <cstring>
#include <type_traits>

using namespace NCB::NModelEvaluation::NDetail;

namespace {
    template <typename TIndex>
    void CheckLeafAdds(TAddLeafFunction<TIndex> single) {
        if (!single) {
            Cerr << "Leaf-add kernel unavailable on this compiler/CPU/OS" << Endl;
            return;
        }
        const size_t leafCount = std::is_same_v<TIndex, ui8> ? 256 : 4096;
        for (size_t count : {0, 1, 3, 4, 5, 7, 8, 9, 63, 64, 65, 127, 128, 129, 257}) {
            TVector<TIndex> indexes(count + 1);
            for (size_t doc = 0; doc < count; ++doc) {
                indexes[doc + 1] = (doc * 509 + 255) % leafCount;
            }
            for (size_t dimensions : {1}) {
                TVector<double> leaves(leafCount * dimensions);
                for (size_t index = 0; index < leaves.size(); ++index) {
                    leaves[index] = (double((index * 73) % 1021) - 511.0) / 64.0;
                }
                TVector<double> actual(count * dimensions + 2, -98765.0);
                for (size_t index = 0; index < count * dimensions; ++index) {
                    actual[index + 1] = double(index) / 16.0;
                }
                auto expected = actual;
                for (size_t doc = 0; doc < count; ++doc) {
                    for (size_t dimension = 0; dimension < dimensions; ++dimension) {
                        expected[1 + doc * dimensions + dimension] += leaves[size_t(indexes[doc + 1]) * dimensions + dimension];
                    }
                }
                single(count, leaves.data(), indexes.data() + 1, actual.data() + 1);
                UNIT_ASSERT_VALUES_EQUAL(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(double)), 0);
                if (dimensions == 1) {
                    for (size_t doc = 0; doc < count; ++doc) {
                        actual[doc + 1] = double(doc) / 16.0;
                    }
                    single(count, leaves.data(), indexes.data() + 1, actual.data() + 1);
                    UNIT_ASSERT_VALUES_EQUAL(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(double)), 0);
                }
            }
        }
    }
} // namespace
Y_UNIT_TEST_SUITE(TLeafAddSimd) {
    Y_UNIT_TEST(Avx2ByteIndexes) {
        CheckLeafAdds(GetAddLeaf8Avx2());
    }
    Y_UNIT_TEST(Avx2DwordIndexes) {
        CheckLeafAdds(GetAddLeaf32Avx2());
    }
    Y_UNIT_TEST(Avx512ByteIndexes) {
        CheckLeafAdds(GetAddLeaf8Avx512());
    }
    Y_UNIT_TEST(Avx512DwordIndexes) {
        CheckLeafAdds(GetAddLeaf32Avx512());
    }
} // Y_UNIT_TEST_SUITE(TLeafAddSimd)
