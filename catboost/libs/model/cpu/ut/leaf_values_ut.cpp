#include <catboost/libs/model/cpu/simd.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/model/model_build_helper.h>
#include <catboost/libs/cat_feature/cat_feature.h>

#include <library/cpp/testing/unittest/registar.h>

#include <util/generic/vector.h>

#include <array>
#include <cmath>
#include <cstring>
#include <limits>

using namespace NCB::NModelEvaluation::NDetail;

namespace {
    void CheckKernel(TLeafValues4Function kernel) {
        if (!kernel) {
            Cerr << "SIMD kernel is unavailable on this compiler/CPU/OS" << Endl;
            return;
        }
        kernel(0, nullptr, nullptr, nullptr);
        for (size_t depth = 0; depth <= 8; ++depth) {
            const size_t leafCount = size_t(1) << depth;
            for (size_t docCount = 16; docCount <= 128; docCount += 16) {
                for (size_t offset = 0; offset < 4; ++offset) {
                    std::array<TVector<double>, 4> leaves;
                    std::array<TVector<ui8>, 4> indexStorage;
                    const double* leafPointers[4];
                    const ui8* indexPointers[4];
                    for (size_t tree = 0; tree < 4; ++tree) {
                        leaves[tree].resize(leafCount);
                        indexStorage[tree].resize(docCount + offset);
                        for (size_t leaf = 0; leaf < leafCount; ++leaf) {
                            leaves[tree][leaf] = (double(leaf) - 127.0) * (tree + 1) / 8.0;
                        }
                        for (size_t doc = 0; doc < docCount; ++doc) {
                            indexStorage[tree][offset + doc] = (doc * 73 + tree * 29 + 255) % leafCount;
                        }
                        leafPointers[tree] = leaves[tree].data();
                        indexPointers[tree] = indexStorage[tree].data() + offset;
                    }
                    TVector<double> actual(docCount + offset + 1, -98765.25);
                    for (size_t doc = 0; doc < docCount; ++doc) {
                        actual[offset + doc] = double(doc) / 16.0;
                    }
                    auto expected = actual;
                    for (size_t tree = 0; tree < 4; ++tree) {
                        for (size_t doc = 0; doc < docCount; ++doc) {
                            expected[offset + doc] += leafPointers[tree][indexPointers[tree][doc]];
                        }
                    }
                    kernel(docCount, leafPointers, indexPointers, actual.data() + offset);
                    UNIT_ASSERT_VALUES_EQUAL(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(double)), 0);
                }
            }
        }
    }

    void CheckAdditionOrder(TLeafValues4Function kernel) {
        if (!kernel) {
            return;
        }
        // Reassociation of the four trees loses the small terms in these inputs.
        const double leaves0[] = {1.0e100, -0.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()};
        const double leaves1[] = {-1.0e100, -0.0, -std::numeric_limits<double>::infinity(), 1.0};
        const double leaves2[] = {1.0, -0.0, 2.0, 2.0};
        const double leaves3[] = {0.25, -0.0, 3.0, 3.0};
        const double* leaves[] = {leaves0, leaves1, leaves2, leaves3};
        ui8 indexStorage[16];
        double actual[16];
        double expected[16];
        for (size_t doc = 0; doc < 16; ++doc) {
            indexStorage[doc] = doc % 4;
            actual[doc] = expected[doc] = -0.0;
            for (size_t tree = 0; tree < 4; ++tree) {
                expected[doc] += leaves[tree][indexStorage[doc]];
            }
        }
        const ui8* indexes[] = {indexStorage, indexStorage, indexStorage, indexStorage};
        kernel(16, leaves, indexes, actual);
        for (size_t doc = 0; doc < 16; ++doc) {
            if (std::isnan(expected[doc])) {
                UNIT_ASSERT(std::isnan(actual[doc]));
            } else {
                UNIT_ASSERT_VALUES_EQUAL(std::memcmp(actual + doc, expected + doc, sizeof(double)), 0);
            }
        }
    }

    TFullModel MakeModel(size_t depth, size_t treeCount) {
        TFullModel model;
        auto* trees = model.ModelTrees.GetMutable();
        TVector<int> splits;
        for (size_t feature = 0; feature < depth; ++feature) {
            trees->AddFloatFeature(TFloatFeature(false, feature, feature, {0.5f}, ""));
            splits.push_back(feature);
        }
        for (size_t tree = 0; tree < treeCount; ++tree) {
            trees->AddBinTree(splits);
            for (size_t leaf = 0; leaf < (size_t(1) << depth); ++leaf) {
                trees->AddLeafValue((double(leaf) - 127.0) * (tree + 1) / 8.0);
            }
        }
        model.UpdateDynamicData();
        model.SetScaleAndBias({0.75, {-2.0}});
        return model;
    }
} // namespace

Y_UNIT_TEST_SUITE(TWideLeafValues) {
    Y_UNIT_TEST(Avx2AllBlocksAndByteIndexes) {
        for (size_t distance : {0, 8, 16, 32, 64}) {
            CheckKernel(GetLeafValues4Avx2(distance));
            CheckAdditionOrder(GetLeafValues4Avx2(distance));
        }
    }

    Y_UNIT_TEST(Avx512AllBlocksAndByteIndexes) {
        for (size_t distance : {0, 8, 16, 32, 64}) {
            CheckKernel(GetLeafValues4Avx512(distance));
            CheckAdditionOrder(GetLeafValues4Avx512(distance));
        }
    }

    Y_UNIT_TEST(DispatchReturnsSupportedKernel) {
        const auto avx512 = GetLeafValues4Avx512();
        const auto avx2 = GetLeafValues4Avx2();
        UNIT_ASSERT(GetWideLeafValues4() == (avx512 ? avx512 : avx2));
    }

    Y_UNIT_TEST(PartialByteBlocksMatchKnownFloatAndCategoricalLeaves) {
        for (bool categorical : {false, true}) {
            for (size_t depth = 1; depth <= 8; ++depth) {
                constexpr size_t TreeCount = 5;
                TFullModel model;
                if (!categorical) {
                    model = MakeModel(depth, TreeCount);
                } else {
                    TVector<TCatFeature> features;
                    TVector<TModelSplit> splits;
                    for (size_t f = 0; f < depth; ++f) {
                        features.push_back({true, int(f), int(f), ToString(f)});
                        splits.emplace_back(TOneHotSplit(f, CalcCatFeatureHashInt("hit")));
                    }
                    TObliviousTreeBuilder builder({}, features, {}, {}, 1);
                    for (size_t tree = 0; tree < TreeCount; ++tree) {
                        TVector<TVector<double>> leaves(1);
                        for (size_t leaf = 0; leaf < (size_t(1) << depth); ++leaf) {
                            leaves[0].push_back((double(leaf) - 127.0) * (tree + 1) / 8.0);
                        }
                        builder.AddTree(splits, leaves);
                    }
                    builder.Build(model.ModelTrees.GetMutable());
                    model.UpdateDynamicData();
                    model.SetScaleAndBias({0.75, {-2.0}});
                }
                // Every short-block remainder, plus full blocks and wide-path tails.
                for (size_t count : {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                     16, 20, 24, 28, 30, 31, 32, 36, 40, 44, 46, 47,
                                     63, 64, 68, 76, 100, 108, 124, 126, 127, 128, 129}) {
                    TVector<TVector<float>> data(count, TVector<float>(depth));
                    TVector<TConstArrayRef<float>> refs;
                    TVector<double> expected(count);
                    for (size_t doc = 0; doc < count; ++doc) {
                        const size_t index = (doc * 73 + 255) % (size_t(1) << depth);
                        for (size_t f = 0; f < depth; ++f) {
                            const bool bit = (index >> f) & 1;
                            data[doc][f] = categorical
                                               ? ConvertCatFeatureHashToFloat(CalcCatFeatureHash(bit ? "hit" : "miss"))
                                               : float(bit);
                        }
                        refs.emplace_back(data[doc]);
                        double sum = 0;
                        for (size_t tree = 0; tree < TreeCount; ++tree) {
                            sum += (double(index) - 127.0) * (tree + 1) / 8.0;
                        }
                        expected[doc] = sum * 0.75 - 2.0;
                    }
                    TVector<double> actual(count + 2, -98765.25);
                    model.CalcFlat(refs, MakeArrayRef(actual.data() + 1, count));
                    UNIT_ASSERT_VALUES_EQUAL(actual.front(), -98765.25);
                    UNIT_ASSERT_VALUES_EQUAL(actual.back(), -98765.25);
                    UNIT_ASSERT_VALUES_EQUAL(std::memcmp(actual.data() + 1, expected.data(), count * sizeof(double)), 0);
                }
            }
        }
    }

    Y_UNIT_TEST(BatchPredictionsMatchSingleDocumentEvaluation) {
        for (size_t depth : {1, 3, 8, 9}) {
            for (size_t treeCount : {3, 4, 5, 8, 9}) {
                const auto model = MakeModel(depth, treeCount);
                for (size_t docCount : {1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257}) {
                    TVector<TVector<float>> data(docCount, TVector<float>(depth));
                    TVector<TConstArrayRef<float>> features(docCount);
                    TVector<double> expected(docCount);
                    for (size_t doc = 0; doc < docCount; ++doc) {
                        const size_t index = (doc * 73 + 255) % (size_t(1) << depth);
                        for (size_t feature = 0; feature < depth; ++feature) {
                            data[doc][feature] = (index >> feature) & 1;
                        }
                        features[doc] = data[doc];
                        model.CalcFlatSingle(features[doc], MakeArrayRef(expected.data() + doc, 1));
                    }
                    TVector<double> actual(docCount + 1, -123456.0);
                    model.CalcFlat(features, MakeArrayRef(actual.data() + 1, docCount));
                    UNIT_ASSERT_VALUES_EQUAL(actual.front(), -123456.0);
                    UNIT_ASSERT_VALUES_EQUAL(std::memcmp(actual.data() + 1, expected.data(), docCount * sizeof(double)), 0);
                    model.CalcFlat(features, 1, treeCount, MakeArrayRef(actual.data() + 1, docCount));
                    for (size_t doc = 0; doc < docCount; ++doc) {
                        model.CalcFlatSingle(features[doc], 1, treeCount, MakeArrayRef(expected.data() + doc, 1));
                    }
                    UNIT_ASSERT_VALUES_EQUAL(std::memcmp(actual.data() + 1, expected.data(), docCount * sizeof(double)), 0);
                }
            }
        }
    }
} // Y_UNIT_TEST_SUITE(TWideLeafValues)
