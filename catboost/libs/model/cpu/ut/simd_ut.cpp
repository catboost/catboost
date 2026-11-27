#include <catboost/libs/model/cpu/simd.h>
#include <catboost/libs/model/cpu/quantization.h>

#include <library/cpp/testing/unittest/registar.h>

#include <util/generic/vector.h>

#include <cmath>
#include <cstring>
#include <limits>

using namespace NCB::NModelEvaluation;
using namespace NCB::NModelEvaluation::NDetail;

namespace {
    template <typename TIndex>
    void CheckIndexes(TTreeIndexesFunction<TIndex> kernel, int maxDepth) {
        if (!kernel) {
            Cerr << "Index kernel unavailable on this compiler/CPU/OS" << Endl;
            return;
        }
        for (int depth : {0, 1, 3, 7, 8, 9, 16, 31}) {
            if (depth > maxDepth) {
                continue;
            }
            for (size_t count = 0; count <= 129; ++count) {
                for (bool useXor : {false, true}) {
                    TVector<ui8> features(17 * count + 1);
                    for (size_t index = 0; index < 17 * count; ++index) {
                        features[index + 1] = (index * 73 + index / 7 + 255) & 255;
                    }
                    TVector<TRepackedBin> splits(depth);
                    for (int level = 0; level < depth; ++level) {
                        splits[level] = {ui16((level * 7) % 17), ui8((level * 53 + 128) & 255), ui8((level * 71) & 255)};
                    }
                    TVector<TIndex> actual(count + 2, TIndex(123));
                    for (size_t doc = 0; doc < count; ++doc) {
                        actual[doc + 1] = 0;
                    }
                    kernel(features.data() + 1, count, actual.data() + 1, splits.data(), depth, useXor);
                    UNIT_ASSERT_VALUES_EQUAL(actual.front(), TIndex(123));
                    UNIT_ASSERT_VALUES_EQUAL(actual.back(), TIndex(123));
                    for (size_t doc = 0; doc < count; ++doc) {
                        ui32 expected = 0;
                        for (int level = 0; level < depth; ++level) {
                            ui8 value = features[1 + splits[level].FeatureIndex * count + doc];
                            if (useXor) {
                                value ^= splits[level].XorMask;
                            }
                            expected |= ui32(value >= splits[level].SplitIdx) << level;
                        }
                        UNIT_ASSERT_VALUES_EQUAL(actual[doc + 1], TIndex(expected));
                    }
                }
            }
        }
    }

    void CheckBinarize(TBinarizeFunction kernel) {
        if (!kernel) {
            Cerr << "Binarization kernel unavailable on this compiler/CPU/OS" << Endl;
            return;
        }
        for (size_t count : {0, 1, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129}) {
            TVector<float> values(count + 1);
            for (size_t doc = 0; doc < count; ++doc) {
                switch (doc % 9) {
                    case 0:
                        values[doc + 1] = std::numeric_limits<float>::quiet_NaN();
                        break;
                    case 1:
                        values[doc + 1] = std::numeric_limits<float>::infinity();
                        break;
                    case 2:
                        values[doc + 1] = -std::numeric_limits<float>::infinity();
                        break;
                    case 3:
                        values[doc + 1] = -0.0f;
                        break;
                    default:
                        values[doc + 1] = float((doc * 73) % 509) / 2.0f - 100.0f;
                }
            }
            for (size_t borderCount : {0, 1, 2, 15, 16, 32, 127, 254, 255, 256, 508, 509}) {
                TVector<float> borders(borderCount);
                for (size_t border = 0; border < borderCount; ++border) {
                    borders[border] = float(border) / 2.0f - 100.0f;
                }
                for (float replacement : {-std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()}) {
                    for (bool substitute : {false, true}) {
                        const size_t blocks = (borderCount + MAX_VALUES_PER_BIN - 1) / MAX_VALUES_PER_BIN;
                        TVector<ui8> actual(blocks * count + 2, 37);
                        auto expected = actual;
                        for (size_t block = 0; block < blocks; ++block) {
                            for (size_t doc = 0; doc < count; ++doc) {
                                float value = values[doc + 1];
                                if (substitute && std::isnan(value)) {
                                    value = replacement;
                                }
                                ui8 sum = 0;
                                for (size_t border = block * MAX_VALUES_PER_BIN;
                                     border < Min((block + 1) * MAX_VALUES_PER_BIN, borderCount); ++border) {
                                    sum += value > borders[border];
                                }
                                expected[1 + block * count + doc] = sum + (doc < count / 16 * 16 ? 0 : 37);
                            }
                        }
                        kernel(values.data() + 1, count, borders.data(), borderCount, substitute, replacement, actual.data() + 1);
                        UNIT_ASSERT_VALUES_EQUAL(actual, expected);
                    }
                }
            }
        }
    }

    void CheckConstantBinarize(TBinarizeFunction kernel) {
        if (!kernel) {
            Cerr << "Constant binarization unavailable on this compiler/CPU/OS" << Endl;
            return;
        }
        const float samples[] = {-7.5f, 7.5f, -0.0f, 0.0f,
                                 std::numeric_limits<float>::denorm_min(), std::numeric_limits<float>::infinity(),
                                 -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()};
        for (size_t count : {20, 63, 64, 65, 100, 127, 128, 129}) {
            for (float sample : samples) {
                for (size_t variation : {0, 1, 2}) {
                    TVector<float> values(count + 2, sample);
                    if (variation) {
                        values[variation == 1 ? count : count / 2 + 1] = sample == 7.5f ? -7.5f : 7.5f;
                    }
                    for (size_t borderCount : {0, 1, 8, 16, 32, 254, 255, 256, 508, 509}) {
                        TVector<float> borders(borderCount);
                        for (size_t b = 0; b < borderCount; ++b) {
                            borders[b] = float(b) / 8.0f - 32.0f;
                        }
                        for (bool substitute : {false, true}) {
                            for (float replacement : {-std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()}) {
                                const size_t blocks = (borderCount + MAX_VALUES_PER_BIN - 1) / MAX_VALUES_PER_BIN;
                                TVector<ui8> actual(blocks * count + 2, 37);
                                auto expected = actual;
                                for (size_t block = 0; block < blocks; ++block) {
                                    for (size_t d = 0; d < count; ++d) {
                                        float value = values[d + 1];
                                        if (substitute && std::isnan(value)) {
                                            value = replacement;
                                        }
                                        ui8 sum = 0;
                                        for (size_t b = block * MAX_VALUES_PER_BIN; b < Min((block + 1) * MAX_VALUES_PER_BIN, borderCount); ++b) {
                                            sum += value > borders[b];
                                        }
                                        expected[1 + block * count + d] = sum + (d < count / 16 * 16 ? 0 : 37);
                                    }
                                }
                                kernel(values.data() + 1, count, borders.data(), borderCount, substitute, replacement, actual.data() + 1);
                                UNIT_ASSERT_VALUES_EQUAL(actual, expected);
                            }
                        }
                    }
                }
            }
        }
    }

    void CheckFloatFeatureGroups(TBinarizeFunction kernel) {
        if (!kernel) {
            Cerr << "Grouped float binarization unavailable on this compiler/CPU/OS" << Endl;
            return;
        }
        for (size_t featureCount : {0, 1, 3, 4, 5, 7, 8, 9, 14}) {
            TVector<TFloatFeature> features;
            const size_t flatCount = featureCount * 2 + 3;
            TFeatureLayout layout;
            TVector<ui32> mapping;
            for (size_t flat = 0; flat < flatCount; ++flat) {
                mapping.push_back(flatCount - 1 - flat);
            }
            layout.FlatIndexes = std::move(mapping);
            const size_t borderCounts[] = {1, 32, 254, 255, 509};
            for (size_t feature = 0; feature < featureCount; ++feature) {
                TVector<float> borders(feature % 4 == 1 ? 0 : borderCounts[feature % 5]);
                for (size_t border = 0; border < borders.size(); ++border) {
                    borders[border] = float(border) / 8.0f - 32.0f;
                }
                features.emplace_back(feature % 2 == 0, feature, feature * 2 + 1, borders);
                features.back().NanValueTreatment = static_cast<TFloatFeature::ENanValueTreatment>(feature % 3);
            }
            for (size_t count : {1, 8, 63, 64, 65, 100, 127, 128}) {
                constexpr size_t start = 3;
                const size_t totalDocs = start + count + 2;
                for (bool remap : {false, true}) {
                    for (bool transposed : {false, true}) {
                        TVector<float> data(flatCount * totalDocs);
                        for (size_t doc = 0; doc < totalDocs; ++doc) {
                            for (size_t flat = 0; flat < flatCount; ++flat) {
                                float value = float((doc * 73 + flat * 17) % 509) / 4.0f - 40.0f;
                                if ((doc + flat) % 7 == 0) {
                                    value = std::numeric_limits<float>::quiet_NaN();
                                } else if ((doc + flat) % 11 == 0) {
                                    value = std::numeric_limits<float>::infinity();
                                }
                                data[transposed ? flat * totalDocs + doc : doc * flatCount + flat] = value;
                            }
                        }
                        auto accessor = [&](TFeaturePosition position, size_t doc) {
                            return data[transposed ? position.FlatIndex * totalDocs + doc : doc * flatCount + position.FlatIndex];
                        };
                        size_t width = 0;
                        for (const auto& feature : features) {
                            width += count * ((feature.Borders.size() + MAX_VALUES_PER_BIN - 1) / MAX_VALUES_PER_BIN);
                        }
                        TVector<ui8> actual(width + 2, 37), expected(actual);
                        size_t offset = 1;
                        for (const auto& feature : features) {
                            const auto position = remap ? layout.GetRemappedPosition(feature) : feature.Position;
                            for (size_t block = 0; block < feature.Borders.size(); block += MAX_VALUES_PER_BIN) {
                                for (size_t doc = 0; doc < count; ++doc) {
                                    float value = accessor(position, start + doc);
                                    if (std::isnan(value) && feature.HasNans && feature.NanValueTreatment != TFloatFeature::ENanValueTreatment::AsIs) {
                                        const float infinity = std::numeric_limits<float>::infinity();
                                        value = feature.NanValueTreatment == TFloatFeature::ENanValueTreatment::AsTrue ? infinity : -infinity;
                                    }
                                    ui8 sum = 0;
                                    for (size_t border = block; border < Min(block + MAX_VALUES_PER_BIN, feature.Borders.size()); ++border) {
                                        sum += value > feature.Borders[border];
                                    }
                                    expected[offset + doc] = sum + (doc < count / 16 * 16 ? 0 : 37);
                                }
                                offset += count;
                            }
                        }
                        auto* result = actual.data() + 1;
                        BinarizeFloatFeaturesWide(features, remap ? &layout : nullptr, count, accessor, start, result, kernel);
                        UNIT_ASSERT_VALUES_EQUAL(result - actual.data(), width + 1);
                        UNIT_ASSERT_VALUES_EQUAL(actual, expected);
                    }
                }
            }
        }
    }

    void CheckScaleBias(TScaleBiasFunction kernel) {
        if (!kernel) {
            Cerr << "Scale/bias kernel unavailable on this compiler/CPU/OS" << Endl;
            return;
        }
        const double samples[] = {0.1, -0.2, 1e100, -1e100, -0.0, 0.0,
                                  std::numeric_limits<double>::denorm_min(), std::numeric_limits<double>::max(),
                                  std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(),
                                  std::numeric_limits<double>::quiet_NaN()};
        for (size_t count : {0, 1, 3, 4, 5, 7, 8, 9, 127, 128, 129, 257}) {
            for (double scale : {0.0, 0.75, 1.0, -2.0}) {
                for (double bias : {-0.0, 0.0, 0.125, 1e100}) {
                    for (bool applyBias : {false, true}) {
                        TVector<double> actual(count + 2, -98765.0);
                        for (size_t index = 0; index < count; ++index) {
                            actual[index + 1] = samples[index % std::size(samples)];
                        }
                        auto expected = actual;
                        for (size_t index = 0; index < count; ++index) {
                            if (scale != 1.0) {
                                volatile double rounded = expected[index + 1] * scale;
                                expected[index + 1] = rounded;
                            }
                            if (applyBias) {
                                expected[index + 1] += bias;
                            }
                        }
                        kernel(actual.data() + 1, count, scale, bias, applyBias);
                        for (size_t index = 0; index < actual.size(); ++index) {
                            if (std::isnan(expected[index])) {
                                UNIT_ASSERT(std::isnan(actual[index]));
                            } else {
                                UNIT_ASSERT_VALUES_EQUAL(std::memcmp(&actual[index], &expected[index], sizeof(double)), 0);
                            }
                        }
                    }
                }
            }
        }
    }
} // namespace

Y_UNIT_TEST_SUITE(TCpuSimdKernels) {
    Y_UNIT_TEST(Avx2FloatFeatureGroups) {
        CheckFloatFeatureGroups(GetBinarizeAvx2());
    }
    Y_UNIT_TEST(Avx512FloatFeatureGroups) {
        CheckFloatFeatureGroups(GetBinarizeAvx512());
    }
    Y_UNIT_TEST(Avx2Indexes) {
        CheckIndexes(GetTreeIndexes8Avx2(), 8);
        CheckIndexes(GetTreeIndexes32Avx2(), 31);
    }
    Y_UNIT_TEST(Avx512Indexes) {
        CheckIndexes(GetTreeIndexes8Avx512(), 8);
        CheckIndexes(GetTreeIndexes32Avx512(), 31);
    }
    Y_UNIT_TEST(Avx2ConstantBinarize) {
        CheckConstantBinarize(GetBinarizeAvx2());
    }
    Y_UNIT_TEST(Avx512ConstantBinarize) {
        CheckConstantBinarize(GetBinarizeAvx512());
    }
    Y_UNIT_TEST(Avx2Binarize) {
        CheckBinarize(GetBinarizeAvx2());
    }
    Y_UNIT_TEST(Avx512Binarize) {
        CheckBinarize(GetBinarizeAvx512());
    }
    Y_UNIT_TEST(Avx2ScaleBias) {
        CheckScaleBias(GetScaleBiasAvx2());
    }
    Y_UNIT_TEST(Avx512ScaleBias) {
        CheckScaleBias(GetScaleBiasAvx512());
    }
} // Y_UNIT_TEST_SUITE(TCpuSimdKernels)
