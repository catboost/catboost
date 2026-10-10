#include <library/cpp/testing/unittest/registar.h>

#include <catboost/libs/model/model.h>
#include <catboost/libs/model/static_ctr_provider.h>
#include <catboost/libs/model/cpu/evaluator.h>
#include <catboost/libs/model/cpu/quantization.h>

Y_UNIT_TEST_SUITE(TModelTreesMetadata) {

    Y_UNIT_TEST(TestPrequantizedFloatBucketsMatchRawBordersAcross254Boundary) {
        using namespace NCB::NModelEvaluation;
        constexpr ui32 rows = 259; // two full evaluator blocks and a partial block
        for (ui32 borderCount : {1u, 253u, 254u, 255u}) {
            TFullModel model;
            auto* trees = model.ModelTrees.GetMutable();
            TVector<float> borders;
            for (ui32 border = 0; border < borderCount; ++border) borders.push_back(border + 0.5f);
            trees->SetFloatFeatures({TFloatFeature(false, 0, 0, borders)});
            trees->SetCatFeatures({TCatFeature(true, 0, 1, "")});
            trees->AddOneHotFeature({0, {42}, {}});
            trees->AddBinTree({static_cast<int>(borderCount - 1), static_cast<int>(borderCount)});
            for (ui32 leaf = 0; leaf < 4; ++leaf) trees->AddLeafValue(leaf);
            model.UpdateDynamicData();
            const ui32 buckets = trees->GetEffectiveBinaryFeaturesBucketsCount();
            TCPUEvaluatorQuantizedData quantized;
            quantized.QuantizedData = NCB::TMaybeOwningArrayHolder<ui8>::CreateOwning(TVector<ui8>(rows * buckets));
            TVector<ui32> hashes(FORMULA_EVALUATION_BLOCK_SIZE);
            TVector<float> ctrs;
            const auto value = [borderCount](size_t row) { return Min<ui32>(row % 256, borderCount); };
            ComputeEvaluatorFeaturesFromPreQuantizedData(*trees, *trees->GetApplyData(), model.CtrProvider,
                [&](TFeaturePosition, size_t row) { return static_cast<ui8>(value(row)); },
                [](TFeaturePosition, size_t row) -> ui32 { return row % 3 ? 7 : 42; },
                0, rows, &quantized, hashes, ctrs);
            UNIT_ASSERT_VALUES_EQUAL(quantized.BlocksCount, 3);
            UNIT_ASSERT_VALUES_EQUAL(quantized.BlockStride, buckets * FORMULA_EVALUATION_BLOCK_SIZE);
            const auto bytes = *quantized.QuantizedData;
            for (ui32 start = 0; start < rows; start += FORMULA_EVALUATION_BLOCK_SIZE) {
                const ui32 count = Min<ui32>(FORMULA_EVALUATION_BLOCK_SIZE, rows - start);
                const ui32 blockOffset = (start / FORMULA_EVALUATION_BLOCK_SIZE) * quantized.BlockStride;
                for (ui32 bucket = 0; bucket + 1 < buckets; ++bucket) {
                    for (ui32 row = 0; row < count; ++row) {
                        ui32 expected = 0;
                        // Independent raw-border comparisons, not the packed
                        // arithmetic used by the prequantized writer.
                        for (ui32 border = bucket * MAX_VALUES_PER_BIN;
                             border < Min<ui32>(borderCount, (bucket + 1) * MAX_VALUES_PER_BIN); ++border) {
                            expected += value(start + row) > borders[border];
                        }
                        UNIT_ASSERT_VALUES_EQUAL(bytes[blockOffset + bucket * count + row], expected);
                    }
                }
                for (ui32 row = 0; row < count; ++row) {
                    UNIT_ASSERT_VALUES_EQUAL(bytes[blockOffset + (buckets - 1) * count + row],
                        (start + row) % 3 == 0);
                }
            }
            auto evaluator = CreateEvaluator(EFormulaEvaluatorType::CPU, model);
            TVector<double> prediction(rows);
            evaluator->Calc(&quantized, 0, 1, prediction);
            for (ui32 row = 0; row < rows; ++row) {
                const double expected = (value(row) > borders.back()) + 2 * (row % 3 == 0);
                UNIT_ASSERT_DOUBLES_EQUAL(prediction[row], expected, 1e-12);
            }
        }
    }

    Y_UNIT_TEST(TestOneHotCtrIndexesIncludeEstimatedBuckets) {
        const TVector<TVector<ui32>> estimatedBorderCounts = {{}, {1}, {254}, {255}, {1, 255}};
        const TVector<ui32> oneHotBucketIndexes = {1, 2, 2, 3, 4};
        for (size_t testCase = 0; testCase < estimatedBorderCounts.size(); ++testCase) {
            TFullModel model;
            auto* trees = model.ModelTrees.GetMutable();
            trees->SetFloatFeatures({TFloatFeature(false, 0, 0, {0.5f})});
            trees->SetCatFeatures({TCatFeature(true, 0, 1, ""), TCatFeature(true, 1, 2, "")});
            trees->SetTextFeatures({TTextFeature(true, 0, 3, "")});
            trees->AddOneHotFeature({0, {42}, {}});
            TVector<TEstimatedFeature> estimatedFeatures;
            for (ui32 borderCount : estimatedBorderCounts[testCase]) {
                TEstimatedFeature feature(0, estimatedFeatures.size(), EEstimatedSourceFeatureType::Text);
                for (ui32 border = 0; border < borderCount; ++border) {
                    feature.Borders.push_back(border + 0.5f);
                }
                estimatedFeatures.push_back(std::move(feature));
            }
            trees->SetEstimatedFeatures(estimatedFeatures);

            auto provider = MakeIntrusive<TStaticCtrProvider>();
            model.CtrProvider = provider;
            model.UpdateDynamicData();
            UNIT_ASSERT_VALUES_EQUAL(provider->GetFloatFeatureIndexes().at(TFloatSplit(0, 0.5f)).BinIndex, 0);
            UNIT_ASSERT_VALUES_EQUAL(provider->GetOneHotFeatureIndexes().at(TOneHotSplit(0, 42)).BinIndex,
                oneHotBucketIndexes[testCase]);

            TModelCtr ctr;
            ctr.Base.CtrType = ECtrType::Counter;
            ctr.Base.Projection.CatFeatures = {1};
            ctr.Base.Projection.OneHotFeatures = {{0, 42}};
            TCtrValueTable table;
            table.ModelCtrBase = ctr.Base;
            table.CounterDenominator = 7;
            auto indexBuilder = table.GetIndexHashBuilder(2);
            auto counts = table.AllocateBlobAndGetArrayRef<int>(2);
            const ui64 categoryHash = CalcHash(0, 99);
            counts[indexBuilder.AddIndex(CalcHash(categoryHash, 0))] = 2;
            counts[indexBuilder.AddIndex(CalcHash(categoryHash, 1))] = 6;
            provider->AddCtrCalcerData(std::move(table));

            TVector<ui8> bins = {0, 1}; // numeric feature
            for (ui32 bucket = 1; bucket < oneHotBucketIndexes[testCase]; ++bucket) {
                // Estimated values deliberately oppose the one-hot predicate.
                bins.insert(bins.end(), {1, 0});
            }
            bins.insert(bins.end(), {0, 1}); // one-hot feature
            const TVector<ui32> hashedCategories = {7, 42, 99, 99};
            TVector<float> result(2);
            provider->CalcCtrs({ctr}, bins, hashedCategories, 2, result);
            UNIT_ASSERT_DOUBLES_EQUAL(result[0], 0.25f, 1e-7);
            UNIT_ASSERT_DOUBLES_EQUAL(result[1], 0.75f, 1e-7);
        }
    }

    Y_UNIT_TEST(TestMetadataUpdate) {
        TFullModel model;
        TModelTrees* trees = model.ModelTrees.GetMutable();
        trees->SetFloatFeatures(
            {
                TFloatFeature {
                    false, 0, 0,
                    {1.f, 2.f}, // bin splits 0, 1
                    ""
                },
                TFloatFeature {
                    false, 1, 2,
                    {}, // ignored feature
                    ""
                },
                TFloatFeature {
                    false, 2, 4,
                    {0.5f}, // bin split 2
                    ""
                },
                TFloatFeature {
                    false, 3, 6,
                    {}, // ignored feature
                    ""
                }
            }
        );
        trees->SetCatFeatures(
            {
                TCatFeature {
                    false,
                    0, 1,
                    ""
                },
                TCatFeature {
                    true,
                    1, 3,
                    ""
                },
                TCatFeature {
                    true,
                    2, 5,
                    ""
                },
                TCatFeature {
                    false,
                    3, 7,
                    ""
                },
                TCatFeature {
                    true,
                    4, 8,
                    ""
                },
                TCatFeature {
                    false,
                    5, 9,
                    ""
                }
            }
        );
        trees->SetTextFeatures(
            {
                TTextFeature {
                    false,
                    0,
                    10,
                    ""
                },
                 TTextFeature {
                    true,
                    1,
                    12,
                    ""
                }
            }
        );
        trees->SetEmbeddingFeatures(
            {
                TEmbeddingFeature {
                    true,
                    0,
                    11,
                    "",
                    0
                },
                TEmbeddingFeature {
                    false,
                    1,
                    13,
                    "",
                    0
                }
            }
        );
        model.UpdateDynamicData();
        model.UpdateDynamicData();// we run update metadata to detect non zeroing of some counters
        UNIT_ASSERT_EQUAL(model.GetMinimalSufficientFloatFeaturesVectorSize(), 3);
        UNIT_ASSERT_EQUAL(model.GetMinimalSufficientCatFeaturesVectorSize(), 5);
        UNIT_ASSERT_EQUAL(model.GetUsedFloatFeaturesCount(), 2);
        UNIT_ASSERT_EQUAL(model.GetUsedCatFeaturesCount(), 3);
        UNIT_ASSERT_EQUAL(model.GetNumFloatFeatures(), 4);
        UNIT_ASSERT_EQUAL(model.GetNumCatFeatures(), 6);

        auto floatIndices = GetModelFloatFeaturesIndices(model);
        TVector<size_t> expectedFloatIndices = {0, 2, 4, 6};
        UNIT_ASSERT_EQUAL(floatIndices.size(), expectedFloatIndices.size());
        for (size_t i = 0; i != floatIndices.size(); ++i) {
            UNIT_ASSERT_EQUAL(expectedFloatIndices[i], floatIndices[i]);
        }

        auto categoryIndices = GetModelCatFeaturesIndices(model);
        TVector<size_t> expectedCategoryIndices = {1, 3, 5, 7, 8, 9};
        UNIT_ASSERT_EQUAL(categoryIndices.size(), expectedCategoryIndices.size());
        for (size_t i = 0; i != categoryIndices.size(); ++i) {
            UNIT_ASSERT_EQUAL(expectedCategoryIndices[i], categoryIndices[i]);
        }

        auto textIndices = GetModelTextFeaturesIndices(model);
        TVector<size_t> expectedTextIndices = {10, 12};
        UNIT_ASSERT_EQUAL(categoryIndices.size(), expectedCategoryIndices.size());
        for (size_t i = 0; i != textIndices.size(); ++i) {
            UNIT_ASSERT_EQUAL(expectedTextIndices[i], textIndices[i]);
        }

        auto embeddingIndices = GetModelEmbeddingFeaturesIndices(model);
        TVector<size_t> expectedEmbeddingIndices = {11, 13};
        UNIT_ASSERT_EQUAL(embeddingIndices.size(), expectedEmbeddingIndices.size());
        for (size_t i = 0; i != embeddingIndices.size(); ++i) {
            UNIT_ASSERT_EQUAL(expectedEmbeddingIndices[i], embeddingIndices[i]);
        }

        trees->DropUnusedFeatures();
        UNIT_ASSERT_EQUAL(model.GetMinimalSufficientFloatFeaturesVectorSize(), 3);
        UNIT_ASSERT_EQUAL(model.GetMinimalSufficientCatFeaturesVectorSize(), 5);
        UNIT_ASSERT_EQUAL(model.GetUsedFloatFeaturesCount(), 2);
        UNIT_ASSERT_EQUAL(model.GetUsedCatFeaturesCount(), 3);
        UNIT_ASSERT_EQUAL(model.GetNumFloatFeatures(), model.GetMinimalSufficientFloatFeaturesVectorSize());
        UNIT_ASSERT_EQUAL(model.GetNumCatFeatures(), model.GetMinimalSufficientCatFeaturesVectorSize());

        floatIndices = GetModelFloatFeaturesIndices(model);
        expectedFloatIndices = {0, 4};
        UNIT_ASSERT_EQUAL(floatIndices.size(), expectedFloatIndices.size());
        for (size_t i = 0; i != floatIndices.size(); ++i) {
            UNIT_ASSERT_EQUAL(expectedFloatIndices[i], floatIndices[i]);
        }

        categoryIndices = GetModelCatFeaturesIndices(model);
        expectedCategoryIndices = {3, 5, 8};
        UNIT_ASSERT_EQUAL(categoryIndices.size(), expectedCategoryIndices.size());
        for (size_t i = 0; i != categoryIndices.size(); ++i) {
            UNIT_ASSERT_EQUAL(expectedCategoryIndices[i], categoryIndices[i]);
        }

        textIndices = GetModelTextFeaturesIndices(model);
        expectedTextIndices = {12};
        UNIT_ASSERT_EQUAL(categoryIndices.size(), expectedCategoryIndices.size());
        for (size_t i = 0; i != textIndices.size(); ++i) {
            UNIT_ASSERT_EQUAL(expectedTextIndices[i], textIndices[i]);
        }

        embeddingIndices = GetModelEmbeddingFeaturesIndices(model);
        expectedEmbeddingIndices = {11};
        UNIT_ASSERT_EQUAL(embeddingIndices.size(), expectedEmbeddingIndices.size());
        for (size_t i = 0; i != embeddingIndices.size(); ++i) {
            UNIT_ASSERT_EQUAL(expectedEmbeddingIndices[i], embeddingIndices[i]);
        }
    }
}
