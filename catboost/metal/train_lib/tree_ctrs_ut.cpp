#include "tree_ctrs.h"

#include <catboost/libs/data/ut/lib/for_objects.h>
#include <catboost/libs/model/hash.h>
#include <catboost/libs/model/model_build_helper.h>
#include <catboost/libs/model/model_export/json_model_helpers.h>
#include <catboost/private/libs/options/cat_feature_options.h>

#include <library/cpp/testing/unittest/registar.h>
#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/generic/map.h>
#include <util/stream/str.h>

#include <algorithm>
#include <numeric>

using namespace NCB;

namespace {
    constexpr ui32 FirstFeature = 11;

    struct TFixture {
        TTrainingDataProvider Data;
        TVector<TVector<ui32>> Hashes;
        TVector<ui8> FloatBins;
        TVector<float> Targets;
        TVector<TFloatFeature> FloatFeatures = {TFloatFeature(false, 0, 0, {0.5f})};
        TVector<TCatFeature> CatFeatures = {
            TCatFeature(true, 0, 1, "a"), TCatFeature(true, 1, 2, "b"), TCatFeature(true, 2, 3, "onehot")};

        explicit TFixture(ui32 rows = 53) {
            // Signed category hashes, including both uint32 extremes, catch
            // accidental sign extension or truncation of compound model keys.
            const TVector<TVector<ui32>> dictionaries = {
                {0, 0x80000000u, 0xffffffffu}, {7, 23, 0x87654321u, 0xfffffffdu}, {41, 0xfffffffbu}};
            TVector<TVector<ui32>> categoryBins(3);
            Hashes.resize(3);
            for (ui32 row = 0; row < rows; ++row) {
                const ui32 bins[] = {(row / 2) % 3, (row / 3) % 4, row % 2};
                for (ui32 feature = 0; feature < 3; ++feature) {
                    categoryBins[feature].push_back(bins[feature]);
                    Hashes[feature].push_back(dictionaries[feature][bins[feature]]);
                }
                FloatBins.push_back((row / 6) % 2);
                Targets.push_back((row % 9) * 0.125f);
            }
            Data.MetaInfo.TargetType = ERawTargetType::Float;
            Data.MetaInfo.TargetCount = 1;
            Data.MetaInfo.FeaturesLayout = MakeIntrusive<TFeaturesLayout>(4, TVector<ui32>{1, 2, 3}, TVector<TString>{});
            Data.ObjectsGrouping = MakeIntrusive<TObjectsGrouping>(rows);
            TCommonObjectsData common;
            common.FeaturesLayout = Data.MetaInfo.FeaturesLayout;
            common.SubsetIndexing = MakeAtomicShared<TArraySubsetIndexing<ui32>>(TFullSubset<ui32>(rows));
            common.Order = EObjectsOrder::Ordered;
            TQuantizedObjectsData quantized;
            NDataNewUT::InitQuantizedFeatures(TVector<TVector<ui8>>{FloatBins}, common.SubsetIndexing.Get(),
                                              {0}, &quantized.FloatFeatures);
            NDataNewUT::InitQuantizedFeatures(categoryBins, common.SubsetIndexing.Get(),
                                              {1, 2, 3}, &quantized.CatFeatures);
            quantized.QuantizedFeaturesInfo = MakeIntrusive<TQuantizedFeaturesInfo>(
                *common.FeaturesLayout, TConstArrayRef<ui32>(),
                NCatboostOptions::TBinarizationOptions(EBorderSelectionType::Uniform, 1, ENanMode::Forbidden));
            quantized.QuantizedFeaturesInfo->SetBorders(TFloatFeatureIdx(0), {0.5f});
            quantized.QuantizedFeaturesInfo->SetNanMode(TFloatFeatureIdx(0), ENanMode::Forbidden);
            for (ui32 feature = 0; feature < 3; ++feature) {
                TCatFeaturePerfectHash perfectHash;
                for (ui32 bin = 0; bin < dictionaries[feature].size(); ++bin) {
                    perfectHash.Map[dictionaries[feature][bin]] = {
                        bin, static_cast<ui32>(std::count(categoryBins[feature].begin(), categoryBins[feature].end(), bin))};
                }
                quantized.QuantizedFeaturesInfo->UpdateCategoricalFeaturesPerfectHash(
                    TCatFeatureIdx(feature), std::move(perfectHash));
            }
            quantized.ExclusiveFeatureBundlesData = TExclusiveFeatureBundlesData(*common.FeaturesLayout, {});
            quantized.PackedBinaryFeaturesData = TPackedBinaryFeaturesData(
                *common.FeaturesLayout, *quantized.QuantizedFeaturesInfo, quantized.ExclusiveFeatureBundlesData, true);
            quantized.FeaturesGroupsData = TFeatureGroupsData(*common.FeaturesLayout, {});
            Data.ObjectsData = MakeIntrusive<TQuantizedObjectsDataProvider>(
                Data.ObjectsGrouping, std::move(common), std::move(quantized), true, Nothing());
            TProcessedTargetData target;
            target.Targets[""] = {MakeAtomicShared<TVector<float>>(Targets)};
            // Deliberately nonuniform: CUDA CTR histories count observations,
            // irrespective of the weights used later by the objective.
            TVector<float> weights(rows);
            for (ui32 row = 0; row < rows; ++row) weights[row] = 1 + row % 5;
            target.Weights[""] = MakeIntrusive<TWeights<float>>(std::move(weights));
            Data.TargetData = MakeIntrusive<TTargetDataProvider>(Data.ObjectsGrouping, std::move(target));
        }

        TVector<TVector<ui32>> Histories() const {
            TVector<TVector<ui32>> orders(4, TVector<ui32>(Targets.size()));
            std::iota(orders[0].begin(), orders[0].end(), 0);
            orders[1] = orders[0];
            std::reverse(orders[1].begin(), orders[1].end());
            orders[2] = orders[0];
            std::rotate(orders[2].begin(), orders[2].begin() + 17, orders[2].end());
            orders[3] = orders[0];
            std::rotate(orders[3].begin(), orders[3].begin() + 7, orders[3].end());
            return orders;
        }
    };

    NCatboostOptions::TCatBoostOptions Options(ECtrType type, ui32 targetBorders = 2) {
        NCatboostOptions::TCatBoostOptions result(ETaskType::GPU);
        result.RandomSeed = 7;
        result.CatFeatureParams->MaxTensorComplexity = 3;
        result.CatFeatureParams->OneHotMaxSize = 2;
        result.CatFeatureParams->CounterCalcMethod = ECounterCalc::SkipTest;
        result.CatFeatureParams->CtrHistoryUnit = ECtrHistoryUnit::Sample;
        result.CatFeatureParams->TargetBinarization = NCatboostOptions::TBinarizationOptions(
            EBorderSelectionType::Uniform, targetBorders, ENanMode::Forbidden);
        result.CatFeatureParams->CombinationCtrs = TVector<NCatboostOptions::TCtrDescription>{
            NCatboostOptions::TCtrDescription(type, {{0.25f, 1.0f}, {0.75f, 2.0f}},
                NCatboostOptions::TBinarizationOptions(EBorderSelectionType::Uniform, 7, ENanMode::Forbidden))};
        return result;
    }

    TModelSplit SimpleCtrSplit() {
        TModelCtrSplit result;
        result.Ctr.Base.Projection.CatFeatures = {0};
        result.Ctr.Base.CtrType = ECtrType::Borders;
        result.Ctr.PriorNum = 0.5f;
        result.Ctr.PriorDenom = 1;
        result.Border = 0.5f;
        return TModelSplit(result);
    }

    ui64 ProjectionHash(const TFeatureCombination& projection, const TVector<TVector<ui32>>& hashes,
                        TConstArrayRef<ui8> floatBins, ui32 row) {
        ui64 hash = 0;
        for (int category : projection.CatFeatures)
            hash = CalcHash(hash, static_cast<ui64>(static_cast<i32>(hashes[category][row])));
        for (const auto& split : projection.BinFeatures) {
            UNIT_ASSERT_VALUES_EQUAL(split.FloatFeature, 0);
            UNIT_ASSERT_VALUES_EQUAL(split.Split, 0.5f);
            hash = CalcHash(hash, floatBins[row] > 0);
        }
        for (const auto& split : projection.OneHotFeatures)
            hash = CalcHash(hash, hashes[split.CatFeatureIdx][row] == static_cast<ui32>(split.Value));
        return hash;
    }

    float Numerator(const TModelCtr& ctr, float target, ui32 targetBorders) {
        if (ctr.Base.CtrType == ECtrType::FloatTargetMeanValue) return target;
        ui32 bin = 0;
        for (ui32 border = 1; border <= targetBorders; ++border)
            bin += target > float(border) / float(targetBorders + 1);
        return ctr.Base.CtrType == ECtrType::Borders ? bin > ctr.TargetBorderIdx : bin == ctr.TargetBorderIdx;
    }

    TVector<float> Reference(const TFixture& fixture, const TModelCtr& ctr, TConstArrayRef<ui32> order,
                             bool pastOnly, const TVector<TVector<ui32>>& queryHashes,
                             TConstArrayRef<ui8> queryFloatBins, ui32 targetBorders = 2) {
        TVector<float> result(queryFloatBins.size());
        TMap<ui64, std::pair<float, ui32>> sums;
        const auto& projection = ctr.Base.Projection;
        const bool frequency = ctr.Base.CtrType == ECtrType::FeatureFreq;
        if (!pastOnly || frequency) {
            for (ui32 row = 0; row < fixture.Targets.size(); ++row) {
                auto& entry = sums[ProjectionHash(projection, fixture.Hashes, fixture.FloatBins, row)];
                entry.first += Numerator(ctr, fixture.Targets[row], targetBorders);
                ++entry.second;
            }
            for (ui32 row = 0; row < result.size(); ++row) {
                const auto entry = sums[ProjectionHash(projection, queryHashes, queryFloatBins, row)];
                result[row] = frequency ? ctr.Calc(entry.second, fixture.Targets.size())
                    : ctr.Calc(entry.first, entry.second);
            }
        } else {
            for (ui32 row : order) {
                auto& entry = sums[ProjectionHash(projection, fixture.Hashes, fixture.FloatBins, row)];
                result[row] = ctr.Calc(entry.first, entry.second);
                entry.first += Numerator(ctr, fixture.Targets[row], targetBorders);
                ++entry.second;
            }
        }
        return result;
    }

    void CheckBatch(const TFixture& fixture, TMetalTreeCtrFeatures* helper, const TMetalTreeCtrBatch& batch,
                    const TVector<TVector<ui32>>& histories, ui32 targetBorders = 2, ui32 borderPermutation = 0) {
        UNIT_ASSERT(batch.GetFeatureCount() > 0);
        UNIT_ASSERT(batch.Stats.kernel_dispatches > 0);
        UNIT_ASSERT(batch.Stats.gpu_seconds >= 0);
        UNIT_ASSERT(batch.Stats.device_name[0]);
        UNIT_ASSERT_VALUES_EQUAL(batch.PermutationBins.size(), histories.size());
        UNIT_ASSERT_VALUES_EQUAL(batch.RegisteredCtrFlags.size(), batch.GetFeatureCount());
        bool saw64BitHash = false;
        TVector<bool> checked(batch.GetFeatureCount(), false);
        for (ui32 candidate = 0; candidate < batch.CandidateFeatures.size(); ++candidate) {
            const ui32 feature = batch.CandidateFeatures[candidate];
            const auto& split = helper->GetSplit(batch.FirstFeature + feature, batch.CandidateBins[candidate]);
            UNIT_ASSERT_VALUES_EQUAL(split.Type, ESplitType::OnlineCtr);
            UNIT_ASSERT_VALUES_EQUAL(batch.CandidateTypes[candidate], 0);
            if (checked[feature]) continue;
            checked[feature] = true;
            const auto& ctr = split.OnlineCtr.Ctr;
            TVector<float> borders;
            for (ui32 other = 0; other < batch.CandidateFeatures.size(); ++other)
                if (batch.CandidateFeatures[other] == feature)
                    borders.push_back(helper->GetSplit(batch.FirstFeature + feature, batch.CandidateBins[other]).OnlineCtr.Border);
            // Unknown tree CTR grids belong to the selected search history,
            // unlike the static CTR dataset's fixed permutation-zero grid.
            const auto gridValues = Reference(fixture, ctr, histories[borderPermutation], true,
                                               fixture.Hashes, fixture.FloatBins, targetBorders);
            const auto [minimum, maximum] = std::minmax_element(gridValues.begin(), gridValues.end());
            UNIT_ASSERT(*minimum < *maximum);
            UNIT_ASSERT_VALUES_EQUAL(borders.size(), 7);
            for (ui32 border = 0; border < borders.size(); ++border) {
                const float expected = *minimum + (border + 1) * (*maximum - *minimum) / 8;
                UNIT_ASSERT_DOUBLES_EQUAL(borders[border], expected, 1e-6);
            }
            ui32 uniqueUpperBound = 1;
            for (int category : ctr.Base.Projection.CatFeatures)
                uniqueUpperBound *= category == 0 ? 3 : category == 1 ? 4 : 2;
            uniqueUpperBound <<= ctr.Base.Projection.BinFeatures.size() + ctr.Base.Projection.OneHotFeatures.size();
            UNIT_ASSERT_VALUES_EQUAL(batch.CtrUniqueValues[feature], Min<ui32>(70, uniqueUpperBound));
            for (ui32 permutation = 0; permutation < histories.size(); ++permutation) {
                const auto values = Reference(fixture, ctr, histories[permutation], true,
                                               fixture.Hashes, fixture.FloatBins, targetBorders);
                UNIT_ASSERT_VALUES_EQUAL(batch.PermutationBins[permutation].size(), batch.GetFeatureCount() * fixture.Targets.size());
                for (ui32 row = 0; row < fixture.Targets.size(); ++row) {
                    const ui8 expected = std::lower_bound(borders.begin(), borders.end(), values[row]) - borders.begin();
                    UNIT_ASSERT_VALUES_EQUAL_C(batch.PermutationBins[permutation][feature * fixture.Targets.size() + row],
                        expected, "feature=" << feature << " permutation=" << permutation << " row=" << row);
                }
            }
            const auto& table = helper->GetCtrProvider()->CtrData.LearnCtrs.at(ctr.Base);
            auto viewer = table.GetIndexHashViewer();
            for (ui32 row = 0; row < fixture.Targets.size(); ++row) {
                const ui64 hash = ProjectionHash(ctr.Base.Projection, fixture.Hashes, fixture.FloatBins, row);
                saw64BitHash |= hash > 0xffffffffull;
                UNIT_ASSERT(viewer.GetIndex(hash) != NCatboost::TDenseIndexHashView::NotFoundIndex);
            }
        }
        UNIT_ASSERT(saw64BitHash);
        UNIT_ASSERT(std::all_of(checked.begin(), checked.end(), [](bool value) { return value; }));
    }

    void CheckInference(const TFixture& fixture, TMetalTreeCtrFeatures* helper,
                        const TMetalTreeCtrBatch& batch, ui32 targetBorders = 2) {
        auto hashes = fixture.Hashes;
        auto floatBins = fixture.FloatBins;
        // Both a new category combination and a wholly unseen category value.
        for (ui32 feature = 0; feature < hashes.size(); ++feature) {
            hashes[feature].push_back(fixture.Hashes[feature][0]);
            hashes[feature].push_back(feature < 2 ? 0x1234abcdu : fixture.Hashes[feature][0]);
        }
        floatBins.push_back(1);
        floatBins.push_back(0);
        const ui32 rows = floatBins.size();
        TVector<ui32> flatHashes;
        for (const auto& column : hashes) flatHashes.insert(flatHashes.end(), column.begin(), column.end());
        TVector<ui8> binFeatures = floatBins;
        for (ui32 row = 0; row < rows; ++row) binFeatures.push_back(hashes[2][row] == 41);
        TOneHotFeature oneHot;
        oneHot.CatFeatureIndex = 2;
        oneHot.Values = {41};
        auto provider = helper->GetCtrProvider();
        provider->SetupBinFeatureIndexes(fixture.FloatFeatures, {oneHot}, fixture.CatFeatures);
        TVector<TModelSplit> selected;
        TVector<TVector<float>> expectedCtrValues;
        for (ui32 candidate = 0; candidate < batch.CandidateFeatures.size(); ++candidate) {
            const auto& split = helper->GetSplit(batch.FirstFeature + batch.CandidateFeatures[candidate], batch.CandidateBins[candidate]);
            if (batch.CandidateBins[candidate] != 0) continue;
            const auto& ctr = split.OnlineCtr.Ctr;
            TVector<float> values(rows);
            provider->CalcCtrs({ctr}, binFeatures, flatHashes, rows, values);
            const auto expected = Reference(fixture, ctr, {}, false, hashes, floatBins, targetBorders);
            for (ui32 row = 0; row < rows; ++row) UNIT_ASSERT_DOUBLES_EQUAL(values[row], expected[row], 1e-6);
            selected.push_back(split);
            expectedCtrValues.push_back(expected);
        }
        TObliviousTreeBuilder builder(fixture.FloatFeatures, fixture.CatFeatures, {}, {}, 1);
        for (const auto& split : selected) builder.AddTree({split}, TVector<double>{-1, 2}, TVector<double>{1, 1});
        TFullModel model;
        builder.Build(model.ModelTrees.GetMutable());
        model.CtrProvider = provider->Clone();
        model.UpdateDynamicData();
        TVector<TVector<float>> floatRows(rows);
        TVector<TVector<int>> catRows(rows);
        TVector<TConstArrayRef<float>> floatViews;
        TVector<TConstArrayRef<int>> catViews;
        TVector<double> expectedPredictions(rows), predictions(rows);
        for (ui32 row = 0; row < rows; ++row) {
            floatRows[row] = {float(floatBins[row])};
            for (const auto& column : hashes) catRows[row].push_back(static_cast<i32>(column[row]));
            floatViews.push_back(floatRows[row]);
            catViews.push_back(catRows[row]);
            for (ui32 tree = 0; tree < selected.size(); ++tree)
                expectedPredictions[row] += expectedCtrValues[tree][row] > selected[tree].OnlineCtr.Border ? 2 : -1;
        }
        for (ui32 roundTrip = 0; roundTrip < 3; ++roundTrip) {
            model.Calc(floatViews, catViews, predictions);
            UNIT_ASSERT_VALUES_EQUAL(predictions, expectedPredictions);
            if (roundTrip == 0) {
                TStringStream stream;
                model.Save(&stream);
                TFullModel restored;
                restored.Load(&stream);
                model = std::move(restored);
            } else if (roundTrip == 1) {
                TFullModel restored;
                ConvertJsonToCatboostModel(ConvertModelToJson(model), &restored);
                model = std::move(restored);
            }
        }
    }
}

Y_UNIT_TEST_SUITE(TMetalTreeCtrFeatures) {
    Y_UNIT_TEST(TestAllTypesNumericAndOneHotPastOnlyHistory) {
        TFixture fixture;
        NPar::TLocalExecutor executor;
        const auto histories = fixture.Histories();
        for (ECtrType type : {ECtrType::Borders, ECtrType::Buckets, ECtrType::FloatTargetMeanValue, ECtrType::FeatureFreq}) {
            auto options = Options(type);
            NCB::TMetalTreeCtrFeatures helper(fixture.Data, options, &executor, FirstFeature, 70, {}, histories);
            helper.BeginTree();
            auto numeric = helper.AddSplit(TModelSplit(TFloatSplit(0, 0.5f)), 1);
            CheckBatch(fixture, &helper, numeric, histories, 2, 1);
            CheckInference(fixture, &helper, numeric);
            auto oneHot = helper.AddSplit(TModelSplit(TOneHotSplit(2, 41)), 1);
            CheckBatch(fixture, &helper, oneHot, histories, 2, 1);
            CheckInference(fixture, &helper, oneHot);
            for (ui32 feature : oneHot.ActiveFeatures) UNIT_ASSERT(feature >= oneHot.FirstFeature);
            auto combined = helper.AddSplit(SimpleCtrSplit(), 1);
            CheckBatch(fixture, &helper, combined, histories, 2, 1);
            CheckInference(fixture, &helper, combined);
            UNIT_ASSERT_VALUES_EQUAL(combined.ActiveFeatures.size(), oneHot.GetFeatureCount() + combined.GetFeatureCount());
        }
    }

    Y_UNIT_TEST(TestSnapshotRestoresBanksGridsAndStableIds) {
        TFixture fixture;
        NPar::TLocalExecutor executor;
        auto options = Options(ECtrType::FloatTargetMeanValue);
        const auto histories = fixture.Histories();
        NCB::TMetalTreeCtrFeatures helper(fixture.Data, options, &executor, FirstFeature, 70, {}, histories);
        auto first = helper.AddSplit(TModelSplit(TFloatSplit(0, 0.5f)), 2);
        auto second = helper.AddSplit(SimpleCtrSplit(), 2);
        TStringStream snapshot;
        helper.Save(&snapshot);
        NCB::TMetalTreeCtrFeatures restored(fixture.Data, options, &executor, FirstFeature, 70, {}, histories);
        auto banks = restored.Restore(&snapshot);
        UNIT_ASSERT_VALUES_EQUAL(banks.FirstFeature, FirstFeature);
        UNIT_ASSERT_VALUES_EQUAL(restored.GetFeatureCount(), helper.GetFeatureCount());
        UNIT_ASSERT(banks.ActiveFeatures.empty());
        for (ui32 permutation = 0; permutation < histories.size(); ++permutation) {
            auto expected = first.PermutationBins[permutation];
            expected.insert(expected.end(), second.PermutationBins[permutation].begin(), second.PermutationBins[permutation].end());
            UNIT_ASSERT_VALUES_EQUAL(banks.PermutationBins[permutation], expected);
        }
        for (ui32 candidate = 0; candidate < banks.CandidateFeatures.size(); ++candidate) {
            const ui32 feature = FirstFeature + banks.CandidateFeatures[candidate];
            const ui32 bin = banks.CandidateBins[candidate];
            UNIT_ASSERT(restored.GetSplit(feature, bin) == helper.GetSplit(feature, bin));
        }
        CheckInference(fixture, &restored, banks);
        // This unselected config retains a variant for its original history.
        auto reactivated = restored.AddSplit(TModelSplit(TFloatSplit(0, 0.5f)), 2);
        UNIT_ASSERT_VALUES_EQUAL(reactivated.GetFeatureCount(), 0);
        UNIT_ASSERT_VALUES_EQUAL(reactivated.Stats.kernel_dispatches, 0);
        UNIT_ASSERT_VALUES_EQUAL(reactivated.ActiveFeatures, first.ActiveFeatures);
        auto continued = restored.AddSplit(TModelSplit(TOneHotSplit(2, 41)), 0);
        UNIT_ASSERT_VALUES_EQUAL(continued.FirstFeature, FirstFeature + helper.GetFeatureCount());
        CheckBatch(fixture, &restored, continued, histories);
    }

    Y_UNIT_TEST(TestBinaryBucketsComplementAndIdentityDefault) {
        TFixture fixture;
        NPar::TLocalExecutor executor;
        auto options = Options(ECtrType::Buckets, 1);
        NCB::TMetalTreeCtrFeatures helper(fixture.Data, options, &executor, FirstFeature, 70);
        UNIT_ASSERT_VALUES_EQUAL(helper.GetPermutationCount(), 1);
        const auto batch = helper.AddSplit(SimpleCtrSplit());
        UNIT_ASSERT_VALUES_EQUAL(batch.GetFeatureCount(), 2); // One target bin, two distinct priors.
        CheckBatch(fixture, &helper, batch, {fixture.Histories()[0]}, 1);
        CheckInference(fixture, &helper, batch, 1);
    }

    Y_UNIT_TEST(TestInvalidBorderPermutationDoesNotMutateScheduler) {
        TFixture fixture;
        NPar::TLocalExecutor executor;
        auto options = Options(ECtrType::Borders);
        NCB::TMetalTreeCtrFeatures helper(fixture.Data, options, &executor, FirstFeature, 70);
        UNIT_ASSERT_EXCEPTION(helper.AddSplit(TModelSplit(TFloatSplit(0, 0.5f)), 1), TCatBoostException);
        UNIT_ASSERT_VALUES_EQUAL(helper.GetFeatureCount(), 0);
        const auto batch = helper.AddSplit(TModelSplit(TOneHotSplit(2, 41)));
        for (ui32 candidate = 0; candidate < batch.CandidateFeatures.size(); ++candidate) {
            const auto& projection = helper.GetSplit(batch.FirstFeature + batch.CandidateFeatures[candidate],
                                                     batch.CandidateBins[candidate]).OnlineCtr.Ctr.Base.Projection;
            UNIT_ASSERT(projection.BinFeatures.empty());
            UNIT_ASSERT_VALUES_EQUAL(projection.OneHotFeatures.size(), 1);
        }
        CheckBatch(fixture, &helper, batch, {fixture.Histories()[0]});
    }

    Y_UNIT_TEST(TestSnapshotRejectsDifferentHistoryContents) {
        TFixture fixture;
        NPar::TLocalExecutor executor;
        auto options = Options(ECtrType::FloatTargetMeanValue);
        auto histories = fixture.Histories();
        NCB::TMetalTreeCtrFeatures helper(fixture.Data, options, &executor, FirstFeature, 70, {}, histories);
        helper.AddSplit(SimpleCtrSplit());
        TStringStream snapshot;
        helper.Save(&snapshot);
        std::swap(histories[1][0], histories[1][1]);
        NCB::TMetalTreeCtrFeatures restored(fixture.Data, options, &executor, FirstFeature, 70, {}, histories);
        UNIT_ASSERT_EXCEPTION(restored.Restore(&snapshot), TCatBoostException);
        UNIT_ASSERT_VALUES_EQUAL(restored.GetFeatureCount(), 0);
    }

    Y_UNIT_TEST(TestUnknownGridsSwitchHistoryAndReuseStableVariants) {
        TFixture fixture;
        NPar::TLocalExecutor executor;
        auto options = Options(ECtrType::FloatTargetMeanValue);
        const auto histories = fixture.Histories();
        NCB::TMetalTreeCtrFeatures helper(fixture.Data, options, &executor, FirstFeature, 70, {}, histories);
        const TModelSplit predicate(TFloatSplit(0, 0.5f));
        auto first = helper.AddSplit(predicate, 0);
        CheckBatch(fixture, &helper, first, histories, 2, 0);
        UNIT_ASSERT(helper.GetRegisteredFeatures().empty());
        helper.BeginTree();
        auto second = helper.AddSplit(predicate, 1);
        CheckBatch(fixture, &helper, second, histories, 2, 1);
        UNIT_ASSERT_VALUES_EQUAL(second.GetFeatureCount(), first.GetFeatureCount());
        UNIT_ASSERT_VALUES_EQUAL(second.FirstFeature, FirstFeature + first.GetFeatureCount());
        bool gridChanged = false;
        for (ui32 feature = 0; feature < first.GetFeatureCount(); ++feature) {
            UNIT_ASSERT_VALUES_EQUAL(second.ActiveFeatures[feature], second.FirstFeature + feature);
            for (ui32 border = 0; border < 7; ++border)
                gridChanged |= helper.GetSplit(first.FirstFeature + feature, border).OnlineCtr.Border !=
                    helper.GetSplit(second.FirstFeature + feature, border).OnlineCtr.Border;
        }
        UNIT_ASSERT(gridChanged);
        helper.BeginTree();
        auto reused = helper.AddSplit(predicate, 0);
        UNIT_ASSERT_VALUES_EQUAL(reused.GetFeatureCount(), 0);
        UNIT_ASSERT_VALUES_EQUAL(reused.Stats.kernel_dispatches, 0);
        UNIT_ASSERT_VALUES_EQUAL(reused.ActiveFeatures, first.ActiveFeatures);
        helper.BeginTree();
        auto fourthHistory = helper.AddSplit(predicate, 3);
        CheckBatch(fixture, &helper, fourthHistory, histories, 2, 3);
        UNIT_ASSERT_VALUES_EQUAL(fourthHistory.GetFeatureCount(), first.GetFeatureCount());
    }

    Y_UNIT_TEST(TestWinnerRegistrationKeepsBinarizationConfigsDistinct) {
        TFixture fixture;
        NPar::TLocalExecutor executor;
        auto options = Options(ECtrType::FloatTargetMeanValue);
        options.CatFeatureParams->CombinationCtrs = TVector<NCatboostOptions::TCtrDescription>{
            NCatboostOptions::TCtrDescription(ECtrType::FloatTargetMeanValue, {{0.25f, 1.0f}},
                NCatboostOptions::TBinarizationOptions(EBorderSelectionType::Uniform, 3, ENanMode::Forbidden)),
            NCatboostOptions::TCtrDescription(ECtrType::FloatTargetMeanValue, {{0.25f, 1.0f}},
                NCatboostOptions::TBinarizationOptions(EBorderSelectionType::Uniform, 7, ENanMode::Forbidden))};
        const auto histories = fixture.Histories();
        NCB::TMetalTreeCtrFeatures helper(fixture.Data, options, &executor, FirstFeature, 70, {}, histories);
        const TModelSplit predicate(TFloatSplit(0, 0.5f));
        auto first = helper.AddSplit(predicate, 0);
        UNIT_ASSERT_VALUES_EQUAL(first.GetFeatureCount(), 4);
        helper.MarkSelected(FirstFeature); // Last-depth winner: no subsequent AddSplit call.
        UNIT_ASSERT_VALUES_EQUAL(helper.GetRegisteredFeatures(), TVector<ui32>{FirstFeature});
        helper.BeginTree();
        auto second = helper.AddSplit(predicate, 1);
        UNIT_ASSERT_VALUES_EQUAL(second.GetFeatureCount(), 3);
        UNIT_ASSERT_VALUES_EQUAL(second.ActiveFeatures,
                                 (TVector<ui32>{FirstFeature, FirstFeature + 4, FirstFeature + 5, FirstFeature + 6}));
        // Same exported CTR, different training binarization config. Selecting
        // the 3-border feature must not freeze the 7-border feature's grid.
        UNIT_ASSERT(helper.GetSplit(FirstFeature, 0).OnlineCtr.Ctr ==
                    helper.GetSplit(FirstFeature + 4, 0).OnlineCtr.Ctr);
        UNIT_ASSERT_EXCEPTION(helper.GetSplit(FirstFeature, 3), TCatBoostException);
        UNIT_ASSERT(helper.GetSplit(FirstFeature + 4, 6).OnlineCtr.Border > 0);
        helper.MarkSelected(FirstFeature + 4);
        helper.BeginTree();
        auto third = helper.AddSplit(predicate, 2);
        UNIT_ASSERT_VALUES_EQUAL(third.GetFeatureCount(), 2);
        UNIT_ASSERT_VALUES_EQUAL(third.ActiveFeatures,
                                 (TVector<ui32>{FirstFeature, FirstFeature + 4, FirstFeature + 7, FirstFeature + 8}));
        TStringStream snapshot;
        helper.Save(&snapshot);
        NCB::TMetalTreeCtrFeatures restored(fixture.Data, options, &executor, FirstFeature, 70, {}, histories);
        auto banks = restored.Restore(&snapshot);
        UNIT_ASSERT_VALUES_EQUAL(restored.GetRegisteredFeatures(), (TVector<ui32>{FirstFeature, FirstFeature + 4}));
        UNIT_ASSERT_VALUES_EQUAL(banks.RegisteredCtrFlags, (TVector<ui8>{1, 0, 0, 0, 1, 0, 0, 0, 0}));
        for (ui32 permutation = 0; permutation < histories.size(); ++permutation) {
            auto expected = first.PermutationBins[permutation];
            expected.insert(expected.end(), second.PermutationBins[permutation].begin(), second.PermutationBins[permutation].end());
            expected.insert(expected.end(), third.PermutationBins[permutation].begin(), third.PermutationBins[permutation].end());
            UNIT_ASSERT_VALUES_EQUAL(banks.PermutationBins[permutation], expected);
        }
        for (ui32 candidate = 0; candidate < banks.CandidateFeatures.size(); ++candidate) {
            const ui32 feature = FirstFeature + banks.CandidateFeatures[candidate];
            const ui32 bin = banks.CandidateBins[candidate];
            UNIT_ASSERT(restored.GetSplit(feature, bin) == helper.GetSplit(feature, bin));
        }
        CheckInference(fixture, &restored, banks);
        const auto reused = restored.AddSplit(predicate, 1);
        UNIT_ASSERT_VALUES_EQUAL(reused.GetFeatureCount(), 0);
        UNIT_ASSERT_VALUES_EQUAL(reused.Stats.kernel_dispatches, 0);
        UNIT_ASSERT_VALUES_EQUAL(reused.ActiveFeatures, second.ActiveFeatures);
    }

    Y_UNIT_TEST(TestEagerRegistrationUsesStrictCategoryOnlyThreshold) {
        TFixture fixture;
        NPar::TLocalExecutor executor;
        const auto histories = fixture.Histories();
        for (ui32 threshold : {2, 3}) {
            auto options = Options(ECtrType::FloatTargetMeanValue);
            options.ObliviousTreeOptions->MaxCtrComplexityForBordersCaching = threshold;
            NCB::TMetalTreeCtrFeatures helper(fixture.Data, options, &executor, FirstFeature, 70, {}, histories);
            auto first = helper.AddSplit(SimpleCtrSplit(), 0); // Two categorical components.
            UNIT_ASSERT_VALUES_EQUAL(first.GetFeatureCount(), 2);
            const bool eager = threshold == 3;
            UNIT_ASSERT_VALUES_EQUAL(first.RegisteredCtrFlags, (TVector<ui8>{ui8(eager), ui8(eager)}));
            UNIT_ASSERT_VALUES_EQUAL(helper.GetRegisteredFeatures().size(), eager ? 2 : 0);
            helper.BeginTree();
            auto second = helper.AddSplit(SimpleCtrSplit(), 1);
            UNIT_ASSERT_VALUES_EQUAL(second.GetFeatureCount(), eager ? 0 : 2);
            helper.BeginTree();
            auto predicates = helper.AddSplit(TModelSplit(TFloatSplit(0, 0.5f)), 1);
            UNIT_ASSERT_VALUES_EQUAL(predicates.RegisteredCtrFlags, (TVector<ui8>{0, 0, 0, 0}));
        }
    }
}
