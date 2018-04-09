#include "load_data.h"
#include <catboost/libs/helpers/permutation.h>

namespace NCatboostCuda {
    void TDataProviderBuilder::StartNextBlock(ui32 blockSize) {
        Cursor = DataProvider.Targets.size();
        const auto newDataSize = Cursor + blockSize;

        DataProvider.Targets.resize(newDataSize);
        DataProvider.Weights.resize(newDataSize, 1.0);
        DataProvider.QueryIds.resize(newDataSize);
        DataProvider.SubgroupIds.resize(newDataSize);
        DataProvider.Timestamp.resize(newDataSize);

        for (ui32 i = Cursor; i < DataProvider.QueryIds.size(); ++i) {
            DataProvider.QueryIds[i] = TGroupId(i);
            DataProvider.SubgroupIds[i] = i;
        }

        for (auto& baseline : DataProvider.Baseline) {
            baseline.resize(newDataSize);
        }
        for (ui32 factor = 0; factor < FeatureValues.size(); ++factor) {
            if (IgnoreFeatures.count(factor) == 0) {
                FeatureValues[factor].resize(newDataSize);
            }
        }

        DataProvider.DocIds.resize(newDataSize);
    }

    inline bool HasQueryIds(const TVector<TGroupId>& qids) {
        for (ui32 i = 0; i < qids.size(); ++i) {
            if (qids[i] != TGroupId(i)) {
                return true;
            }
        }
        return false;
    }

    void TDataProviderBuilder::Finish() {
        CB_ENSURE(!IsDone, "Error: can't finish more than once");
        DataProvider.Features.reserve(FeatureValues.size());

        DataProvider.Order.resize(DataProvider.Targets.size());
        std::iota(DataProvider.Order.begin(),
                  DataProvider.Order.end(), 0);

        if (!AreEqualTo<ui64>(DataProvider.Timestamp, 0)) {
            ShuffleFlag = false;
            DataProvider.Order = CreateOrderByKey(DataProvider.Timestamp);
        }

        bool hasQueryIds = HasQueryIds(DataProvider.QueryIds);
        if (!hasQueryIds) {
            DataProvider.QueryIds.resize(0);
        }

        //TODO(noxoomo): it's not safe here, if we change order with shuffle everything'll go wrong
        if (Pairs.size()) {
            //they are local, so we don't need shuffle
            CB_ENSURE(hasQueryIds, "Error: for GPU pairwise learning you should provide query id column. Query ids will be used to split data between devices and for dynamic boosting learning scheme.");
            DataProvider.FillQueryPairs(Pairs);
        }

        if (ShuffleFlag) {
            if (hasQueryIds) {
                //should not change order inside query for pairs consistency
                QueryConsistentShuffle(Seed, 1, DataProvider.QueryIds, &DataProvider.Order);
            } else {
                Shuffle(Seed, 1, DataProvider.Targets.size(), &DataProvider.Order);
            }
            DataProvider.SetShuffleSeed(Seed);
        }

        if (ShuffleFlag || !DataProvider.Timestamp.empty()) {
            DataProvider.ApplyOrderToMetaColumns();
        }

        TVector<TString> featureNames;
        featureNames.resize(FeatureValues.size());

        TAdaptiveLock lock;

        NPar::TLocalExecutor executor;
        executor.RunAdditionalThreads(BuildThreads - 1);

        TVector<TFeatureColumnPtr> featureColumns(FeatureValues.size());

        if (!IsTest) {
            RegisterFeaturesInFeatureManager(featureColumns);
        }

        TVector<TVector<float>> grid;
        grid.resize(FeatureValues.size());

        NPar::ParallelFor(executor, 0, FeatureValues.size(), [&](ui32 featureId) {
            auto featureName = GetFeatureName(featureId);
            featureNames[featureId] = featureName;

            if (FeatureValues[featureId].size() == 0) {
                return;
            }

            TVector<float> line(DataProvider.Order.size());
            for (ui32 i = 0; i < DataProvider.Order.size(); ++i) {
                line[i] = FeatureValues[featureId][DataProvider.Order[i]];
            }

            if (CatFeatureIds.has(featureId)) {
                static_assert(sizeof(float) == sizeof(ui32), "Error: float size should be equal to ui32 size");
                const bool shouldSkip = IsTest && (CatFeaturesPerfectHashHelper.GetUniqueValues(featureId) == 0);
                if (!shouldSkip) {
                    auto data = CatFeaturesPerfectHashHelper.UpdatePerfectHashAndBinarize(featureId,
                                                                                          ~line,
                                                                                          line.size());

                    const ui32 uniqueValues = CatFeaturesPerfectHashHelper.GetUniqueValues(featureId);

                    if (uniqueValues > 1) {
                        auto compressedData = CompressVector<ui64>(~data, line.size(), IntLog2(uniqueValues));
                        featureColumns[featureId] = MakeHolder<TCatFeatureValuesHolder>(featureId,
                                                                                        line.size(),
                                                                                        std::move(compressedData),
                                                                                        uniqueValues,
                                                                                        featureName);
                    }
                }
            } else {
                auto floatFeature = MakeHolder<TFloatValuesHolder>(featureId,
                                                                   std::move(line),
                                                                   featureName);

                TVector<float>& borders = grid[featureId];

                ENanMode nanMode = ENanMode::Forbidden;
                {
                    TGuard<TAdaptiveLock> guard(lock);
                    nanMode = FeaturesManager.GetOrCreateNanMode(*floatFeature);
                }

                if (FeaturesManager.HasFloatFeatureBorders(*floatFeature)) {
                    borders = FeaturesManager.GetFloatFeatureBorders(*floatFeature);
                }

                if (borders.empty() && !IsTest) {
                    const auto& floatValues = floatFeature->GetValues();
                    NCatboostOptions::TBinarizationOptions config = FeaturesManager.GetFloatFeatureBinarization();
                    config.NanMode = nanMode;
                    borders = BuildBorders(floatValues, floatFeature->GetId(), config);
                }
                if (borders.ysize() == 0) {
                    MATRIXNET_DEBUG_LOG << "Float Feature #" << featureId << " is empty" << Endl;
                    return;
                }

                auto binarizedData = BinarizeLine(floatFeature->GetValues().data(),
                                                  floatFeature->GetValues().size(),
                                                  nanMode,
                                                  borders);

                const int binCount = static_cast<const int>(borders.size() + 1 + (ENanMode::Forbidden != nanMode));
                auto compressedLine = CompressVector<ui64>(binarizedData, IntLog2(binCount));

                featureColumns[featureId] = MakeHolder<TBinarizedFloatValuesHolder>(featureId,
                                                                                    floatFeature->GetValues().size(),
                                                                                    nanMode,
                                                                                    borders,
                                                                                    std::move(compressedLine),
                                                                                    featureName);
            }

            //Free memory
            {
                auto emptyVec = TVector<float>();
                FeatureValues[featureId].swap(emptyVec);
            }
        });

        for (ui32 featureId = 0; featureId < featureColumns.size(); ++featureId) {
            if (CatFeatureIds.has(featureId)) {
                if (featureColumns[featureId] == nullptr && (!IsTest)) {
                    MATRIXNET_DEBUG_LOG << "Cat Feature #" << featureId << " is empty" << Endl;
                }
            } else if (featureColumns[featureId] != nullptr) {
                if (!FeaturesManager.HasFloatFeatureBordersForDataProviderFeature(featureId)) {
                    FeaturesManager.SetFloatFeatureBordersForDataProviderId(featureId,
                                                                            std::move(grid[featureId]));
                }
            }
            if (featureColumns[featureId] != nullptr) {
                DataProvider.Features.push_back(std::move(featureColumns[featureId]));
            }
        }

        DataProvider.BuildIndicesRemap();

        if (!IsTest) {
            TOnCpuGridBuilderFactory gridBuilderFactory;
            FeaturesManager.SetTargetBorders(TBordersBuilder(gridBuilderFactory,
                                                             DataProvider.GetTargets())(FeaturesManager.GetTargetBinarizationDescription()));
        }

        DataProvider.FeatureNames = featureNames;
        DataProvider.CatFeatureIds = CatFeatureIds;

        if (ClassesWeights.size()) {
            Reweight(DataProvider.Targets, ClassesWeights, &DataProvider.Weights);
        }
        IsDone = true;
    }
}
