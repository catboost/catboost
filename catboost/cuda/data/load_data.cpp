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

        for (ui32 featureId = 0; featureId < FeatureBlobs.size(); ++featureId) {
            if (IgnoreFeatures.count(featureId) == 0) {
                FeatureBlobs[featureId].resize(newDataSize * GetBytesPerFeature(featureId));
            }
        }

        DataProvider.DocIds.resize(newDataSize);
    }

    static inline bool HasQueryIds(const TVector<TGroupId>& qids) {
        for (ui32 i = 0; i < qids.size(); ++i) {
            if (qids[i] != TGroupId(i)) {
                return true;
            }
        }
        return false;
    }


    template <class T>
    static inline TVector<T> MakeOrderedLine(const TVector<ui8>& source,
                                             const TVector<ui64>& order) {
        CB_ENSURE(source.size() ==  sizeof(T) * order.size(), "Error: size should be consistent " << source.size() << "  "<< order.size() << " " << sizeof(T));
        TVector<T> line(order.size());

        for (size_t i = 0; i < order.size(); ++i) {
            const T* rawSourcePtr = reinterpret_cast<const T*>(source.data());
            line[i] = rawSourcePtr[order[i]];
        }
        return line;
    }


    void TDataProviderBuilder::Finish() {
        CB_ENSURE(!IsDone, "Error: can't finish more than once");
        DataProvider.Features.reserve(FeatureBlobs.size());

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
        featureNames.resize(FeatureBlobs.size());

        TAdaptiveLock lock;

        NPar::TLocalExecutor executor;
        executor.RunAdditionalThreads(BuildThreads - 1);

        TVector<TFeatureColumnPtr> featureColumns(FeatureBlobs.size());

        if (!IsTest) {
            RegisterFeaturesInFeatureManager(featureColumns);
        }


        NPar::ParallelFor(executor, 0, static_cast<ui32>(FeatureBlobs.size()), [&](ui32 featureId) {
            auto featureName = GetFeatureName(featureId);
            featureNames[featureId] = featureName;

            if (FeatureBlobs[featureId].size() == 0) {
                return;
            }


            EFeatureValuesType featureValuesType = FeatureTypes[featureId];

            if (featureValuesType == EFeatureValuesType::Categorical) {
                CB_ENSURE(featureValuesType == EFeatureValuesType::Categorical, "Wrong type " << featureValuesType);

                auto line = MakeOrderedLine<float>(FeatureBlobs[featureId],
                                                   DataProvider.Order);

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
            } else if (featureValuesType == EFeatureValuesType::BinarizedFloat) {
                const TVector<float>& borders = Borders.at(featureId);
                const ENanMode nanMode = NanModes.at(featureId);
                if (borders.ysize() == 0) {
                    MATRIXNET_DEBUG_LOG << "Float Feature #" << featureId << " is empty" << Endl;
                    return;
                }

                TVector<ui8> binarizedData = MakeOrderedLine<ui8>(FeatureBlobs[featureId],
                                                                  DataProvider.Order);

                const int binCount = static_cast<const int>(borders.size() + 1 + (ENanMode::Forbidden != nanMode));
                auto compressedLine = CompressVector<ui64>(binarizedData, IntLog2(binCount));

                featureColumns[featureId] = MakeHolder<TBinarizedFloatValuesHolder>(featureId,
                                                                                    DataProvider.Order.size(),
                                                                                    nanMode,
                                                                                    borders,
                                                                                    std::move(compressedLine),
                                                                                    featureName);
                with_lock(lock) {
                    FeaturesManager.SetOrCheckNanMode(*featureColumns[featureId],
                                                      nanMode);
                }
            } else {
                CB_ENSURE(featureValuesType == EFeatureValuesType::Float, "Wrong feature values type (" << featureValuesType << ") for feature #" << featureId);
                TVector<float> line(DataProvider.Order.size());
                for (ui32 i = 0; i < DataProvider.Order.size(); ++i) {
                    const float* floatFeatureSource = reinterpret_cast<float*>(FeatureBlobs[featureId].data());
                    line[i] = floatFeatureSource[DataProvider.Order[i]];
                }
                auto floatFeature = MakeHolder<TFloatValuesHolder>(featureId,
                                                                   std::move(line),
                                                                   featureName);

                TVector<float>& borders = Borders[featureId];

                auto& nanMode = NanModes[featureId];
                {
                    TGuard<TAdaptiveLock> guard(lock);
                    nanMode = FeaturesManager.GetOrComputeNanMode(*floatFeature);
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
                auto emptyVec = TVector<ui8>();
                FeatureBlobs[featureId].swap(emptyVec);
            }
        });

        for (ui32 featureId = 0; featureId < featureColumns.size(); ++featureId) {
            if (FeatureTypes[featureId] == EFeatureValuesType::Categorical) {
                if (featureColumns[featureId] == nullptr && (!IsTest)) {
                    MATRIXNET_DEBUG_LOG << "Cat Feature #" << featureId << " is empty" << Endl;
                }
            } else if (featureColumns[featureId] != nullptr) {
                if (!FeaturesManager.HasFloatFeatureBordersForDataProviderFeature(featureId)) {
                    FeaturesManager.SetFloatFeatureBordersForDataProviderId(featureId,
                                                                            std::move(Borders[featureId]));
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

        if (ClassesWeights.size()) {
            Reweight(DataProvider.Targets, ClassesWeights, &DataProvider.Weights);
        }
        IsDone = true;
    }

    void TDataProviderBuilder::WriteBinarizedFeatureToBlobImpl(ui32 localIdx, ui32 featureId, ui8 feature) {
        Y_ASSERT(IgnoreFeatures.count(featureId) == 0);
        Y_ASSERT(FeatureTypes[featureId] == EFeatureValuesType::BinarizedFloat);
        ui8* featureColumn = FeatureBlobs[featureId].data();
        featureColumn[GetLineIdx(localIdx)] = feature;
    }

    void TDataProviderBuilder::WriteFloatOrCatFeatureToBlobImpl(ui32 localIdx, ui32 featureId, float feature) {
        Y_ASSERT(IgnoreFeatures.count(featureId) == 0);
        Y_ASSERT(FeatureTypes[featureId] == EFeatureValuesType::Float || FeatureTypes[featureId] == EFeatureValuesType::Categorical);

        auto* featureColumn = reinterpret_cast<float*>(FeatureBlobs[featureId].data());
        featureColumn[GetLineIdx(localIdx)] = feature;
    }

    void TDataProviderBuilder::Start(const TPoolMetaInfo& poolMetaInfo,
                                     int docCount,
                                     const TVector<int>& catFeatureIds) {
        DataProvider.Features.clear();

        DataProvider.Baseline.clear();
        DataProvider.Baseline.resize(poolMetaInfo.BaselineCount);

        Cursor = 0;
        IsDone = false;

        FeatureBlobs.clear();
        FeatureBlobs.resize(poolMetaInfo.FeatureCount);

        FeatureTypes.resize(poolMetaInfo.FeatureCount, EFeatureValuesType::Float);
        for (int catFeature : catFeatureIds) {
            FeatureTypes[catFeature] = EFeatureValuesType::Categorical;
        }
        Borders.resize(poolMetaInfo.FeatureCount);
        NanModes.resize(poolMetaInfo.FeatureCount);

        for (size_t i = 0; i < BinarizedFeaturesMetaInfo.BinarizedFeatureIds.size(); ++i) {
            const size_t binarizedFeatureId = static_cast<const size_t>(BinarizedFeaturesMetaInfo.BinarizedFeatureIds[i]);
            const TVector<float>& borders = BinarizedFeaturesMetaInfo.Borders.at(i);
            CB_ENSURE(binarizedFeatureId < poolMetaInfo.FeatureCount, "Error: binarized feature " << binarizedFeatureId << " is out of range");
            FeatureTypes[binarizedFeatureId] = EFeatureValuesType::BinarizedFloat;
            NanModes[binarizedFeatureId] = BinarizedFeaturesMetaInfo.NanModes.at(i);
            Borders[binarizedFeatureId] = borders;
        }

        for (ui32 i = 0; i < poolMetaInfo.FeatureCount; ++i) {
            if (!IgnoreFeatures.has(i)) {
                ui32 bytesPerFeature = GetBytesPerFeature(i);
                FeatureBlobs[i].reserve(docCount * bytesPerFeature);
            }
        }

        DataProvider.CatFeatureIds = TSet<int>(catFeatureIds.begin(), catFeatureIds.end());
    }


}
