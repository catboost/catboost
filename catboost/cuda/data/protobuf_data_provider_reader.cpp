#include "protobuf_data_provider_reader.h"
#include "data_utils.h"
#include <catboost/libs/helpers/permutation.h>

using namespace NCatboostCuda;

inline TVector<ui64> GatherCompressed(const TVector<ui64>& order,
                                      const TVector<ui64>& src, ui32 binCount) {
    TVector<ui32> decompressedLine = DecompressVector<ui64, ui32>(src, (ui32)order.size(), IntLog2(binCount));
    ApplyPermutation(order, decompressedLine);
    return CompressVector<ui64>(decompressedLine, IntLog2(binCount));
}

void TCatBoostProtoPoolReader::AddFeatureColumn(TIFStream& input,
                                                ui32 docCount,
                                                const TVector<ui64>* order,
                                                TVector<TFeatureColumnPtr>* nzColumns,
                                                TVector<TString>* featureNames,
                                                TSet<int>* catFeatureIds) {
    ReadMessage(input, FeatureColumn);
    const auto& featureDescription = FeatureColumn.GetFeatureDescription();
    auto description = featureDescription;
    const ui32 featureId = featureDescription.GetFeatureId();
    const TString featureName = featureDescription.HasFeatureName()
                                    ? featureDescription.GetFeatureName()
                                    : ToString(featureId);
    featureNames->push_back(featureName);
    const bool skipFeature = IgnoreFeatures.has(featureId);

    switch (description.GetFeatureType()) {
        case ::NCompressedPool::TFeatureType::Float: {
            if (!FeaturesManager.IsKnown(featureId)) {
                FeaturesManager.RegisterDataProviderFloatFeature(featureId);
            }
            auto data = FromProtoToVector(FeatureColumn.GetFloatColumn().GetValues());
            if (order) {
                ApplyPermutation(*order, data);
            }
            auto values = MakeHolder<TFloatValuesHolder>(featureId, std::move(data));
            TOnCpuGridBuilderFactory gridBuilderFactory;
            auto borders = FeaturesManager.GetOrCreateFloatFeatureBorders(*values, TBordersBuilder(gridBuilderFactory, values->GetValues()));
            if (borders.size() && !skipFeature) {
                nzColumns->push_back(FloatToBinarizedColumn(*values, borders));
            }
            break;
        }
        case ::NCompressedPool::TFeatureType::Binarized: {
            TVector<float> borders = FromProtoToVector(FeatureColumn.GetBinarization().GetBorders());
            if (!FeaturesManager.IsKnown(featureId)) {
                FeaturesManager.RegisterDataProviderFloatFeature(featureId);
            }
            if (!FeaturesManager.HasFloatFeatureBordersForDataProviderFeature(featureId)) {
                FeaturesManager.SetFloatFeatureBordersForDataProviderId(featureId, std::move(borders));
            }
            if (borders.size() && !skipFeature) {
                const auto& binarizedData = FeatureColumn.GetBinarizedColumn().GetData();
                auto values = FromProtoToVector(binarizedData);
                if (order) {
                    values = GatherCompressed(*order, values, borders.size() + 1);
                }
                CB_ENSURE(borders.size(), "Error: binarization should be positive");
                auto feature = MakeHolder<TBinarizedFloatValuesHolder>(featureId,
                                                                       docCount,
                                                                       ENanMode::Forbidden,
                                                                       borders,
                                                                       std::move(values),
                                                                       featureName);
                nzColumns->push_back(feature.Release());
            }
            break;
        }
        case ::NCompressedPool::TFeatureType::Categorical: {
            const auto& binarizedData = FeatureColumn.GetBinarizedColumn().GetData();
            if (!FeaturesManager.IsKnown(featureId)) {
                FeaturesManager.RegisterDataProviderCatFeature(featureId);
            }
            catFeatureIds->insert(featureId);

            if (FeatureColumn.GetUniqueValues()) {
                auto data = FromProtoToVector(binarizedData);
                if (order) {
                    data = GatherCompressed(*order, data, FeatureColumn.GetUniqueValues());
                }
                auto values = MakeHolder<TCatFeatureValuesHolder>(featureId,
                                                                  docCount,
                                                                  std::move(data),
                                                                  FeatureColumn.GetUniqueValues(),
                                                                  featureName);
                nzColumns->push_back(values.Release());
            }
            break;
        }
        default: {
            ythrow yexception() << "Error: unknown column type";
        }
    }
}

TDataProvider TCatBoostProtoPoolReader::Read(TIFStream& input) {
    TDataProvider dataProvider;
    NCompressedPool::TPoolStructure poolStructure;
    ReadMessage(input, poolStructure);

    ReadFloatColumn(input, dataProvider.Targets);

    if (poolStructure.GetWeightColumn()) {
        ReadFloatColumn(input, dataProvider.Weights);
    } else {
        dataProvider.Weights.resize(dataProvider.Targets.size());
        std::fill(dataProvider.Weights.begin(), dataProvider.Weights.begin(), 1.0f);
    }

    if (poolStructure.GetDocIdColumn()) {
        ReadUnsignedIntColumn(input, dataProvider.DocIds);
    }

    if (poolStructure.GetTimestampColumn()) {
        ReadUnsignedIntColumn(input, dataProvider.Timestamp);
    }

    if (poolStructure.GetGroupIdColumn()) {
        ReadUnsignedIntColumn(input, dataProvider.QueryIds);
    }
    if (poolStructure.GetSubgroupIdColumn()) {
        ReadUnsignedIntColumn(input, dataProvider.SubgroupIds);
    }

    dataProvider.Baseline.resize(poolStructure.GetBaselineColumn());

    for (ui32 i = 0; i < poolStructure.GetBaselineColumn(); ++i) {
        ReadFloatColumn(input, dataProvider.Baseline[i]);
    }

    dataProvider.Order.resize(dataProvider.Targets.size());
    std::iota(dataProvider.Order.begin(),
              dataProvider.Order.end(), 0);

    const bool hasQueryIds = poolStructure.GetGroupIdColumn();

    if (Pairs.size()) {
        //they are local, so we don't need shuffle
        CB_ENSURE(hasQueryIds, "Error: for GPU pairwise learning you should provide query id column. Query ids will be used to split data between devices and for dynamic boosting learning scheme.");
        dataProvider.FillQueryPairs(Pairs);
    }

    bool needGather = true;
    if (!AreEqualTo<ui64>(dataProvider.Timestamp, 0)) {
        dataProvider.Order = CreateOrderByKey(dataProvider.Timestamp);
    } else if (!HasTime) {
        if (hasQueryIds) {
            QueryConsistentShuffle(Seed, 1, dataProvider.QueryIds, &dataProvider.Order);
        } else {
            Shuffle(Seed, 1, dataProvider.Targets.size(), &dataProvider.Order);
        }
        dataProvider.SetShuffleSeed(Seed);
    } else {
        needGather = false;
    }
    if (needGather) {
        dataProvider.ApplyOrderToMetaColumns();
    }

    for (ui32 feature = 0; feature < poolStructure.GetFeatureCount(); ++feature) {
        AddFeatureColumn(input,
                         poolStructure.GetDocCount(),
                         needGather ? &dataProvider.Order : nullptr,
                         &dataProvider.Features,
                         &dataProvider.FeatureNames,
                         &dataProvider.CatFeatureIds);
    }

    if (FeaturesManager.GetTargetBorders().size() == 0) {
        TOnCpuGridBuilderFactory gridBuilderFactory;
        FeaturesManager.SetTargetBorders(TBordersBuilder(gridBuilderFactory,
                                                         dataProvider.GetTargets())(FeaturesManager.GetTargetBinarizationDescription()));
    }

    dataProvider.BuildIndicesRemap();

    if (ClassesWeights.size()) {
        Reweight(dataProvider.Targets, ClassesWeights, &dataProvider.Weights);
    }
    if (dataProvider.CatFeatureIds.size()) {
        ythrow TCatboostException() << "Error: load catFeatures from protobuf is unfinished yet";
    }
    return dataProvider;
}
