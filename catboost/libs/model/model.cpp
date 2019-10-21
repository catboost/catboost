#include "model.h"

#include "flatbuffers_serializer_helper.h"
#include "model_import_interface.h"
#include "model_build_helper.h"
#include "static_ctr_provider.h"

#include <catboost/libs/model/flatbuffers/model.fbs.h>

#include <catboost/libs/cat_feature/cat_feature.h>
#include <catboost/libs/helpers/borders_io.h>
#include <catboost/libs/logging/logging.h>

#include <catboost/private/libs/options/enum_helpers.h>
#include <catboost/private/libs/options/loss_description.h>
#include <catboost/private/libs/options/multiclass_label_options.h>

#include <library/json/json_reader.h>
#include <library/dbg_output/dump.h>
#include <library/dbg_output/auto.h>

#include <util/generic/algorithm.h>
#include <util/generic/fwd.h>
#include <util/generic/guid.h>
#include <util/generic/variant.h>
#include <util/generic/xrange.h>
#include <util/generic/ylimits.h>
#include <util/string/builder.h>
#include <util/stream/str.h>


static const char MODEL_FILE_DESCRIPTOR_CHARS[4] = {'C', 'B', 'M', '1'};

static ui32 GetModelFormatDescriptor() {
    return *reinterpret_cast<const ui32*>(MODEL_FILE_DESCRIPTOR_CHARS);
}

static const char* CURRENT_CORE_FORMAT_STRING = "FlabuffersModel_v1";

void OutputModel(const TFullModel& model, IOutputStream* const out) {
    Save(out, model);
}

void OutputModel(const TFullModel& model, const TStringBuf modelFile) {
    TOFStream f(TString{modelFile});  // {} because of the most vexing parse
    OutputModel(model, &f);
}

bool IsDeserializableModelFormat(EModelType format) {
    return NCB::TModelLoaderFactory::Has(format);
}

TFullModel ReadModel(const TString& modelFile, EModelType format) {
    CB_ENSURE(
        NCB::TModelLoaderFactory::Has(format),
        "Model format " << format << " deserialization not supported or missing. Link with catboost/libs/model/model_export if you need CoreML or JSON"
    );
    THolder<NCB::IModelLoader> modelLoader = NCB::TModelLoaderFactory::Construct(format);
    return modelLoader->ReadModel(modelFile);
}

TFullModel ReadModel(const void* binaryBuffer, size_t binaryBufferSize, EModelType format) {
    CB_ENSURE(
        NCB::TModelLoaderFactory::Has(format),
        "Model format " << format << " deserialization not supported or missing. Link with catboost/libs/model/model_export if you need CoreML or JSON"
    );
    THolder<NCB::IModelLoader> modelLoader = NCB::TModelLoaderFactory::Construct(format);
    return modelLoader->ReadModel(binaryBuffer, binaryBufferSize);
}

TString SerializeModel(const TFullModel& model) {
    TStringStream ss;
    OutputModel(model, &ss);
    return ss.Str();
}

TFullModel DeserializeModel(TMemoryInput serializedModel) {
    TFullModel model;
    Load(&serializedModel, model);
    return model;
}

TFullModel DeserializeModel(const TString& serializedModel) {
    return DeserializeModel(TMemoryInput{serializedModel.data(), serializedModel.size()});
}

static bool EstimatedFeatureIdsAreEqual(const TEstimatedFeature& lhs, const TEstimatedFeatureSplit& rhs) {
    return std::tie(lhs.SourceFeatureIndex, lhs.CalcerId, lhs.LocalIndex)
        == std::tie(rhs.SourceFeatureId, rhs.CalcerId, rhs.LocalId);
}

void TObliviousTrees::ProcessSplitsSet(
    const TSet<TModelSplit>& modelSplitSet,
    const TVector<size_t>& floatFeaturesInternalIndexesMap,
    const TVector<size_t>& catFeaturesInternalIndexesMap,
    const TVector<size_t>& textFeaturesInternalIndexesMap
) {
    THashSet<int> usedCatFeatureIndexes;
    THashSet<int> usedTextFeatureIndexes;
    for (const auto& split : modelSplitSet) {
        if (split.Type == ESplitType::FloatFeature) {
            const size_t internalFloatIndex = floatFeaturesInternalIndexesMap.at((size_t)split.FloatFeature.FloatFeature);
            FloatFeatures.at(internalFloatIndex).Borders.push_back(split.FloatFeature.Split);
        } else if (split.Type == ESplitType::EstimatedFeature) {
            const TEstimatedFeatureSplit estimatedFeatureSplit = split.EstimatedFeature;
            usedTextFeatureIndexes.insert(estimatedFeatureSplit.SourceFeatureId);

            if (EstimatedFeatures.empty() ||
                !EstimatedFeatureIdsAreEqual(EstimatedFeatures.back(), estimatedFeatureSplit)
            ) {
                TEstimatedFeature estimatedFeature{
                    estimatedFeatureSplit.SourceFeatureId,
                    estimatedFeatureSplit.CalcerId,
                    estimatedFeatureSplit.LocalId
                };
                EstimatedFeatures.emplace_back(estimatedFeature);
            }

            EstimatedFeatures.back().Borders.push_back(estimatedFeatureSplit.Split);
        } else if (split.Type == ESplitType::OneHotFeature) {
            usedCatFeatureIndexes.insert(split.OneHotFeature.CatFeatureIdx);
            if (OneHotFeatures.empty() || OneHotFeatures.back().CatFeatureIndex != split.OneHotFeature.CatFeatureIdx) {
                auto& ref = OneHotFeatures.emplace_back();
                ref.CatFeatureIndex = split.OneHotFeature.CatFeatureIdx;
            }
            OneHotFeatures.back().Values.push_back(split.OneHotFeature.Value);
        } else {
            const auto& projection = split.OnlineCtr.Ctr.Base.Projection;
            usedCatFeatureIndexes.insert(projection.CatFeatures.begin(), projection.CatFeatures.end());
            if (CtrFeatures.empty() || CtrFeatures.back().Ctr != split.OnlineCtr.Ctr) {
                CtrFeatures.emplace_back();
                CtrFeatures.back().Ctr = split.OnlineCtr.Ctr;
            }
            CtrFeatures.back().Borders.push_back(split.OnlineCtr.Border);
        }
    }
    for (const int usedCatFeatureIdx : usedCatFeatureIndexes) {
        CatFeatures[catFeaturesInternalIndexesMap.at(usedCatFeatureIdx)].SetUsedInModel(true);
    }
    for (const int usedTextFeatureIdx : usedTextFeatureIndexes) {
        TextFeatures[textFeaturesInternalIndexesMap.at(usedTextFeatureIdx)].SetUsedInModel(true);
    }
}

void TObliviousTrees::TruncateTrees(size_t begin, size_t end) {
    //TODO(eermishkina): support non symmetric trees
    CB_ENSURE(IsOblivious(), "Truncate support only symmetric trees");
    CB_ENSURE(begin <= end, "begin tree index should be not greater than end tree index.");
    CB_ENSURE(end <= TreeSplits.size(), "end tree index should be not greater than tree count.");
    TObliviousTreeBuilder builder(FloatFeatures, CatFeatures, TextFeatures, ApproxDimension);
    const auto& leafOffsets = RuntimeData->TreeFirstLeafOffsets;
    for (size_t treeIdx = begin; treeIdx < end; ++treeIdx) {
        TVector<TModelSplit> modelSplits;
        for (int splitIdx = TreeStartOffsets[treeIdx];
             splitIdx < TreeStartOffsets[treeIdx] + TreeSizes[treeIdx];
             ++splitIdx)
        {
            modelSplits.push_back(RuntimeData->BinFeatures[TreeSplits[splitIdx]]);
        }
        TConstArrayRef<double> leafValuesRef(
            LeafValues.begin() + leafOffsets[treeIdx],
            LeafValues.begin() + leafOffsets[treeIdx] + ApproxDimension * (1u << TreeSizes[treeIdx])
        );
        builder.AddTree(
            modelSplits,
            leafValuesRef,
            LeafWeights.empty() ? TConstArrayRef<double>() : TConstArrayRef<double>(
                LeafWeights.begin() + leafOffsets[treeIdx] / ApproxDimension,
                LeafWeights.begin() + leafOffsets[treeIdx] / ApproxDimension + (1u << TreeSizes[treeIdx])
            )
        );
    }
    builder.Build(this);
}

flatbuffers::Offset<NCatBoostFbs::TObliviousTrees>
TObliviousTrees::FBSerialize(TModelPartsCachingSerializer& serializer) const {
    std::vector<flatbuffers::Offset<NCatBoostFbs::TCatFeature>> catFeaturesOffsets;
    for (const auto& catFeature : CatFeatures) {
        catFeaturesOffsets.push_back(catFeature.FBSerialize(serializer.FlatbufBuilder));
    }
    std::vector<flatbuffers::Offset<NCatBoostFbs::TFloatFeature>> floatFeaturesOffsets;
    for (const auto& floatFeature : FloatFeatures) {
        floatFeaturesOffsets.push_back(floatFeature.FBSerialize(serializer.FlatbufBuilder));
    }
    std::vector<flatbuffers::Offset<NCatBoostFbs::TTextFeature>> textFeaturesOffsets;
    for (const auto& textFeature : TextFeatures) {
        textFeaturesOffsets.push_back(textFeature.FBSerialize(serializer.FlatbufBuilder));
    }
    std::vector<flatbuffers::Offset<NCatBoostFbs::TEstimatedFeature>> estimatedFeaturesOffsets;
    for (const auto& estimatedFeature : EstimatedFeatures) {
        estimatedFeaturesOffsets.push_back(estimatedFeature.FBSerialize(serializer.FlatbufBuilder));
    }
    std::vector<flatbuffers::Offset<NCatBoostFbs::TOneHotFeature>> oneHotFeaturesOffsets;
    for (const auto& oneHotFeature : OneHotFeatures) {
        oneHotFeaturesOffsets.push_back(oneHotFeature.FBSerialize(serializer.FlatbufBuilder));
    }
    std::vector<flatbuffers::Offset<NCatBoostFbs::TCtrFeature>> ctrFeaturesOffsets;
    for (const auto& ctrFeature : CtrFeatures) {
        ctrFeaturesOffsets.push_back(ctrFeature.FBSerialize(serializer));
    }
    TVector<NCatBoostFbs::TNonSymmetricTreeStepNode> fbsNonSymmetricTreeStepNode;
    fbsNonSymmetricTreeStepNode.reserve(NonSymmetricStepNodes.size());
    for (const auto& nonSymmetricStep: NonSymmetricStepNodes) {
        fbsNonSymmetricTreeStepNode.emplace_back(NCatBoostFbs::TNonSymmetricTreeStepNode{
            nonSymmetricStep.LeftSubtreeDiff,
            nonSymmetricStep.RightSubtreeDiff
        });
    }
    return NCatBoostFbs::CreateTObliviousTreesDirect(
        serializer.FlatbufBuilder,
        ApproxDimension,
        &TreeSplits,
        &TreeSizes,
        &TreeStartOffsets,
        &catFeaturesOffsets,
        &floatFeaturesOffsets,
        &oneHotFeaturesOffsets,
        &ctrFeaturesOffsets,
        &LeafValues,
        &LeafWeights,
        &fbsNonSymmetricTreeStepNode,
        &NonSymmetricNodeIdToLeafId,
        &textFeaturesOffsets,
        &estimatedFeaturesOffsets
    );
}

void TObliviousTrees::UpdateRuntimeData() const {
    struct TFeatureSplitId {
        ui32 FeatureIdx = 0;
        ui32 SplitIdx = 0;
    };
    RuntimeData = TRuntimeData{}; // reset RuntimeData
    TVector<TFeatureSplitId> splitIds;
    auto& ref = RuntimeData.GetRef();

    ref.TreeFirstLeafOffsets.resize(TreeSizes.size());
    if (IsOblivious()) {
        size_t currentOffset = 0;
        for (size_t i = 0; i < TreeSizes.size(); ++i) {
            ref.TreeFirstLeafOffsets[i] = currentOffset;
            currentOffset += (1 << TreeSizes[i]) * ApproxDimension;
        }
    } else {
        for (size_t treeId = 0; treeId < TreeSizes.size(); ++treeId) {
            const int treeNodesStart = TreeStartOffsets[treeId];
            const int treeNodesEnd = treeNodesStart + TreeSizes[treeId];
            ui32 minLeafValueIndex = Max();
            ui32 maxLeafValueIndex = 0;
            ui32 valueNodeCount = 0; // count of nodes with values
            for (auto nodeIndex = treeNodesStart; nodeIndex < treeNodesEnd; ++nodeIndex) {
                const auto& node = NonSymmetricStepNodes[nodeIndex];
                if (node.LeftSubtreeDiff == 0|| node.RightSubtreeDiff == 0) {
                    const ui32 leafValueIndex = NonSymmetricNodeIdToLeafId[nodeIndex];
                    Y_ASSERT(leafValueIndex != Max<ui32>());
                    Y_VERIFY_DEBUG(
                        leafValueIndex % ApproxDimension == 0,
                        "Expect that leaf values are aligned."
                    );
                    minLeafValueIndex = Min(minLeafValueIndex, leafValueIndex);
                    maxLeafValueIndex = Max(maxLeafValueIndex, leafValueIndex);
                    ++valueNodeCount;
                }
            }
            Y_ASSERT(valueNodeCount > 0);
            Y_ASSERT(maxLeafValueIndex == minLeafValueIndex + (valueNodeCount - 1) * ApproxDimension);
            ref.TreeFirstLeafOffsets[treeId] = minLeafValueIndex;
        }
    }

    for (const auto& ctrFeature : CtrFeatures) {
        ref.UsedModelCtrs.push_back(ctrFeature.Ctr);
    }
    ref.EffectiveBinFeaturesBucketCount = 0;
    ref.UsedFloatFeaturesCount = 0;
    ref.UsedCatFeaturesCount = 0;
    ref.UsedTextFeaturesCount = 0;
    ref.UsedEstimatedFeaturesCount = 0;
    ref.MinimalSufficientFloatFeaturesVectorSize = 0;
    ref.MinimalSufficientCatFeaturesVectorSize = 0;
    for (const auto& feature : FloatFeatures) {
        if (!feature.UsedInModel()) {
            continue;
        }
        ++ref.UsedFloatFeaturesCount;
        ref.MinimalSufficientFloatFeaturesVectorSize = static_cast<size_t>(feature.Position.Index) + 1;
        for (int borderId = 0; borderId < feature.Borders.ysize(); ++borderId) {
            TFloatSplit fs{feature.Position.Index, feature.Borders[borderId]};
            ref.BinFeatures.emplace_back(fs);
            auto& bf = splitIds.emplace_back();
            bf.FeatureIdx = ref.EffectiveBinFeaturesBucketCount + borderId / MAX_VALUES_PER_BIN;
            bf.SplitIdx = (borderId % MAX_VALUES_PER_BIN) + 1;
        }
        ref.EffectiveBinFeaturesBucketCount
            += (feature.Borders.size() + MAX_VALUES_PER_BIN - 1) / MAX_VALUES_PER_BIN;
    }
    for (const auto& feature : CatFeatures) {
        if (!feature.UsedInModel()) {
            continue;
        }
        ++ref.UsedCatFeaturesCount;
        ref.MinimalSufficientCatFeaturesVectorSize = static_cast<size_t>(feature.Position.Index) + 1;
    }
    for (const auto& feature : TextFeatures) {
        if (!feature.UsedInModel()) {
            continue;
        }
        ++ref.UsedTextFeaturesCount;
        ref.MinimalSufficientTextFeaturesVectorSize = static_cast<size_t>(feature.Position.Index) + 1;
    }
    for (const auto& feature : EstimatedFeatures) {
        for (int borderId = 0; borderId < feature.Borders.ysize(); ++borderId) {
            TEstimatedFeatureSplit split{
                feature.SourceFeatureIndex,
                feature.CalcerId,
                feature.LocalIndex,
                feature.Borders[borderId]
            };
            ref.BinFeatures.emplace_back(split);
            auto& bf = splitIds.emplace_back();
            bf.FeatureIdx = ref.EffectiveBinFeaturesBucketCount + borderId / MAX_VALUES_PER_BIN;
            bf.SplitIdx = (borderId % MAX_VALUES_PER_BIN) + 1;
        }
        ref.EffectiveBinFeaturesBucketCount +=
            (feature.Borders.size() + MAX_VALUES_PER_BIN - 1) / MAX_VALUES_PER_BIN;
        ++ref.UsedEstimatedFeaturesCount;
    }
    for (size_t i = 0; i < OneHotFeatures.size(); ++i) {
        const auto& feature = OneHotFeatures[i];
        for (int valueId = 0; valueId < feature.Values.ysize(); ++valueId) {
            TOneHotSplit oh{feature.CatFeatureIndex, feature.Values[valueId]};
            ref.BinFeatures.emplace_back(oh);
            auto& bf = splitIds.emplace_back();
            bf.FeatureIdx = ref.EffectiveBinFeaturesBucketCount + valueId / MAX_VALUES_PER_BIN;
            bf.SplitIdx = (valueId % MAX_VALUES_PER_BIN) + 1;
        }
        ref.EffectiveBinFeaturesBucketCount
            += (feature.Values.size() + MAX_VALUES_PER_BIN - 1) / MAX_VALUES_PER_BIN;
    }
    for (size_t i = 0; i < CtrFeatures.size(); ++i) {
        const auto& feature = CtrFeatures[i];
        if (i > 0) {
            Y_ASSERT(CtrFeatures[i - 1] < feature);
        }
        for (int borderId = 0; borderId < feature.Borders.ysize(); ++borderId) {
            TModelCtrSplit ctrSplit;
            ctrSplit.Ctr = feature.Ctr;
            ctrSplit.Border = feature.Borders[borderId];
            ref.BinFeatures.emplace_back(std::move(ctrSplit));
            auto& bf = splitIds.emplace_back();
            bf.FeatureIdx = ref.EffectiveBinFeaturesBucketCount + borderId / MAX_VALUES_PER_BIN;
            bf.SplitIdx = (borderId % MAX_VALUES_PER_BIN) + 1;
        }
        ref.EffectiveBinFeaturesBucketCount
            += (feature.Borders.size() + MAX_VALUES_PER_BIN - 1) / MAX_VALUES_PER_BIN;
    }
    for (const auto& binSplit : TreeSplits) {
        const auto& feature = ref.BinFeatures[binSplit];
        const auto& featureIndex = splitIds[binSplit];
        Y_ENSURE(
            featureIndex.FeatureIdx <= 0xffff,
            "To many features in model, ask catboost team for support"
        );
        TRepackedBin rb;
        rb.FeatureIndex = featureIndex.FeatureIdx;
        if (feature.Type != ESplitType::OneHotFeature) {
            rb.SplitIdx = featureIndex.SplitIdx;
        } else {
            rb.XorMask = ((~featureIndex.SplitIdx) & 0xff);
            rb.SplitIdx = 0xff;
        }
        ref.RepackedBins.push_back(rb);
    }
}

void TObliviousTrees::DropUnusedFeatures() {
    EraseIf(FloatFeatures, [](const TFloatFeature& feature) { return !feature.UsedInModel();});
    EraseIf(CatFeatures, [](const TCatFeature& feature) { return !feature.UsedInModel(); });
    EraseIf(TextFeatures, [](const TTextFeature& feature) { return !feature.UsedInModel(); });
    UpdateRuntimeData();
}

void TObliviousTrees::ConvertObliviousToAsymmetric() {
    if (!IsOblivious()) {
        return;
    }
    TVector<int> treeSplits;
    TVector<int> treeSizes;
    TVector<int> treeStartOffsets;
    TVector<TNonSymmetricTreeStepNode> nonSymmetricStepNodes;
    TVector<ui32> nonSymmetricNodeIdToLeafId;

    size_t leafStartOffset = 0;
    for (size_t treeId = 0; treeId < TreeSizes.size(); ++treeId) {
        size_t treeSize = 0;
        treeStartOffsets.push_back(treeSplits.size());
        for (int depth = 0; depth < TreeSizes[treeId]; ++depth) {
            const auto split = TreeSplits[TreeStartOffsets[treeId] + TreeSizes[treeId] - 1 - depth];
            for (size_t cloneId = 0; cloneId < (1u << depth); ++cloneId) {
                treeSplits.push_back(split);
                nonSymmetricNodeIdToLeafId.push_back(Max<ui32>());
                nonSymmetricStepNodes.emplace_back(TNonSymmetricTreeStepNode{static_cast<ui16>(treeSize + 1), static_cast<ui16>(treeSize + 2)});
                ++treeSize;
            }
        }
        for (size_t cloneId = 0; cloneId < (1u << TreeSizes[treeId]); ++cloneId) {
            treeSplits.push_back(0);
            nonSymmetricNodeIdToLeafId.push_back((leafStartOffset + cloneId) * ApproxDimension);
            nonSymmetricStepNodes.emplace_back(TNonSymmetricTreeStepNode{0, 0});
            ++treeSize;
        }
        leafStartOffset += (1u << TreeSizes[treeId]);
        treeSizes.push_back(treeSize);
    }
    TreeSplits = std::move(treeSplits);
    TreeSizes = std::move(treeSizes);
    TreeStartOffsets = std::move(treeStartOffsets);
    NonSymmetricStepNodes = std::move(nonSymmetricStepNodes);
    NonSymmetricNodeIdToLeafId = std::move(nonSymmetricNodeIdToLeafId);
    UpdateRuntimeData();
}

TVector<ui32> TObliviousTrees::GetTreeLeafCounts() const {
    const auto& firstLeafOfsets = GetFirstLeafOffsets();
    Y_ASSERT(IsSorted(firstLeafOfsets.begin(), firstLeafOfsets.end()));
    TVector<ui32> treeLeafCounts;
    treeLeafCounts.reserve(GetTreeCount());
    for (size_t treeNum = 0; treeNum < GetTreeCount(); ++treeNum) {
        const size_t currTreeLeafValuesEnd = (
            treeNum + 1 < GetTreeCount()
            ? firstLeafOfsets[treeNum + 1]
            : LeafValues.size()
        );
        const size_t currTreeLeafValuesCount = currTreeLeafValuesEnd - firstLeafOfsets[treeNum];
        Y_ASSERT(currTreeLeafValuesCount % ApproxDimension == 0);
        treeLeafCounts.push_back(currTreeLeafValuesCount / ApproxDimension);
    }
    return treeLeafCounts;
}

void TObliviousTrees::AddNumberToAllTreeLeafValues(ui32 treeId, double numberToAdd) {
    const auto& firstLeafOfsets = GetFirstLeafOffsets();
    if (numberToAdd == 0 || firstLeafOfsets.size() <= treeId) {
        return;
    }
    ui32 begin = firstLeafOfsets[treeId];
    ui32 end = treeId + 1 == firstLeafOfsets.size() ? LeafValues.size() : firstLeafOfsets[treeId + 1];
    for (ui32 i = begin; i < end; ++i) {
        LeafValues[i] += numberToAdd;
    }
}

void TObliviousTrees::FBDeserialize(const NCatBoostFbs::TObliviousTrees* fbObj) {
    ApproxDimension = fbObj->ApproxDimension();
    if (fbObj->TreeSplits()) {
        TreeSplits.assign(fbObj->TreeSplits()->begin(), fbObj->TreeSplits()->end());
    }
    if (fbObj->TreeSizes()) {
        TreeSizes.assign(fbObj->TreeSizes()->begin(), fbObj->TreeSizes()->end());
    }
    if (fbObj->TreeStartOffsets()) {
        TreeStartOffsets.assign(fbObj->TreeStartOffsets()->begin(), fbObj->TreeStartOffsets()->end());
    }

    if (fbObj->LeafValues()) {
        LeafValues.assign(fbObj->LeafValues()->begin(), fbObj->LeafValues()->end());
    }
    if (fbObj->NonSymmetricStepNodes()) {
        NonSymmetricStepNodes.resize(fbObj->NonSymmetricStepNodes()->size());
        std::copy(
            fbObj->NonSymmetricStepNodes()->begin(),
            fbObj->NonSymmetricStepNodes()->end(),
            NonSymmetricStepNodes.begin()
        );
    }
    if (fbObj->NonSymmetricNodeIdToLeafId()) {
        NonSymmetricNodeIdToLeafId.assign(
            fbObj->NonSymmetricNodeIdToLeafId()->begin(), fbObj->NonSymmetricNodeIdToLeafId()->end()
        );
    }

#define FBS_ARRAY_DESERIALIZER(var) \
        if (fbObj->var()) {\
            var.resize(fbObj->var()->size());\
            for (size_t i = 0; i < fbObj->var()->size(); ++i) {\
                var[i].FBDeserialize(fbObj->var()->Get(i));\
            }\
        }
    FBS_ARRAY_DESERIALIZER(CatFeatures)
    FBS_ARRAY_DESERIALIZER(FloatFeatures)
    FBS_ARRAY_DESERIALIZER(TextFeatures)
    FBS_ARRAY_DESERIALIZER(EstimatedFeatures)
    FBS_ARRAY_DESERIALIZER(OneHotFeatures)
    FBS_ARRAY_DESERIALIZER(CtrFeatures)
#undef FBS_ARRAY_DESERIALIZER
    if (fbObj->LeafWeights() && fbObj->LeafWeights()->size() > 0) {
            LeafWeights.assign(fbObj->LeafWeights()->begin(), fbObj->LeafWeights()->end());
    }
}

void TFullModel::CalcFlat(
    TConstArrayRef<TConstArrayRef<float>> features,
    size_t treeStart,
    size_t treeEnd,
    TArrayRef<double> results,
    const TFeatureLayout* featureInfo) const {
    GetCurrentEvaluator()->CalcFlat(features, treeStart, treeEnd, results, featureInfo);
}

void TFullModel::CalcFlatSingle(
    TConstArrayRef<float> features,
    size_t treeStart,
    size_t treeEnd,
    TArrayRef<double> results,
    const TFeatureLayout* featureInfo) const {
    GetCurrentEvaluator()->CalcFlatSingle(features, treeStart, treeEnd, results, featureInfo);
}

void TFullModel::CalcFlatTransposed(
    TConstArrayRef<TConstArrayRef<float>> transposedFeatures,
    size_t treeStart,
    size_t treeEnd,
    TArrayRef<double> results,
    const TFeatureLayout* featureInfo) const {
    GetCurrentEvaluator()->CalcFlatTransposed(transposedFeatures, treeStart, treeEnd, results, featureInfo);
}

void TFullModel::Calc(
    TConstArrayRef<TConstArrayRef<float>> floatFeatures,
    TConstArrayRef<TConstArrayRef<int>> catFeatures,
    size_t treeStart,
    size_t treeEnd,
    TArrayRef<double> results,
    const TFeatureLayout* featureInfo
) const {
    GetCurrentEvaluator()->Calc(floatFeatures, catFeatures, treeStart, treeEnd, results, featureInfo);
}

void TFullModel::Calc(
    TConstArrayRef<TConstArrayRef<float>> floatFeatures,
    TConstArrayRef<TVector<TStringBuf>> catFeatures,
    size_t treeStart,
    size_t treeEnd,
    TArrayRef<double> results,
    const TFeatureLayout* featureInfo
) const {
    TVector<TConstArrayRef<TStringBuf>> stringbufVecRefs{catFeatures.begin(), catFeatures.end()};
    GetCurrentEvaluator()->Calc(floatFeatures, stringbufVecRefs, treeStart, treeEnd, results, featureInfo);
}

void TFullModel::Calc(
    TConstArrayRef<TConstArrayRef<float>> floatFeatures,
    TConstArrayRef<TVector<TStringBuf>> catFeatures,
    TConstArrayRef<TVector<TStringBuf>> textFeatures,
    size_t treeStart,
    size_t treeEnd,
    TArrayRef<double> results,
    const TFeatureLayout* featureInfo
) const {
    TVector<TConstArrayRef<TStringBuf>> stringbufCatVecRefs{catFeatures.begin(), catFeatures.end()};
    TVector<TConstArrayRef<TStringBuf>> stringbufTextVecRefs{textFeatures.begin(), textFeatures.end()};
    GetCurrentEvaluator()->Calc(floatFeatures, stringbufCatVecRefs, stringbufTextVecRefs, treeStart, treeEnd, results, featureInfo);
}

void TFullModel::CalcLeafIndexesSingle(
    TConstArrayRef<float> floatFeatures,
    TConstArrayRef<TStringBuf> catFeatures,
    size_t treeStart,
    size_t treeEnd,
    TArrayRef<ui32> indexes,
    const TFeatureLayout* featureInfo
) const {
    GetCurrentEvaluator()->CalcLeafIndexesSingle(floatFeatures, catFeatures, treeStart, treeEnd, indexes, featureInfo);
}

void TFullModel::CalcLeafIndexes(
    TConstArrayRef<TConstArrayRef<float>> floatFeatures,
    TConstArrayRef<TConstArrayRef<TStringBuf>> catFeatures,
    size_t treeStart,
    size_t treeEnd,
    TArrayRef<ui32> indexes,
    const TFeatureLayout* featureInfo
) const {
    GetCurrentEvaluator()->CalcLeafIndexes(floatFeatures, catFeatures, treeStart, treeEnd, indexes, featureInfo);
}


void TFullModel::Save(IOutputStream* s) const {
    using namespace flatbuffers;
    using namespace NCatBoostFbs;
    ::Save(s, GetModelFormatDescriptor());
    TModelPartsCachingSerializer serializer;
    auto obliviousTreesOffset = ObliviousTrees->FBSerialize(serializer);
    std::vector<flatbuffers::Offset<TKeyValue>> infoMap;
    for (const auto& key_value : ModelInfo) {
        auto keyValueOffset = CreateTKeyValue(
            serializer.FlatbufBuilder,
            serializer.FlatbufBuilder.CreateString(
                key_value.first.c_str(),
                key_value.first.size()
            ),
            serializer.FlatbufBuilder.CreateString(
                key_value.second.c_str(),
                key_value.second.size()
            )
        );
        infoMap.push_back(keyValueOffset);
    }
    std::vector<flatbuffers::Offset<flatbuffers::String>> modelPartIds;
    if (!!CtrProvider && CtrProvider->IsSerializable()) {
        modelPartIds.push_back(serializer.FlatbufBuilder.CreateString(CtrProvider->ModelPartIdentifier()));
    }
    if (!!TextProcessingCollection) {
        modelPartIds.push_back(serializer.FlatbufBuilder.CreateString(TextProcessingCollection->GetStringIdentifier()));
    }
    auto coreOffset = CreateTModelCoreDirect(
        serializer.FlatbufBuilder,
        CURRENT_CORE_FORMAT_STRING,
        obliviousTreesOffset,
        infoMap.empty() ? nullptr : &infoMap,
        modelPartIds.empty() ? nullptr : &modelPartIds
    );
    serializer.FlatbufBuilder.Finish(coreOffset);
    SaveSize(s, serializer.FlatbufBuilder.GetSize());
    s->Write(serializer.FlatbufBuilder.GetBufferPointer(), serializer.FlatbufBuilder.GetSize());
    if (!!CtrProvider && CtrProvider->IsSerializable()) {
        CtrProvider->Save(s);
    }
    if (!!TextProcessingCollection) {
        TextProcessingCollection->Save(s);
    }
}

void TFullModel::Load(IInputStream* s) {
    using namespace flatbuffers;
    using namespace NCatBoostFbs;
    ui32 fileDescriptor;
    ::Load(s, fileDescriptor);
    CB_ENSURE(fileDescriptor == GetModelFormatDescriptor(), "Incorrect model file descriptor");
    auto coreSize = ::LoadSize(s);
    TArrayHolder<ui8> arrayHolder = new ui8[coreSize];
    s->LoadOrFail(arrayHolder.Get(), coreSize);

    {
        flatbuffers::Verifier verifier(arrayHolder.Get(), coreSize);
        CB_ENSURE(VerifyTModelCoreBuffer(verifier), "Flatbuffers model verification failed");
    }
    auto fbModelCore = GetTModelCore(arrayHolder.Get());
    CB_ENSURE(
        fbModelCore->FormatVersion() && fbModelCore->FormatVersion()->str() == CURRENT_CORE_FORMAT_STRING,
        "Unsupported model format: " << fbModelCore->FormatVersion()->str()
    );
    if (fbModelCore->ObliviousTrees()) {
        ObliviousTrees.GetMutable()->FBDeserialize(fbModelCore->ObliviousTrees());
    }
    ModelInfo.clear();
    if (fbModelCore->InfoMap()) {
        for (auto keyVal : *fbModelCore->InfoMap()) {
            ModelInfo[keyVal->Key()->str()] = keyVal->Value()->str();
        }
    }
    TVector<TString> modelParts;
    if (fbModelCore->ModelPartIds()) {
        for (auto part : *fbModelCore->ModelPartIds()) {
            modelParts.emplace_back(part->str());
        }
    }
    if (!modelParts.empty()) {
        for (const auto& modelPartId : modelParts) {
            if (modelPartId == TStaticCtrProvider::ModelPartId()) {
                CtrProvider = new TStaticCtrProvider;
                CtrProvider->Load(s);
            } else if (modelPartId == NCB::TTextProcessingCollection::GetStringIdentifier()) {
                TextProcessingCollection = new NCB::TTextProcessingCollection();
                TextProcessingCollection->Load(s);
            } else {
                CB_ENSURE(
                    false,
                    "Got unknown partId = " << modelPartId << " via deserialization"
                        << "only static ctr and text processing collection model parts are supported"
                );
            }
        }
    }
    UpdateDynamicData();
}

void TFullModel::UpdateDynamicData() {
    ObliviousTrees->UpdateRuntimeData();
    if (CtrProvider) {
        CtrProvider->SetupBinFeatureIndexes(
            ObliviousTrees->GetFloatFeatures(),
            ObliviousTrees->GetOneHotFeatures(),
            ObliviousTrees->GetCatFeatures());
    }
    with_lock(CurrentEvaluatorLock) {
        Evaluator.Reset();
    }
}

TVector<TString> GetModelUsedFeaturesNames(const TFullModel& model) {
    TVector<int> featuresIdxs;
    TVector<TString> featuresNames;
    const TObliviousTrees& forest = *model.ObliviousTrees;

    for (const TFloatFeature& feature : forest.GetFloatFeatures()) {
        featuresIdxs.push_back(feature.Position.FlatIndex);
        featuresNames.push_back(
            feature.FeatureId == "" ? ToString(feature.Position.FlatIndex) : feature.FeatureId
        );
    }
    for (const TCatFeature& feature : forest.GetCatFeatures()) {
        featuresIdxs.push_back(feature.Position.FlatIndex);
        featuresNames.push_back(
            feature.FeatureId == "" ? ToString(feature.Position.FlatIndex) : feature.FeatureId
        );
    }

    TVector<int> featuresOrder(featuresIdxs.size());
    Iota(featuresOrder.begin(), featuresOrder.end(), 0);
    Sort(featuresOrder.begin(), featuresOrder.end(),
        [featuresIdxs](int index1, int index2) {
            return featuresIdxs[index1] < featuresIdxs[index2];
        }
    );

    TVector<TString> result(featuresNames.size());
    for (int featureIdx = 0; featureIdx < featuresNames.ysize(); ++featureIdx) {
        result[featureIdx] = featuresNames[featuresOrder[featureIdx]];
    }
    return result;
}

void SetModelExternalFeatureNames(const TVector<TString>& featureNames, TFullModel* model) {
    TObliviousTrees& forest = *(model->ObliviousTrees.GetMutable());
    CB_ENSURE(
        (forest.GetFloatFeatures().empty() || featureNames.ysize() > forest.GetFloatFeatures().back().Position.FlatIndex) &&
        (forest.GetCatFeatures().empty() || featureNames.ysize() > forest.GetCatFeatures().back().Position.FlatIndex),
        "Features in model not corresponds to features names array length not correspond");
    forest.ApplyFeatureNames(featureNames);
}

static TMaybe<NCatboostOptions::TLossDescription> GetLossDescription(const TFullModel& model) {
    TMaybe<NCatboostOptions::TLossDescription> lossDescription;
    if (model.ModelInfo.contains("loss_function")) {
        lossDescription.ConstructInPlace();
        lossDescription->Load(ReadTJsonValue(model.ModelInfo.at("loss_function")));
    }
    if (model.ModelInfo.contains("params")) {
        const auto& params = ReadTJsonValue(model.ModelInfo.at("params"));
        if (params.Has("loss_function")) {
            lossDescription.ConstructInPlace();
            lossDescription->Load(params["loss_function"]);
        }
    }
    return lossDescription;
}

TString TFullModel::GetLossFunctionName() const {
    const TMaybe<NCatboostOptions::TLossDescription> lossDescription = GetLossDescription(*this);
    if (lossDescription.Defined()) {
        return ToString(lossDescription->GetLossFunction());
    }
    return {};
}

inline TVector<TString> ExtractClassNamesFromJsonArray(const NJson::TJsonValue& arr) {
    TVector<TString> classNames;
    for (const auto& token : arr.GetArraySafe()) {
        classNames.push_back(token.GetStringSafe());
    }
    return classNames;
}

TVector<TString> TFullModel::GetModelClassNames() const {
    TVector<TString> classNames;
    if (ModelInfo.contains("multiclass_params")) {
        TMulticlassLabelOptions multiclassOptions;
        multiclassOptions.Load(ReadTJsonValue(ModelInfo.at("multiclass_params")));
        if (multiclassOptions.ClassNames.IsSet()) {
            classNames = multiclassOptions.ClassNames.Get();
            return classNames;
        } else if (multiclassOptions.ClassToLabel.IsSet()) {
            classNames.reserve(multiclassOptions.ClassToLabel->size());
            for (float label : multiclassOptions.ClassToLabel.Get()) {
                classNames.push_back(ToString(ui32(label)));
            }
            return classNames;
        }
    }
    if (ModelInfo.contains("params")) {
        const TString& modelInfoParams = ModelInfo.at("params");
        NJson::TJsonValue paramsJson = ReadTJsonValue(modelInfoParams);
        if (paramsJson.Has("data_processing_options")
            && paramsJson["data_processing_options"].Has("class_names")) {
            classNames = ExtractClassNamesFromJsonArray(paramsJson["data_processing_options"]["class_names"]);
            if (!classNames.empty()) {
                return classNames;
            }
        }
    }

    const size_t dimensionsCount = GetDimensionsCount();
    if (dimensionsCount > 1) {
        classNames.reserve(dimensionsCount);
        for (auto i : xrange(dimensionsCount)) {
            classNames.push_back(ToString(i));
        }
    } else {
        const TMaybe<NCatboostOptions::TLossDescription> lossDescription = GetLossDescription(*this);
        if (lossDescription.Defined() && IsClassificationObjective(lossDescription->GetLossFunction())) {
            classNames = TVector<TString>{"0", "1"};
        }
    }

    return classNames;
}

void TFullModel::UpdateEstimatedFeaturesIndices(TVector<TEstimatedFeature>&& newEstimatedFeatures) {
    CB_ENSURE(
        TextProcessingCollection,
        "UpdateEstimatedFeatureIndices called when TextProcessingCollection is not defined"
    );

    ObliviousTrees.GetMutable()->SetEstimatedFeatures(std::move(newEstimatedFeatures));
    ObliviousTrees->UpdateRuntimeData();
}

namespace {
    struct TUnknownFeature {};

    struct TFlatFeature {

        TVariant<TUnknownFeature, TFloatFeature, TCatFeature> FeatureVariant;

    public:
        TFlatFeature() = default;

        template <class TFeatureType>
        void SetOrCheck(const TFeatureType& other) {
            if (HoldsAlternative<TUnknownFeature>(FeatureVariant)) {
                FeatureVariant = other;
            }
            CB_ENSURE(HoldsAlternative<TFeatureType>(FeatureVariant),
                "Feature type mismatch: Categorical != Float for flat feature index: " <<
                other.Position.FlatIndex
            );
            TFeatureType& feature = Get<TFeatureType>(FeatureVariant);
            CB_ENSURE(feature.Position.FlatIndex == other.Position.FlatIndex);
            CB_ENSURE(
                feature.Position.Index == other.Position.Index,
                "Internal feature index mismatch: " << feature.Position.Index << " != "
                << other.Position.Index << " flat feature index: " << feature.Position.FlatIndex
            );
            CB_ENSURE(
                feature.FeatureId.empty() || feature.FeatureId == other.FeatureId,
                "Feature name mismatch: " << feature.FeatureId << " != " << other.FeatureId
                << " flat feature index: " << feature.Position.FlatIndex
            );
            feature.FeatureId = other.FeatureId;
            if constexpr (std::is_same_v<TFeatureType, TFloatFeature>) {
                CB_ENSURE(
                    feature.NanValueTreatment == other.NanValueTreatment,
                    "Nan value treatment differs: " << (int) feature.NanValueTreatment << " != " <<
                    (int) other.NanValueTreatment
                );
                feature.HasNans |= other.HasNans;
            }
        }
    };

    struct TFlatFeatureMergerVisitor {
        void operator()(TUnknownFeature&) {
        }
        void operator()(TFloatFeature& s) {
            MergedFloatFeatures.push_back(s);
        }
        void operator()(TCatFeature& x) {
            MergedCatFeatures.push_back(x);
        }
        TVector<TFloatFeature> MergedFloatFeatures;
        TVector<TCatFeature> MergedCatFeatures;
    };
}

static void StreamModelTreesToBuilder(
    const TObliviousTrees& trees,
    double leafMultiplier,
    TObliviousTreeBuilder* builder,
    bool streamLeafWeights)
{
    const auto& binFeatures = trees.GetBinFeatures();
    const auto& leafOffsets = trees.GetFirstLeafOffsets();
    for (size_t treeIdx = 0; treeIdx < trees.GetTreeSizes().size(); ++treeIdx) {
        TVector<TModelSplit> modelSplits;
        for (int splitIdx = trees.GetTreeStartOffsets()[treeIdx];
             splitIdx < trees.GetTreeStartOffsets()[treeIdx] + trees.GetTreeSizes()[treeIdx];
             ++splitIdx)
        {
            modelSplits.push_back(binFeatures[trees.GetTreeSplits()[splitIdx]]);
        }
        if (leafMultiplier == 1.0) {
            TConstArrayRef<double> leafValuesRef(
                trees.GetLeafValues().begin() + leafOffsets[treeIdx],
                trees.GetLeafValues().begin() + leafOffsets[treeIdx]
                    + trees.GetDimensionsCount() * (1u << trees.GetTreeSizes()[treeIdx])
            );
            builder->AddTree(
                modelSplits,
                leafValuesRef,
                !streamLeafWeights ? TConstArrayRef<double>() : TConstArrayRef<double>(
                    trees.GetLeafWeights().begin() + leafOffsets[treeIdx] / trees.GetDimensionsCount(),
                    trees.GetLeafWeights().begin() + leafOffsets[treeIdx] / trees.GetDimensionsCount()
                        + (1u << trees.GetTreeSizes()[treeIdx])
                )
            );
        } else {
            TVector<double> leafValues(
                trees.GetLeafValues().begin() + leafOffsets[treeIdx],
                trees.GetLeafValues().begin() + leafOffsets[treeIdx]
                    + trees.GetDimensionsCount() * (1u << trees.GetTreeSizes()[treeIdx])
            );
            for (auto& leafValue: leafValues) {
                leafValue *= leafMultiplier;
            }
            builder->AddTree(
                modelSplits,
                leafValues,
                !streamLeafWeights ? TConstArrayRef<double>() : TConstArrayRef<double>(
                    trees.GetLeafWeights().begin() + leafOffsets[treeIdx] / trees.GetDimensionsCount(),
                    (1u << trees.GetTreeSizes()[treeIdx])
                )
            );
        }
    }
}

static void SumModelsParams(
    const TVector<const TFullModel*> modelVector,
    THashMap<TString, TString>* modelInfo
) {
    const char* multiClassParamsName = "multiclass_params";
    if (modelVector.back()->ModelInfo.contains(multiClassParamsName)) {
        const TString multiClassParams = modelVector.back()->ModelInfo.at(multiClassParamsName);
        const bool allMultiClassParamsAreSame = AllOf(
            modelVector,
            [&](const TFullModel* model) {
                return model->ModelInfo.contains(multiClassParamsName) &&
                       model->ModelInfo.at(multiClassParamsName) == multiClassParams;
            }
        );

        CB_ENSURE(
            allMultiClassParamsAreSame,
            "Cannot sum models with different multiclass_params"
        );
        (*modelInfo)[multiClassParamsName] = multiClassParams;
    }

    const auto lossDescription = GetLossDescription(*modelVector.back());
    if (lossDescription.Defined()) {
        const bool allLossesAreSame = AllOf(
            modelVector,
            [&lossDescription](const TFullModel* model) {
                const auto currentLossDescription = GetLossDescription(*model);
                return currentLossDescription.Defined() && ((*currentLossDescription) == (*lossDescription));
            }
        );

        if (allLossesAreSame) {
            NJson::TJsonValue lossDescriptionJson;
            lossDescription->Save(&lossDescriptionJson);
            (*modelInfo)["loss_function"] = ToString(lossDescriptionJson);
        }
    }

}

TFullModel SumModels(
    const TVector<const TFullModel*> modelVector,
    const TVector<double>& weights,
    ECtrTableMergePolicy ctrMergePolicy)
{
    CB_ENSURE(!modelVector.empty(), "empty model vector unexpected");
    CB_ENSURE(modelVector.size() == weights.size());
    const auto approxDimension = modelVector.back()->GetDimensionsCount();
    size_t maxFlatFeatureVectorSize = 0;
    TVector<TIntrusivePtr<ICtrProvider>> ctrProviders;
    bool allModelsHaveLeafWeights = true;
    bool someModelHasLeafWeights = false;
    for (const auto& model : modelVector) {
        Y_ASSERT(model != nullptr);
        //TODO(eermishkina): support non symmetric trees
        CB_ENSURE(model->IsOblivious(), "Models summation supported only for symmetric trees");
        CB_ENSURE(
            model->ObliviousTrees->GetTextFeatures().empty(),
            "Models summation is not supported for models with text features"
        );
        CB_ENSURE(
            model->GetDimensionsCount() == approxDimension,
            "Approx dimensions don't match: " << model->GetDimensionsCount() << " != "
            << approxDimension
        );
        maxFlatFeatureVectorSize = Max(
            maxFlatFeatureVectorSize,
            model->ObliviousTrees->GetFlatFeatureVectorExpectedSize()
        );
        ctrProviders.push_back(model->CtrProvider);
        // empty model does not disable LeafWeights:
        if (model->ObliviousTrees->GetLeafWeights().size() < model->GetTreeCount()) {
            allModelsHaveLeafWeights = false;
        }
        if (!model->ObliviousTrees->GetLeafWeights().empty()) {
            someModelHasLeafWeights = true;
        }
    }
    if (!allModelsHaveLeafWeights && someModelHasLeafWeights) {
        CATBOOST_WARNING_LOG << "Leaf weights for some models are ignored " <<
        "because not all models have leaf weights" << Endl;
    }
    TVector<TFlatFeature> flatFeatureInfoVector(maxFlatFeatureVectorSize);
    for (const auto& model : modelVector) {
        for (const auto& floatFeature : model->ObliviousTrees->GetFloatFeatures()) {
            flatFeatureInfoVector[floatFeature.Position.FlatIndex].SetOrCheck(floatFeature);
        }
        for (const auto& catFeature : model->ObliviousTrees->GetCatFeatures()) {
            flatFeatureInfoVector[catFeature.Position.FlatIndex].SetOrCheck(catFeature);
        }
    }
    TFlatFeatureMergerVisitor merger;
    for (auto& flatFeature: flatFeatureInfoVector) {
        Visit(merger, flatFeature.FeatureVariant);
    }
    TObliviousTreeBuilder builder(merger.MergedFloatFeatures, merger.MergedCatFeatures, {}, approxDimension);
    for (const auto modelId : xrange(modelVector.size())) {
        StreamModelTreesToBuilder(
            *modelVector[modelId]->ObliviousTrees,
            weights[modelId],
            &builder,
            allModelsHaveLeafWeights
        );
    }
    TFullModel result;
    builder.Build(result.ObliviousTrees.GetMutable());
    for (const auto modelIdx : xrange(modelVector.size())) {
        TStringBuilder keyPrefix;
        keyPrefix << "model" << modelIdx << ":";
        for (const auto& [key, value]: modelVector[modelIdx]->ModelInfo) {
            result.ModelInfo[keyPrefix + key] = value;
        }
    }
    result.CtrProvider = MergeCtrProvidersData(ctrProviders, ctrMergePolicy);
    result.UpdateDynamicData();
    result.ModelInfo["model_guid"] = CreateGuidAsString();
    SumModelsParams(modelVector, &result.ModelInfo);
    return result;
}

void SaveModelBorders(
    const TString& file,
    const TFullModel& model) {

    TOFStream out(file);

    for (const auto& feature : model.ObliviousTrees->GetFloatFeatures()) {
        NCB::OutputFeatureBorders(
            feature.Position.FlatIndex,
            feature.Borders,
            NanValueTreatmentToNanMode(feature.NanValueTreatment),
            &out
        );
    }
}

DEFINE_DUMPER(TRepackedBin, FeatureIndex, XorMask, SplitIdx)

DEFINE_DUMPER(TNonSymmetricTreeStepNode, LeftSubtreeDiff, RightSubtreeDiff)

DEFINE_DUMPER(
    TObliviousTrees::TRuntimeData,
    UsedFloatFeaturesCount,
    UsedCatFeaturesCount,
    MinimalSufficientFloatFeaturesVectorSize,
    MinimalSufficientCatFeaturesVectorSize,
    UsedModelCtrs,
    BinFeatures,
    RepackedBins,
    EffectiveBinFeaturesBucketCount,
    TreeFirstLeafOffsets
)

//DEFINE_DUMPER(TObliviousTrees),
//    TreeSplits, TreeSizes,
//    TreeStartOffsets, NonSymmetricStepNodes,
//    NonSymmetricNodeIdToLeafId, LeafValues);

TNonSymmetricTreeStepNode& TNonSymmetricTreeStepNode::operator=(const NCatBoostFbs::TNonSymmetricTreeStepNode* stepNode) {
    LeftSubtreeDiff = stepNode->LeftSubtreeDiff();
    RightSubtreeDiff = stepNode->RightSubtreeDiff();
    return *this;
}
