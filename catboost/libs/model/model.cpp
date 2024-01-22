#include "model.h"

#include "evaluation_interface.h"

#include "ctr_helpers.h"
#include "flatbuffers_serializer_helper.h"
#include "model_import_interface.h"
#include "model_build_helper.h"
#include "static_ctr_provider.h"

#include <catboost/libs/model/flatbuffers/model.fbs.h>

#include <catboost/libs/cat_feature/cat_feature.h>
#include <catboost/libs/helpers/borders_io.h>
#include <catboost/libs/logging/logging.h>

#include <catboost/private/libs/options/enum_helpers.h>
#include <catboost/private/libs/options/json_helper.h>
#include <catboost/private/libs/options/loss_description.h>
#include <catboost/private/libs/options/class_label_options.h>

#include <library/cpp/json/json_reader.h>
#include <library/cpp/dbg_output/dump.h>
#include <library/cpp/dbg_output/auto.h>

#include <util/generic/algorithm.h>
#include <util/generic/cast.h>
#include <util/generic/fwd.h>
#include <util/generic/guid.h>
#include <util/generic/variant.h>
#include <util/generic/xrange.h>
#include <util/generic/ylimits.h>
#include <util/generic/ymath.h>
#include <util/string/builder.h>
#include <util/stream/str.h>


using namespace NCB;


static const char MODEL_FILE_DESCRIPTOR_CHARS[4] = {'C', 'B', 'M', '1'};

static void ReferenceMainFactoryRegistrators() {
    // We HAVE TO manually reference some pointers to make factory registrators work. Blessed static linking!
    CB_ENSURE(NCB::NModelEvaluation::CPUEvaluationBackendRegistratorPointer);
    CB_ENSURE(NCB::BinaryModelLoaderRegistratorPointer);
}

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

static void CheckFormat(EModelType format) {
    ReferenceMainFactoryRegistrators();
    CB_ENSURE(
        NCB::TModelLoaderFactory::Has(format),
        "Model format " << format << " deserialization not supported or missing. Link with catboost/libs/model/model_export if you need CoreML or JSON"
    );
}

TFullModel ReadModel(const TString& modelFile, EModelType format) {
    CheckFormat(format);
    THolder<NCB::IModelLoader> modelLoader(NCB::TModelLoaderFactory::Construct(format));
    return modelLoader->ReadModel(modelFile);
}

TFullModel ReadModel(const void* binaryBuffer, size_t binaryBufferSize, EModelType format) {
    CheckFormat(format);
    THolder<NCB::IModelLoader> modelLoader(NCB::TModelLoaderFactory::Construct(format));
    return modelLoader->ReadModel(binaryBuffer, binaryBufferSize);
}

TFullModel ReadZeroCopyModel(const void* binaryBuffer, size_t binaryBufferSize) {
    TFullModel model;
    model.InitNonOwning(binaryBuffer, binaryBufferSize);
    return model;
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

struct TSolidModelTree : IModelTreeData {
    TConstArrayRef<int> GetTreeSplits() const override;
    TConstArrayRef<int> GetTreeSizes() const override;
    TConstArrayRef<int> GetTreeStartOffsets() const override;
    TConstArrayRef<TNonSymmetricTreeStepNode> GetNonSymmetricStepNodes() const override;
    TConstArrayRef<ui32> GetNonSymmetricNodeIdToLeafId() const override;
    TConstArrayRef<double> GetLeafValues() const override;
    TConstArrayRef<double> GetLeafWeights() const override;
    THolder<IModelTreeData> Clone(ECloningPolicy policy) const override;

    void SetTreeSplits(const TVector<int>&) override;
    void SetTreeSizes(const TVector<int>&) override;
    void SetTreeStartOffsets(const TVector<int>&) override;
    void SetNonSymmetricStepNodes(const TVector<TNonSymmetricTreeStepNode>&) override;
    void SetNonSymmetricNodeIdToLeafId(const TVector<ui32>&) override;
    void SetLeafValues(const TVector<double>&) override;
    void SetLeafWeights(const TVector<double>&) override;

    TVector<int> TreeSplits;
    TVector<int> TreeSizes;
    TVector<int> TreeStartOffsets;
    TVector<TNonSymmetricTreeStepNode> NonSymmetricStepNodes;
    TVector<ui32> NonSymmetricNodeIdToLeafId;
    TVector<double> LeafValues;
    TVector<double> LeafWeights;
};

static TSolidModelTree* CastToSolidTree(const TModelTrees& trees) {
    auto ptr = dynamic_cast<TSolidModelTree*>(trees.GetModelTreeData().Get());
    CB_ENSURE(ptr, "Only solid models are modifiable");
    return ptr;
}

struct TOpaqueModelTree : IModelTreeData {
    TConstArrayRef<int> GetTreeSplits() const override;
    TConstArrayRef<int> GetTreeSizes() const override;
    TConstArrayRef<int> GetTreeStartOffsets() const override;
    TConstArrayRef<TNonSymmetricTreeStepNode> GetNonSymmetricStepNodes() const override;
    TConstArrayRef<ui32> GetNonSymmetricNodeIdToLeafId() const override;
    TConstArrayRef<double> GetLeafValues() const override;
    TConstArrayRef<double> GetLeafWeights() const override;
    THolder<IModelTreeData> Clone(ECloningPolicy policy) const override;

    void SetTreeSplits(const TVector<int>&) override;
    void SetTreeSizes(const TVector<int>&) override;
    void SetTreeStartOffsets(const TVector<int>&) override;
    void SetNonSymmetricStepNodes(const TVector<TNonSymmetricTreeStepNode>&) override;
    void SetNonSymmetricNodeIdToLeafId(const TVector<ui32>&) override;
    void SetLeafValues(const TVector<double>&) override;
    void SetLeafWeights(const TVector<double>&) override;

    TConstArrayRef<int> TreeSplits;
    TConstArrayRef<int> TreeSizes;
    TConstArrayRef<int> TreeStartOffsets;
    TConstArrayRef<TNonSymmetricTreeStepNode> NonSymmetricStepNodes;
    TConstArrayRef<ui32> NonSymmetricNodeIdToLeafId;
    TConstArrayRef<double> LeafValues;
    TConstArrayRef<double> LeafWeights;
};

static TOpaqueModelTree* CastToOpaqueTree(const TModelTrees& trees) {
    auto ptr = dynamic_cast<TOpaqueModelTree*>(trees.GetModelTreeData().Get());
    CB_ENSURE(ptr, "Not an opaque model");
    return ptr;
}

TModelTrees::TModelTrees() {
    ModelTreeData = MakeHolder<TSolidModelTree>();
    UpdateRuntimeData();
}

void TModelTrees::ProcessSplitsSet(
    const TSet<TModelSplit>& modelSplitSet,
    const TVector<size_t>& floatFeaturesInternalIndexesMap,
    const TVector<size_t>& catFeaturesInternalIndexesMap,
    const TVector<size_t>& textFeaturesInternalIndexesMap,
    const TVector<size_t>& embeddingFeaturesInternalIndexesMap
) {
    THashSet<int> usedCatFeatureIndexes;
    THashSet<int> usedTextFeatureIndexes;
    THashSet<int> usedEmbeddingFeatureIndexes;
    for (const auto& split : modelSplitSet) {
        if (split.Type == ESplitType::FloatFeature) {
            const size_t internalFloatIndex = floatFeaturesInternalIndexesMap.at((size_t)split.FloatFeature.FloatFeature);
            FloatFeatures.at(internalFloatIndex).Borders.push_back(split.FloatFeature.Split);
        } else if (split.Type == ESplitType::EstimatedFeature) {
            const TEstimatedFeatureSplit estimatedFeatureSplit = split.EstimatedFeature;
            if (estimatedFeatureSplit.ModelEstimatedFeature.SourceFeatureType == EEstimatedSourceFeatureType::Text) {
                usedTextFeatureIndexes.insert(estimatedFeatureSplit.ModelEstimatedFeature.SourceFeatureId);
            } else {
                usedEmbeddingFeatureIndexes.insert(estimatedFeatureSplit.ModelEstimatedFeature.SourceFeatureId);
            }
            if (EstimatedFeatures.empty() ||
                EstimatedFeatures.back().ModelEstimatedFeature != estimatedFeatureSplit.ModelEstimatedFeature
            ) {
                TEstimatedFeature estimatedFeature(estimatedFeatureSplit.ModelEstimatedFeature);
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
    for (const int usedEmbeddingFeatureIdx : usedEmbeddingFeatureIndexes) {
        EmbeddingFeatures[embeddingFeaturesInternalIndexesMap.at(usedEmbeddingFeatureIdx)].SetUsedInModel(true);
    }
}

void TModelTrees::AddBinTree(const TVector<int>& binSplits) {
    auto& data = *CastToSolidTree(*this);

    Y_ASSERT(data.TreeSizes.size() == data.TreeStartOffsets.size() && data.TreeSplits.empty() == data.TreeSizes.empty());
    data.TreeSplits.insert(data.TreeSplits.end(), binSplits.begin(), binSplits.end());
    if (data.TreeStartOffsets.empty()) {
        data.TreeStartOffsets.push_back(0);
    } else {
        data.TreeStartOffsets.push_back(data.TreeStartOffsets.back() + data.TreeSizes.back());
    }
    data.TreeSizes.push_back(binSplits.ysize());
}

void TModelTrees::ClearLeafWeights() {
    CastToSolidTree(*this)->LeafWeights.clear();
}

void TModelTrees::AddTreeSplit(int treeSplit) {
    CastToSolidTree(*this)->TreeSplits.push_back(treeSplit);
}
void TModelTrees::AddTreeSize(int treeSize) {
    auto& data = *CastToSolidTree(*this);
    if (data.TreeStartOffsets.empty()) {
        data.TreeStartOffsets.push_back(0);
    } else {
        data.TreeStartOffsets.push_back(data.TreeStartOffsets.back() + data.TreeSizes.back());
    }
    data.TreeSizes.push_back(treeSize);
}

void TModelTrees::AddLeafValue(double leafValue) {
    CastToSolidTree(*this)->LeafValues.push_back(leafValue);
}

void TModelTrees::AddLeafWeight(double leafWeight) {
    CastToSolidTree(*this)->LeafWeights.push_back(leafWeight);
}

bool TModelTrees::IsSolid() const {
    return dynamic_cast<TSolidModelTree*>(ModelTreeData.Get());
}

void TModelTrees::TruncateTrees(size_t begin, size_t end) {
    //TODO(eermishkina): support non symmetric trees
    CB_ENSURE(IsOblivious(), "Truncate support only symmetric trees");
    CB_ENSURE(begin <= end, "begin tree index should be not greater than end tree index.");
    CB_ENSURE(end <= GetModelTreeData()->GetTreeSplits().size(), "end tree index should be not greater than tree count.");
    auto savedScaleAndBias = GetScaleAndBias();
    TObliviousTreeBuilder builder(FloatFeatures,
                                  CatFeatures,
                                  TextFeatures,
                                  EmbeddingFeatures,
                                  ApproxDimension);
    auto applyData = GetApplyData();
    const auto& leafOffsets = applyData->TreeFirstLeafOffsets;

    const auto treeSizes = GetModelTreeData()->GetTreeSizes();
    const auto treeSplits = GetModelTreeData()->GetTreeSplits();
    const auto leafValues = GetModelTreeData()->GetLeafValues();
    const auto leafWeights = GetModelTreeData()->GetLeafWeights();
    const auto treeStartOffsets = GetModelTreeData()->GetTreeStartOffsets();
    for (size_t treeIdx = begin; treeIdx < end; ++treeIdx) {
        TVector<TModelSplit> modelSplits;
        for (int splitIdx = treeStartOffsets[treeIdx];
             splitIdx < treeStartOffsets[treeIdx] + treeSizes[treeIdx];
             ++splitIdx)
        {
            modelSplits.push_back(GetBinFeatures()[treeSplits[splitIdx]]);
        }
        TConstArrayRef<double> leafValuesRef(
            leafValues.begin() + leafOffsets[treeIdx],
            leafValues.begin() + leafOffsets[treeIdx] + ApproxDimension * (1u << treeSizes[treeIdx])
        );
        builder.AddTree(
            modelSplits,
            leafValuesRef,
            leafWeights.empty() ? TConstArrayRef<double>() : TConstArrayRef<double>(
                leafWeights.begin() + leafOffsets[treeIdx] / ApproxDimension,
                leafWeights.begin() + leafOffsets[treeIdx] / ApproxDimension + (1ull << treeSizes[treeIdx])
            )
        );
    }
    builder.Build(this);
    this->SetScaleAndBias(savedScaleAndBias);
}

flatbuffers::Offset<NCatBoostFbs::TModelTrees>
TModelTrees::FBSerialize(TModelPartsCachingSerializer& serializer) const {
    auto& builder = serializer.FlatbufBuilder;

    std::vector<flatbuffers::Offset<NCatBoostFbs::TCatFeature>> catFeaturesOffsets;
    for (const auto& catFeature : CatFeatures) {
        catFeaturesOffsets.push_back(catFeature.FBSerialize(builder));
    }
    auto fbsCatFeaturesOffsets = builder.CreateVector(catFeaturesOffsets);

    std::vector<flatbuffers::Offset<NCatBoostFbs::TFloatFeature>> floatFeaturesOffsets;
    for (const auto& floatFeature : FloatFeatures) {
        floatFeaturesOffsets.push_back(floatFeature.FBSerialize(builder));
    }
    auto fbsFloatFeaturesOffsets = builder.CreateVector(floatFeaturesOffsets);

    std::vector<flatbuffers::Offset<NCatBoostFbs::TTextFeature>> textFeaturesOffsets;
    for (const auto& textFeature : TextFeatures) {
        textFeaturesOffsets.push_back(textFeature.FBSerialize(builder));
    }
    auto fbsTextFeaturesOffsets = builder.CreateVector(textFeaturesOffsets);

    std::vector<flatbuffers::Offset<NCatBoostFbs::TEmbeddingFeature>> embeddingFeaturesOffsets;
    for (const auto& embeddingFeature : EmbeddingFeatures) {
        embeddingFeaturesOffsets.push_back(embeddingFeature.FBSerialize(builder));
    }
    auto fbsEmbeddingFeaturesOffsets = builder.CreateVector(embeddingFeaturesOffsets);

    std::vector<flatbuffers::Offset<NCatBoostFbs::TEstimatedFeature>> estimatedFeaturesOffsets;
    for (const auto& estimatedFeature : EstimatedFeatures) {
        estimatedFeaturesOffsets.push_back(estimatedFeature.FBSerialize(builder));
    }
    auto fbsEstimatedFeaturesOffsets = builder.CreateVector(estimatedFeaturesOffsets);

    std::vector<flatbuffers::Offset<NCatBoostFbs::TOneHotFeature>> oneHotFeaturesOffsets;
    for (const auto& oneHotFeature : OneHotFeatures) {
        oneHotFeaturesOffsets.push_back(oneHotFeature.FBSerialize(builder));
    }
    auto fbsOneHotFeaturesOffsets = builder.CreateVector(oneHotFeaturesOffsets);

    std::vector<flatbuffers::Offset<NCatBoostFbs::TCtrFeature>> ctrFeaturesOffsets;
    for (const auto& ctrFeature : CtrFeatures) {
        ctrFeaturesOffsets.push_back(ctrFeature.FBSerialize(serializer));
    }
    auto fbsCtrFeaturesOffsets = builder.CreateVector(ctrFeaturesOffsets);

    TVector<NCatBoostFbs::TNonSymmetricTreeStepNode> nonSymmetricTreeStepNode;
    nonSymmetricTreeStepNode.reserve(GetModelTreeData()->GetNonSymmetricStepNodes().size());
    for (const auto& nonSymmetricStep: GetModelTreeData()->GetNonSymmetricStepNodes()) {
        nonSymmetricTreeStepNode.emplace_back(NCatBoostFbs::TNonSymmetricTreeStepNode{
            nonSymmetricStep.LeftSubtreeDiff,
            nonSymmetricStep.RightSubtreeDiff
        });
    }
    auto fbsNonSymmetricTreeStepNode = builder.CreateVectorOfStructs(nonSymmetricTreeStepNode);

    TVector<NCatBoostFbs::TRepackedBin> repackedBins;
    repackedBins.reserve(GetRepackedBins().size());
    for (const auto& repackedBin: GetRepackedBins()) {
        repackedBins.emplace_back(NCatBoostFbs::TRepackedBin{
            repackedBin.FeatureIndex,
            repackedBin.XorMask,
            repackedBin.SplitIdx
        });
    }
    auto fbsRepackedBins = builder.CreateVectorOfStructs(repackedBins);

    auto& data = GetModelTreeData();
    auto fbsTreeSplits = builder.CreateVector(data->GetTreeSplits().data(), data->GetTreeSplits().size());
    auto fbsTreeSizes = builder.CreateVector(data->GetTreeSizes().data(), data->GetTreeSizes().size());
    auto fbsTreeStartOffsets = builder.CreateVector(data->GetTreeStartOffsets().data(), data->GetTreeStartOffsets().size());
    auto fbsLeafValues = builder.CreateVector(data->GetLeafValues().data(), data->GetLeafValues().size());
    auto fbsLeafWeights = builder.CreateVector(data->GetLeafWeights().data(), data->GetLeafWeights().size());
    auto fbsNonSymmetricNodeIdToLeafId = builder.CreateVector(data->GetNonSymmetricNodeIdToLeafId().data(), data->GetNonSymmetricNodeIdToLeafId().size());
    auto bias = GetScaleAndBias().GetBiasRef();
    auto fbsBias = builder.CreateVector(bias.data(), bias.size());
    return NCatBoostFbs::CreateTModelTrees(
        builder,
        ApproxDimension,
        fbsTreeSplits,
        fbsTreeSizes,
        fbsTreeStartOffsets,
        fbsCatFeaturesOffsets,
        fbsFloatFeaturesOffsets,
        fbsOneHotFeaturesOffsets,
        fbsCtrFeaturesOffsets,
        fbsLeafValues,
        fbsLeafWeights,
        fbsNonSymmetricTreeStepNode,
        fbsNonSymmetricNodeIdToLeafId,
        fbsTextFeaturesOffsets,
        fbsEstimatedFeaturesOffsets,
        GetScaleAndBias().Scale,
        0,
        fbsBias,
        fbsRepackedBins,
        fbsEmbeddingFeaturesOffsets
    );
}

static_assert(sizeof(TRepackedBin) == sizeof(NCatBoostFbs::TRepackedBin));

void TModelTrees::UpdateRuntimeData() {
    CalcForApplyData();
    CalcBinFeatures();
}

void TModelTrees::ProcessFloatFeatures() {
    for (const auto& feature : FloatFeatures) {
        if (feature.UsedInModel()) {
            ++ApplyData->UsedFloatFeaturesCount;
            ApplyData->MinimalSufficientFloatFeaturesVectorSize = static_cast<size_t>(feature.Position.Index) + 1;
        }
    }
}

void TModelTrees::ProcessCatFeatures() {
    for (const auto& feature : CatFeatures) {
        if (feature.UsedInModel()) {
            ++ApplyData->UsedCatFeaturesCount;
            ApplyData->MinimalSufficientCatFeaturesVectorSize = static_cast<size_t>(feature.Position.Index) + 1;
        }
    }
}

void TModelTrees::ProcessTextFeatures() {
    for (const auto& feature : TextFeatures) {
        if (feature.UsedInModel()) {
            ++ApplyData->UsedTextFeaturesCount;
            ApplyData->MinimalSufficientTextFeaturesVectorSize = static_cast<size_t>(feature.Position.Index) + 1;
        }
    }
}

void TModelTrees::ProcessEmbeddingFeatures() {
    for (const auto& feature : EmbeddingFeatures) {
        if (feature.UsedInModel()) {
            ++ApplyData->UsedEmbeddingFeaturesCount;
            ApplyData->MinimalSufficientEmbeddingFeaturesVectorSize = static_cast<size_t>(feature.Position.Index) + 1;
        }
    }
}

void TModelTrees::ProcessEstimatedFeatures() {
    ApplyData->UsedEstimatedFeaturesCount = EstimatedFeatures.size();
}

void TModelTrees::CalcBinFeatures() {

    auto runtimeData = MakeAtomicShared<TRuntimeData>();

    struct TFeatureSplitId {
        ui32 FeatureIdx = 0;
        ui32 SplitIdx = 0;
    };
    TVector<TFeatureSplitId> splitIds;

    auto& ref = *runtimeData;
    for (const auto& feature : FloatFeatures) {
        if (!feature.UsedInModel()) {
            continue;
        }
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

    for (const auto& feature : EstimatedFeatures) {
        for (int borderId = 0; borderId < feature.Borders.ysize(); ++borderId) {
            TEstimatedFeatureSplit split{
                feature.ModelEstimatedFeature,
                feature.Borders[borderId]
            };
            ref.BinFeatures.emplace_back(split);
            auto& bf = splitIds.emplace_back();
            bf.FeatureIdx = ref.EffectiveBinFeaturesBucketCount + borderId / MAX_VALUES_PER_BIN;
            bf.SplitIdx = (borderId % MAX_VALUES_PER_BIN) + 1;
        }
        ref.EffectiveBinFeaturesBucketCount
            += (feature.Borders.size() + MAX_VALUES_PER_BIN - 1) / MAX_VALUES_PER_BIN;
    }

    for (const auto& feature : OneHotFeatures) {
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
    RuntimeData = runtimeData;

    TVector<TRepackedBin> repackedBins;

    auto treeSplits = GetModelTreeData()->GetTreeSplits();
    // All the "trees" have no conditions, should treat them carefully
    // Case only valid for nonsymmetric trees
    if (splitIds.empty() && !treeSplits.empty()) {
        for (const auto& binSplit : treeSplits) {
            CB_ENSURE_INTERNAL(binSplit == 0, "expected 0 as empty nodes marker");
        }
        CB_ENSURE_INTERNAL(
            !IsOblivious(),
            "Expected asymmetric trees: Split ids present and are zeroes, but there is no actual splits in model"
        );
        repackedBins.resize(treeSplits.size(), TRepackedBin{});
    } else {
        for (const auto& binSplit : treeSplits) {
            const auto& feature = ref.BinFeatures[binSplit];
            const auto& featureIndex = splitIds[binSplit];
            CB_ENSURE(
                featureIndex.FeatureIdx <= 0xffff,
                "Too many features in model, ask catboost team for support"
            );
            TRepackedBin rb;
            rb.FeatureIndex = featureIndex.FeatureIdx;
            if (feature.Type != ESplitType::OneHotFeature) {
                rb.SplitIdx = featureIndex.SplitIdx;
            } else {
                rb.XorMask = ((~featureIndex.SplitIdx) & 0xff);
                rb.SplitIdx = 0xff;
            }
            repackedBins.push_back(rb);
        }
    }
    RepackedBins = NCB::TMaybeOwningConstArrayHolder<TRepackedBin>::CreateOwning(std::move(repackedBins));
}

void TModelTrees::CalcUsedModelCtrs() {
    auto& ref = ApplyData->UsedModelCtrs;
    for (const auto& ctrFeature : CtrFeatures) {
        ref.push_back(ctrFeature.Ctr);
    }
}

void TModelTrees::CalcFirstLeafOffsets() {
    auto treeSizes = GetModelTreeData()->GetTreeSizes();
    auto treeStartOffsets = GetModelTreeData()->GetTreeStartOffsets();

    auto& ref = ApplyData->TreeFirstLeafOffsets;
    ref.resize(treeSizes.size());
    if (IsOblivious()) {
        size_t currentOffset = 0;
        for (size_t i = 0; i < treeSizes.size(); ++i) {
            ref[i] = currentOffset;
            currentOffset += (1 << treeSizes[i]) * ApproxDimension;
        }
    } else {
        for (size_t treeId = 0; treeId < treeSizes.size(); ++treeId) {
            const int treeNodesStart = treeStartOffsets[treeId];
            const int treeNodesEnd = treeNodesStart + treeSizes[treeId];
            ui32 minLeafValueIndex = Max();
            ui32 maxLeafValueIndex = 0;
            ui32 valueNodeCount = 0; // count of nodes with values
            for (auto nodeIndex = treeNodesStart; nodeIndex < treeNodesEnd; ++nodeIndex) {
                const auto &node = GetModelTreeData()->GetNonSymmetricStepNodes()[nodeIndex];
                if (node.LeftSubtreeDiff == 0 || node.RightSubtreeDiff == 0) {
                    const ui32 leafValueIndex = GetModelTreeData()->GetNonSymmetricNodeIdToLeafId()[nodeIndex];
                    Y_ASSERT(leafValueIndex != Max<ui32>());
                    CB_ENSURE(
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
            ref[treeId] = minLeafValueIndex;
        }
    }
}

void TModelTrees::DropUnusedFeatures() {
    EraseIf(FloatFeatures, [](const TFloatFeature& feature) { return !feature.UsedInModel();});
    EraseIf(CatFeatures, [](const TCatFeature& feature) { return !feature.UsedInModel(); });
    EraseIf(TextFeatures, [](const TTextFeature& feature) { return !feature.UsedInModel(); });
    EraseIf(EmbeddingFeatures, [](const TEmbeddingFeature& feature) { return !feature.UsedInModel(); });
    UpdateRuntimeData();
}

void TModelTrees::ConvertObliviousToAsymmetric() {
    if (!IsOblivious() || !IsSolid()) {
        return;
    }
    TVector<int> treeSplits;
    TVector<int> treeSizes;
    TVector<int> treeStartOffsets;
    TVector<TNonSymmetricTreeStepNode> nonSymmetricStepNodes;
    TVector<ui32> nonSymmetricNodeIdToLeafId;

    size_t leafStartOffset = 0;
    auto& data = *CastToSolidTree(*this);
    for (size_t treeId = 0; treeId < data.TreeSizes.size(); ++treeId) {
        size_t treeSize = 0;
        treeStartOffsets.push_back(treeSplits.size());
        for (int depth = 0; depth < data.TreeSizes[treeId]; ++depth) {
            const auto split = data.TreeSplits[data.TreeStartOffsets[treeId] + data.TreeSizes[treeId] - 1 - depth];
            for (size_t cloneId = 0; cloneId < (1ull << depth); ++cloneId) {
                treeSplits.push_back(split);
                nonSymmetricNodeIdToLeafId.push_back(Max<ui32>());
                nonSymmetricStepNodes.emplace_back(TNonSymmetricTreeStepNode{static_cast<ui16>(treeSize + 1), static_cast<ui16>(treeSize + 2)});
                ++treeSize;
            }
        }
        for (size_t cloneId = 0; cloneId < (1ull << data.TreeSizes[treeId]); ++cloneId) {
            treeSplits.push_back(0);
            nonSymmetricNodeIdToLeafId.push_back((leafStartOffset + cloneId) * ApproxDimension);
            nonSymmetricStepNodes.emplace_back(TNonSymmetricTreeStepNode{0, 0});
            ++treeSize;
        }
        leafStartOffset += (1ull << data.TreeSizes[treeId]);
        treeSizes.push_back(treeSize);
    }

    data.TreeSplits = std::move(treeSplits);
    data.TreeSizes = std::move(treeSizes);
    data.TreeStartOffsets = std::move(treeStartOffsets);
    data.NonSymmetricStepNodes = std::move(nonSymmetricStepNodes);
    data.NonSymmetricNodeIdToLeafId = std::move(nonSymmetricNodeIdToLeafId);
    UpdateRuntimeData();
}

TVector<ui32> TModelTrees::GetTreeLeafCounts() const {
    auto applyData = GetApplyData();
    const auto& firstLeafOfsets = applyData->TreeFirstLeafOffsets;
    Y_ASSERT(IsSorted(firstLeafOfsets.begin(), firstLeafOfsets.end()));
    TVector<ui32> treeLeafCounts;
    treeLeafCounts.reserve(GetTreeCount());
    for (size_t treeNum = 0; treeNum < GetTreeCount(); ++treeNum) {
        const size_t currTreeLeafValuesEnd = (
            treeNum + 1 < GetTreeCount()
            ? firstLeafOfsets[treeNum + 1]
            : GetModelTreeData()->GetLeafValues().size()
        );
        const size_t currTreeLeafValuesCount = currTreeLeafValuesEnd - firstLeafOfsets[treeNum];
        Y_ASSERT(currTreeLeafValuesCount % ApproxDimension == 0);
        treeLeafCounts.push_back(currTreeLeafValuesCount / ApproxDimension);
    }
    return treeLeafCounts;
}

void TModelTrees::SetScaleAndBias(const TScaleAndBias& scaleAndBias) {
    CB_ENSURE(IsValidFloat(scaleAndBias.Scale), "Invalid scale " << scaleAndBias.Scale);
    TVector<double> bias = scaleAndBias.GetBiasRef();
    for (auto b: bias) {
        CB_ENSURE(IsValidFloat(b), "Invalid bias " << b);
    }
    if (bias.empty()) {
        bias.resize(GetDimensionsCount(), 0);
    }
    CB_ENSURE(
        GetDimensionsCount() == bias.size(),
        "Inappropraite dimension of bias, should be " << GetDimensionsCount() << " found " << bias.size());

    ScaleAndBias = TScaleAndBias(scaleAndBias.Scale, bias);
}

void TModelTrees::SetScaleAndBias(const NCatBoostFbs::TModelTrees* fbObj) {
    ApproxDimension = fbObj->ApproxDimension();
    TVector<double> bias;
    if (fbObj->MultiBias() && fbObj->MultiBias()->size()) {
        bias.assign(fbObj->MultiBias()->data(), fbObj->MultiBias()->data() + fbObj->MultiBias()->size());
    } else {
        CB_ENSURE(ApproxDimension == 1 || fbObj->Bias() == 0,
                  "Inappropraite dimension of bias, should be " << GetDimensionsCount() << " found 1");
        bias.resize(ApproxDimension, fbObj->Bias());
    }
    SetScaleAndBias({fbObj->Scale(), bias});
}

void TModelTrees::DeserializeFeatures(const NCatBoostFbs::TModelTrees* fbObj) {
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
    FBS_ARRAY_DESERIALIZER(EmbeddingFeatures)
    FBS_ARRAY_DESERIALIZER(EstimatedFeatures)
    FBS_ARRAY_DESERIALIZER(OneHotFeatures)
    FBS_ARRAY_DESERIALIZER(CtrFeatures)
#undef FBS_ARRAY_DESERIALIZER
}

void TModelTrees::FBDeserializeOwning(const NCatBoostFbs::TModelTrees* fbObj) {
    ApproxDimension = fbObj->ApproxDimension();
    SetScaleAndBias(fbObj);

    auto& data = *CastToSolidTree(*this);

    if (fbObj->TreeSplits()) {
        data.TreeSplits.assign(fbObj->TreeSplits()->begin(), fbObj->TreeSplits()->end());
    }
    if (fbObj->TreeSizes()) {
        data.TreeSizes.assign(fbObj->TreeSizes()->begin(), fbObj->TreeSizes()->end());
    }
    if (fbObj->TreeStartOffsets()) {
        data.TreeStartOffsets.assign(fbObj->TreeStartOffsets()->begin(), fbObj->TreeStartOffsets()->end());
    }

    if (fbObj->LeafValues()) {
        data.LeafValues.assign(
            fbObj->LeafValues()->data(),
            fbObj->LeafValues()->data() + fbObj->LeafValues()->size()
        );
    }
    if (fbObj->NonSymmetricStepNodes()) {
        data.NonSymmetricStepNodes.resize(fbObj->NonSymmetricStepNodes()->size());
        std::copy(
            fbObj->NonSymmetricStepNodes()->begin(),
            fbObj->NonSymmetricStepNodes()->end(),
            data.NonSymmetricStepNodes.begin()
        );
    }
    if (fbObj->NonSymmetricNodeIdToLeafId()) {
        data.NonSymmetricNodeIdToLeafId.assign(
            fbObj->NonSymmetricNodeIdToLeafId()->begin(), fbObj->NonSymmetricNodeIdToLeafId()->end()
        );
    }
    if (fbObj->LeafWeights() && fbObj->LeafWeights()->size() > 0) {
        data.LeafWeights.assign(
            fbObj->LeafWeights()->data(),
            fbObj->LeafWeights()->data() + fbObj->LeafWeights()->size()
        );
    }

    if (fbObj->RepackedBins()) {
        TVector<TRepackedBin> repackedBins(fbObj->RepackedBins()->size());
        std::copy(
            fbObj->RepackedBins()->begin(),
            fbObj->RepackedBins()->end(),
            repackedBins.begin()
        );
        RepackedBins = NCB::TMaybeOwningConstArrayHolder<TRepackedBin>::CreateOwning(std::move(repackedBins));
    }

    DeserializeFeatures(fbObj);
}

void TModelTrees::FBDeserializeNonOwning(const NCatBoostFbs::TModelTrees* fbObj) {
    ModelTreeData = MakeHolder<TOpaqueModelTree>();

    ApproxDimension = fbObj->ApproxDimension();
    SetScaleAndBias(fbObj);
    DeserializeFeatures(fbObj);

    auto& data = *CastToOpaqueTree(*this);

    if (fbObj->TreeSplits()) {
        data.TreeSplits = TConstArrayRef<int>(fbObj->TreeSplits()->data(), fbObj->TreeSplits()->size());
    }
    if (fbObj->TreeSizes()) {
        data.TreeSizes = TConstArrayRef<int>(fbObj->TreeSizes()->data(), fbObj->TreeSizes()->size());
    }
    if (fbObj->TreeStartOffsets()) {
        data.TreeStartOffsets = TConstArrayRef<int>(fbObj->TreeStartOffsets()->data(), fbObj->TreeStartOffsets()->size());
    }

    if (fbObj->LeafValues()) {
        data.LeafValues = TConstArrayRef<double>(fbObj->LeafValues()->data(), fbObj->LeafValues()->size());
    }
    if (fbObj->NonSymmetricStepNodes()) {
        static_assert(sizeof(TNonSymmetricTreeStepNode) == sizeof(NCatBoostFbs::TNonSymmetricTreeStepNode));
        auto ptr = reinterpret_cast<const TNonSymmetricTreeStepNode*>(fbObj->NonSymmetricStepNodes()->data());
        data.NonSymmetricStepNodes = TConstArrayRef<TNonSymmetricTreeStepNode>(ptr, fbObj->NonSymmetricStepNodes()->size());
    }
    if (fbObj->NonSymmetricNodeIdToLeafId()) {
        data.NonSymmetricNodeIdToLeafId = TConstArrayRef<ui32>(fbObj->NonSymmetricNodeIdToLeafId()->data(), fbObj->NonSymmetricNodeIdToLeafId()->size());
    }
    if (fbObj->LeafWeights() && fbObj->LeafWeights()->size() > 0) {
        data.LeafWeights = TConstArrayRef<double>(fbObj->LeafWeights()->data(), fbObj->LeafWeights()->size());
    }

    if (fbObj->RepackedBins()) {
        auto ptr = reinterpret_cast<const TRepackedBin*>(fbObj->RepackedBins()->data());
        RepackedBins = NCB::TMaybeOwningConstArrayHolder<TRepackedBin>::CreateNonOwning(TArrayRef(ptr, fbObj->RepackedBins()->size()));
    }
}

TConstArrayRef<int> TSolidModelTree::GetTreeSplits() const {
    return TreeSplits;
}

TConstArrayRef<int> TSolidModelTree::GetTreeSizes() const {
    return TreeSizes;
}

TConstArrayRef<int> TSolidModelTree::GetTreeStartOffsets() const {
    return TreeStartOffsets;
}

TConstArrayRef<TNonSymmetricTreeStepNode> TSolidModelTree::GetNonSymmetricStepNodes() const {
    return NonSymmetricStepNodes;
}

TConstArrayRef<ui32> TSolidModelTree::GetNonSymmetricNodeIdToLeafId() const {
    return NonSymmetricNodeIdToLeafId;
}

TConstArrayRef<double> TSolidModelTree::GetLeafValues() const {
    return LeafValues;
}

TConstArrayRef<double> TSolidModelTree::GetLeafWeights() const {
    return LeafWeights;
}

THolder<IModelTreeData> TSolidModelTree::Clone(ECloningPolicy policy) const {
    switch (policy) {
        case ECloningPolicy::CloneAsOpaque: {
            auto holder = MakeHolder<TOpaqueModelTree>();
            holder->LeafValues = TConstArrayRef<double>(LeafValues.data(), LeafValues.size());
            holder->LeafWeights = TConstArrayRef<double>(LeafWeights.data(), LeafWeights.size());
            holder->NonSymmetricNodeIdToLeafId = TConstArrayRef<ui32>(NonSymmetricNodeIdToLeafId.data(), NonSymmetricNodeIdToLeafId.size());
            holder->NonSymmetricStepNodes = TConstArrayRef<TNonSymmetricTreeStepNode>(NonSymmetricStepNodes.data(), NonSymmetricStepNodes.size());
            holder->TreeSizes = TConstArrayRef<int>(TreeSizes.data(), TreeSizes.size());
            holder->TreeSplits = TConstArrayRef<int>(TreeSplits.data(), TreeSplits.size());
            holder->TreeStartOffsets = TConstArrayRef<int>(TreeStartOffsets.data(), TreeStartOffsets.size());
            return holder;
        }
        default:
            return MakeHolder<TSolidModelTree>(*this);
    }
}

void TSolidModelTree::SetTreeSplits(const TVector<int> &v) {
    TreeSplits = v;
}

void TSolidModelTree::SetTreeSizes(const TVector<int> &v) {
    TreeSizes = v;
}

void TSolidModelTree::SetTreeStartOffsets(const TVector<int> &v) {
    TreeStartOffsets = v;
}

void TSolidModelTree::SetNonSymmetricStepNodes(const TVector<TNonSymmetricTreeStepNode> &v) {
    NonSymmetricStepNodes = v;
}

void TSolidModelTree::SetNonSymmetricNodeIdToLeafId(const TVector<ui32> &v) {
    NonSymmetricNodeIdToLeafId = v;
}

void TSolidModelTree::SetLeafValues(const TVector<double> &v) {
    LeafValues = v;
}

void TSolidModelTree::SetLeafWeights(const TVector<double> &v) {
    LeafWeights = v;
}


TConstArrayRef<int> TOpaqueModelTree::GetTreeSplits() const {
    return TreeSplits;
}

TConstArrayRef<int> TOpaqueModelTree::GetTreeSizes() const {
    return TreeSizes;
}

TConstArrayRef<int> TOpaqueModelTree::GetTreeStartOffsets() const {
    return TreeStartOffsets;
}

TConstArrayRef<TNonSymmetricTreeStepNode> TOpaqueModelTree::GetNonSymmetricStepNodes() const {
    return NonSymmetricStepNodes;
}

TConstArrayRef<ui32> TOpaqueModelTree::GetNonSymmetricNodeIdToLeafId() const {
    return NonSymmetricNodeIdToLeafId;
}

TConstArrayRef<double> TOpaqueModelTree::GetLeafValues() const {
    return LeafValues;
}

TConstArrayRef<double> TOpaqueModelTree::GetLeafWeights() const {
    return LeafWeights;
}

THolder<IModelTreeData> TOpaqueModelTree::Clone(ECloningPolicy policy) const {
    switch (policy) {
        case ECloningPolicy::CloneAsSolid: {
            auto holder = MakeHolder<TSolidModelTree>();
            holder->TreeSplits = TVector<int>(TreeSplits.begin(), TreeSplits.end());
            holder->TreeSizes = TVector<int>(TreeSizes.begin(), TreeSizes.end());
            holder->TreeStartOffsets = TVector<int>(TreeStartOffsets.begin(), TreeStartOffsets.end());
            holder->NonSymmetricStepNodes = TVector<TNonSymmetricTreeStepNode>(NonSymmetricStepNodes.begin(), NonSymmetricStepNodes.end());
            holder->NonSymmetricNodeIdToLeafId = TVector<ui32>(NonSymmetricNodeIdToLeafId.begin(), NonSymmetricNodeIdToLeafId.end());
            holder->LeafValues = TVector<double>(LeafValues.begin(), LeafValues.end());
            holder->LeafWeights = TVector<double>(LeafWeights.begin(), LeafWeights.end());
            return holder;
        }
        default:
            return MakeHolder<TOpaqueModelTree>(*this);
    }
}

void TOpaqueModelTree::SetTreeSplits(const TVector<int>&) {
    CB_ENSURE(false, "Only solid models are modifiable");
}

void TOpaqueModelTree::SetTreeSizes(const TVector<int>&) {
    CB_ENSURE(false, "Only solid models are modifiable");
}

void TOpaqueModelTree::SetTreeStartOffsets(const TVector<int>&) {
    CB_ENSURE(false, "Only solid models are modifiable");
}

void TOpaqueModelTree::SetNonSymmetricStepNodes(const TVector<TNonSymmetricTreeStepNode>&) {
    CB_ENSURE(false, "Only solid models are modifiable");
}

void TOpaqueModelTree::SetNonSymmetricNodeIdToLeafId(const TVector<ui32>&) {
    CB_ENSURE(false, "Only solid models are modifiable");
}

void TOpaqueModelTree::SetLeafValues(const TVector<double>&) {
    CB_ENSURE(false, "Only solid models are modifiable");
}

void TOpaqueModelTree::SetLeafWeights(const TVector<double>&) {
    CB_ENSURE(false, "Only solid models are modifiable");
}

TVector<EFormulaEvaluatorType> TFullModel::GetSupportedEvaluatorTypes() {
    TVector<EFormulaEvaluatorType> result;
    for (auto formulaEvaluatorType : GetEnumAllValues<EFormulaEvaluatorType>()) {
        if (NCB::NModelEvaluation::TEvaluationBackendFactory::Has(formulaEvaluatorType)) {
            result.push_back(formulaEvaluatorType);
        }
    }
    return result;
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

void TFullModel::CalcWithHashedCatAndTextAndEmbeddings(
    TConstArrayRef<TConstArrayRef<float>> floatFeatures,
    TConstArrayRef<TConstArrayRef<int>> catFeatures,
    TConstArrayRef<TVector<TStringBuf>> textFeatures,
    TConstArrayRef<TConstArrayRef<TConstArrayRef<float>>> embeddingFeatures,
    size_t treeStart,
    size_t treeEnd,
    TArrayRef<double> results,
    const TFeatureLayout* featureInfo
) const {
    TVector<TConstArrayRef<TStringBuf>> stringbufTextVecRefs{textFeatures.begin(), textFeatures.end()};
    GetCurrentEvaluator()->CalcWithHashedCatAndTextAndEmbeddings(floatFeatures, catFeatures, stringbufTextVecRefs, embeddingFeatures, treeStart, treeEnd, results, featureInfo);
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
    TConstArrayRef<TConstArrayRef<TConstArrayRef<float>>> embeddingFeatures,
    size_t treeStart,
    size_t treeEnd,
    TArrayRef<double> results,
    const TFeatureLayout* featureInfo
) const {
    TVector<TConstArrayRef<TStringBuf>> stringbufCatVecRefs{catFeatures.begin(), catFeatures.end()};
    TVector<TConstArrayRef<TStringBuf>> stringbufTextVecRefs{textFeatures.begin(), textFeatures.end()};
    GetCurrentEvaluator()->Calc(floatFeatures, stringbufCatVecRefs, stringbufTextVecRefs, embeddingFeatures, treeStart, treeEnd, results, featureInfo);
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
    auto modelTreesOffset = ModelTrees->FBSerialize(serializer);
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
    if (!!EmbeddingProcessingCollection) {
        modelPartIds.push_back(serializer.FlatbufBuilder.CreateString(EmbeddingProcessingCollection->GetStringIdentifier()));
    }
    auto coreOffset = CreateTModelCoreDirect(
        serializer.FlatbufBuilder,
        CURRENT_CORE_FORMAT_STRING,
        modelTreesOffset,
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
    if (!!EmbeddingProcessingCollection) {
        EmbeddingProcessingCollection->Save(s);
    }
}

void TFullModel::DefaultFullModelInit(const NCatBoostFbs::TModelCore* fbModelCore) {
    CB_ENSURE(
        fbModelCore->FormatVersion() && fbModelCore->FormatVersion()->str() == CURRENT_CORE_FORMAT_STRING,
        "Unsupported model format: " << fbModelCore->FormatVersion()->str()
    );

    ModelInfo.clear();
    if (fbModelCore->InfoMap()) {
        for (auto keyVal : *fbModelCore->InfoMap()) {
            ModelInfo[keyVal->Key()->str()] = keyVal->Value()->str();
        }
    }
}

void TFullModel::Load(IInputStream* s) {
    ReferenceMainFactoryRegistrators();
    using namespace flatbuffers;
    using namespace NCatBoostFbs;
    ui32 fileDescriptor;
    ::Load(s, fileDescriptor);
    CB_ENSURE(fileDescriptor == GetModelFormatDescriptor(), "Incorrect model file descriptor");
    auto coreSize = ::LoadSize(s);
    TArrayHolder<ui8> arrayHolder(new ui8[coreSize]);
    s->LoadOrFail(arrayHolder.Get(), coreSize);

    {
        flatbuffers::Verifier verifier(arrayHolder.Get(), coreSize, 64 /* max depth */, 256000000 /* max tables */);
        CB_ENSURE(VerifyTModelCoreBuffer(verifier), "Flatbuffers model verification failed");
    }
    auto fbModelCore = GetTModelCore(arrayHolder.Get());
    DefaultFullModelInit(fbModelCore);

    if (fbModelCore->ModelTrees()) {
        ModelTrees.GetMutable()->FBDeserializeOwning(fbModelCore->ModelTrees());
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
            } else if (modelPartId == NCB::TEmbeddingProcessingCollection::GetStringIdentifier()) {
                EmbeddingProcessingCollection = new NCB::TEmbeddingProcessingCollection();
                EmbeddingProcessingCollection->Load(s);
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

void TFullModel::InitNonOwning(const void* binaryBuffer, size_t binarySize) {
    using namespace flatbuffers;
    using namespace NCatBoostFbs;

    TMemoryInput in(binaryBuffer, binarySize);
    ui32 fileDescriptor;
    ::Load(&in, fileDescriptor);
    CB_ENSURE(fileDescriptor == GetModelFormatDescriptor(), "Incorrect model file descriptor");

    size_t coreSize = ::LoadSize(&in);
    const ui8* fbPtr = reinterpret_cast<const ui8*>(in.Buf());
    in.Skip(coreSize);

    {
        flatbuffers::Verifier verifier(fbPtr, coreSize, 64 /* max depth */, 256000000 /* max tables */);
        CB_ENSURE(VerifyTModelCoreBuffer(verifier), "Flatbuffers model verification failed");
    }

    auto fbModelCore = GetTModelCore(fbPtr);
    DefaultFullModelInit(fbModelCore);

    if (fbModelCore->ModelTrees()) {
        ModelTrees.GetMutable()->FBDeserializeNonOwning(fbModelCore->ModelTrees());
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
                auto ptr = new TStaticCtrProvider;
                CtrProvider = ptr;
                ptr->LoadNonOwning(&in);
            } else if (modelPartId == NCB::TTextProcessingCollection::GetStringIdentifier()) {
                TextProcessingCollection = new NCB::TTextProcessingCollection();
                TextProcessingCollection->LoadNonOwning(&in);
            } else if (modelPartId == NCB::TEmbeddingProcessingCollection::GetStringIdentifier()) {
                EmbeddingProcessingCollection = new NCB::TEmbeddingProcessingCollection();
                EmbeddingProcessingCollection->LoadNonOwning(&in);
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
    ModelTrees.GetMutable()->UpdateRuntimeData();
    if (CtrProvider) {
        CtrProvider->SetupBinFeatureIndexes(
            ModelTrees->GetFloatFeatures(),
            ModelTrees->GetOneHotFeatures(),
            ModelTrees->GetCatFeatures());
    }
    with_lock(CurrentEvaluatorLock) {
        Evaluator.Reset();
    }
}

TVector<TString> GetModelUsedFeaturesNames(const TFullModel& model) {
    TVector<int> featuresIdxs;
    TVector<TString> featuresNames;
    const TModelTrees& forest = *model.ModelTrees;

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
    for (const TTextFeature& feature : forest.GetTextFeatures()) {
        featuresIdxs.push_back(feature.Position.FlatIndex);
        featuresNames.push_back(
            feature.FeatureId == "" ? ToString(feature.Position.FlatIndex) : feature.FeatureId
        );
    }
    for (const TEmbeddingFeature& feature : forest.GetEmbeddingFeatures()) {
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

template <typename T>
TVector<size_t> MakeIndicesVector(const TConstArrayRef<T>& features) {
    TVector<size_t> indices;
    indices.reserve(features.size());
    for (const auto& feature : features) {
        indices.push_back(SafeIntegerCast<size_t>(feature.Position.FlatIndex));
    }
    return indices;
}

TVector<size_t> GetModelCatFeaturesIndices(const TFullModel& model) {
    return MakeIndicesVector(model.ModelTrees->GetCatFeatures());
}

TVector<size_t> GetModelFloatFeaturesIndices(const TFullModel& model) {
    return MakeIndicesVector(model.ModelTrees->GetFloatFeatures());
}

TVector<size_t> GetModelTextFeaturesIndices(const TFullModel& model) {
    return MakeIndicesVector(model.ModelTrees->GetTextFeatures());
}

TVector<size_t> GetModelEmbeddingFeaturesIndices(const TFullModel& model) {
    return MakeIndicesVector(model.ModelTrees->GetEmbeddingFeatures());
}

void SetModelExternalFeatureNames(const TVector<TString>& featureNames, TFullModel* model) {
    model->ModelTrees.GetMutable()->ApplyFeatureNames(featureNames);
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


double TFullModel::GetBinClassProbabilityThreshold() const {
    double threshold = DEFAULT_BINCLASS_PROBABILITY_THRESHOLD;
    if (ModelInfo.contains("binclass_probability_threshold")) {
        if (!TryFromString<double>(ModelInfo.at("binclass_probability_threshold"), threshold)) {
            CATBOOST_WARNING_LOG << "Float number at metadata key binclass_probability_threshold cannot be parsed" << Endl;
        }
    }
    return threshold;
}


double TFullModel::GetBinClassLogitThreshold() const {
    return NCB::Logit(GetBinClassProbabilityThreshold());
}


static TVector<NJson::TJsonValue> GetSequentialIntegerClassLabels(size_t classCount) {
    TVector<NJson::TJsonValue> classLabels;
    classLabels.reserve(classCount);
    for (int classIdx : xrange(SafeIntegerCast<int>(classCount))) {
        classLabels.emplace_back(classIdx);
    }
    return classLabels;
}


TVector<NJson::TJsonValue> TFullModel::GetModelClassLabels() const {
    TVector<NJson::TJsonValue> classLabels;

    TMaybe<TClassLabelOptions> classOptions;

    // "class_params" is new, more generic option, used for binclass as well
    for (const auto& paramName : {"class_params", "multiclass_params"}) {
        if (ModelInfo.contains(paramName)) {
            classOptions.ConstructInPlace();
            classOptions->Load(ReadTJsonValue(ModelInfo.at(paramName)));
            break;
        }
    }
    if (classOptions.Defined()) {
        if (classOptions->ClassLabels.IsSet()) {
            classLabels = classOptions->ClassLabels.Get();
            if (!classLabels.empty()) {
                return classLabels;
            }
        }
        if (classOptions->ClassesCount.IsSet()) {
            const size_t classesCount = SafeIntegerCast<size_t>(classOptions->ClassesCount.Get());
            if (classesCount) {
                return GetSequentialIntegerClassLabels(classesCount);
            }
        }
        if (classOptions->ClassToLabel.IsSet()) {
            classLabels.reserve(classOptions->ClassToLabel->size());
            for (float label : classOptions->ClassToLabel.Get()) {
                classLabels.emplace_back(int(label));
            }
            return classLabels;
        }
    }
    if (ModelInfo.contains("params")) {
        const TString& modelInfoParams = ModelInfo.at("params");
        NJson::TJsonValue paramsJson = ReadTJsonValue(modelInfoParams);
        if (paramsJson.Has("data_processing_options")
            && paramsJson["data_processing_options"].Has("class_names")) {

            const NJson::TJsonValue::TArray& classLabelsJsonArray
                = paramsJson["data_processing_options"]["class_names"].GetArraySafe();

            if (!classLabelsJsonArray.empty()) {
                classLabels.assign(classLabelsJsonArray.begin(), classLabelsJsonArray.end());
                return classLabels;
            }
        }
    }

    const TMaybe<NCatboostOptions::TLossDescription> lossDescription = GetLossDescription(*this);
    if (lossDescription.Defined() && IsClassificationObjective(lossDescription->GetLossFunction())) {
        const size_t dimensionsCount = GetDimensionsCount();
        return GetSequentialIntegerClassLabels((dimensionsCount == 1) ? 2 : dimensionsCount);
    }

    return classLabels;
}

void TFullModel::UpdateEstimatedFeaturesIndices(TVector<TEstimatedFeature>&& newEstimatedFeatures) {
    CB_ENSURE(
        TextProcessingCollection || EmbeddingProcessingCollection,
        "UpdateEstimatedFeatureIndices called when ProcessingCollections aren't defined"
    );

    ModelTrees.GetMutable()->SetEstimatedFeatures(std::move(newEstimatedFeatures));
    ModelTrees.GetMutable()->UpdateRuntimeData();
}

bool TFullModel::IsPosteriorSamplingModel() const {
    if (ModelInfo.contains("params")) {
        const TString& modelInfoParams = ModelInfo.at("params");
        NJson::TJsonValue paramsJson = ReadTJsonValue(modelInfoParams);
        if (paramsJson.Has("boosting_options") && paramsJson["boosting_options"].Has("posterior_sampling")) {
            return paramsJson["boosting_options"]["posterior_sampling"].GetBoolean();
        }
    }
    return false;
}

float TFullModel::GetActualShrinkCoef() const {
    CB_ENSURE(ModelInfo.contains("params"), "No params in model");
    const TString& modelInfoParams = ModelInfo.at("params");
    NJson::TJsonValue paramsJson = ReadTJsonValue(modelInfoParams);
    CB_ENSURE(paramsJson.Has("boosting_options"), "No boosting_options parameters in model");
    CB_ENSURE(paramsJson["boosting_options"].Has("learning_rate"),
        "No parameter learning_rate in model boosting_options");
    CB_ENSURE(paramsJson["boosting_options"].Has("model_shrink_rate"),
              "No parameter model_shrink_rate in model boosting_options");
    return paramsJson["boosting_options"]["learning_rate"].GetDouble() * paramsJson["boosting_options"]["model_shrink_rate"].GetDouble();
}

namespace {
    struct TUnknownFeature {};

    struct TFlatFeature {

        std::variant<TUnknownFeature, TFloatFeature, TCatFeature> FeatureVariant;

    public:
        TFlatFeature() = default;

        template <class TFeatureType>
        void SetOrCheck(const TFeatureType& other) {
            if (std::holds_alternative<TUnknownFeature>(FeatureVariant)) {
                FeatureVariant = other;
            }
            CB_ENSURE(std::holds_alternative<TFeatureType>(FeatureVariant),
                "Feature type mismatch: Categorical != Float for flat feature index: " <<
                other.Position.FlatIndex
            );
            TFeatureType& feature = std::get<TFeatureType>(FeatureVariant);
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
                constexpr auto asFalse = TFloatFeature::ENanValueTreatment::AsFalse;
                constexpr auto asIs = TFloatFeature::ENanValueTreatment::AsIs;
                if (
                    (feature.NanValueTreatment == asIs && other.NanValueTreatment == asFalse) ||
                    (feature.NanValueTreatment == asFalse && other.NanValueTreatment == asIs)
                    ) {
                    // We can relax Nan treatmen comparison as nans within AsIs strategy are always treated like AsFalse
                    // TODO(kirillovs): later implement splitted storage for float feautres with different Nan treatment
                    feature.NanValueTreatment = asFalse;
                } else {
                    CB_ENSURE(
                            feature.NanValueTreatment == other.NanValueTreatment,
                            "Nan value treatment differs: " << (int) feature.NanValueTreatment << " != " <<
                                                            (int) other.NanValueTreatment
                    );
                }
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

static void StreamModelTreesWithoutScaleAndBiasToBuilder(
    const TModelTrees& trees,
    double leafMultiplier,
    TObliviousTreeBuilder* builder,
    bool streamLeafWeights,
    ECtrTableMergePolicy ctrTableMergePolicy,
    THashMap<TModelCtrBaseMergeKey, TCtrTablesMergeStatus>* ctrTablesIndices
) {
    auto& data = trees.GetModelTreeData();
    const auto& binFeatures = trees.GetBinFeatures();
    auto applyData = trees.GetApplyData();
    const auto& leafOffsets = applyData->TreeFirstLeafOffsets;
    for (size_t treeIdx = 0; treeIdx < data->GetTreeSizes().size(); ++treeIdx) {
        TVector<TModelSplit> modelSplits;
        for (int splitIdx = data->GetTreeStartOffsets()[treeIdx];
             splitIdx < data->GetTreeStartOffsets()[treeIdx] + data->GetTreeSizes()[treeIdx];
             ++splitIdx)
        {
            modelSplits.push_back(binFeatures[data->GetTreeSplits()[splitIdx]]);
            auto& split = modelSplits.back();
            if ((split.Type == ESplitType::OnlineCtr) && (ctrTableMergePolicy == ECtrTableMergePolicy::KeepAllTables)) {
                auto& ctrBase = split.OnlineCtr.Ctr.Base;
                ctrBase.TargetBorderClassifierIdx = (*ctrTablesIndices)[ctrBase].GetResultIndex(ctrBase.TargetBorderClassifierIdx);
            }
        }
        if (leafMultiplier == 1.0) {
            TConstArrayRef<double> leafValuesRef(
                data->GetLeafValues().begin() + leafOffsets[treeIdx],
                data->GetLeafValues().begin() + leafOffsets[treeIdx]
                    + trees.GetDimensionsCount() * (1ull << data->GetTreeSizes()[treeIdx])
            );
            builder->AddTree(
                modelSplits,
                leafValuesRef,
                !streamLeafWeights ? TConstArrayRef<double>() : TConstArrayRef<double>(
                    data->GetLeafWeights().begin() + leafOffsets[treeIdx] / trees.GetDimensionsCount(),
                    data->GetLeafWeights().begin() + leafOffsets[treeIdx] / trees.GetDimensionsCount()
                        + (1ull << data->GetTreeSizes()[treeIdx])
                )
            );
        } else {
            TVector<double> leafValues(
                data->GetLeafValues().begin() + leafOffsets[treeIdx],
                data->GetLeafValues().begin() + leafOffsets[treeIdx]
                    + trees.GetDimensionsCount() * (1ull << data->GetTreeSizes()[treeIdx])
            );
            for (auto& leafValue: leafValues) {
                leafValue *= leafMultiplier;
            }
            builder->AddTree(
                modelSplits,
                leafValues,
                !streamLeafWeights ? TConstArrayRef<double>() : TConstArrayRef<double>(
                    data->GetLeafWeights().begin() + leafOffsets[treeIdx] / trees.GetDimensionsCount(),
                    (1ull << data->GetTreeSizes()[treeIdx])
                )
            );
        }
    }

    if (ctrTableMergePolicy == ECtrTableMergePolicy::KeepAllTables) {
        for (auto& [key, status] : *ctrTablesIndices) {
            status.FinishModel();
        }
    }
}

static THolder<TNonSymmetricTreeNode> GetTree(
    const TModelTrees& trees,
    double leafMultiplier,
    bool streamLeafWeights,
    int nodeIdx
) {
    const auto& data = trees.GetModelTreeData();
    const auto& nodes = data->GetNonSymmetricStepNodes();
    const auto leftDiff = nodes[nodeIdx].LeftSubtreeDiff;
    const auto rightDiff = nodes[nodeIdx].RightSubtreeDiff;

    auto tree = MakeHolder<TNonSymmetricTreeNode>();
    if (leftDiff) {
        tree->Left = GetTree(trees, leafMultiplier, streamLeafWeights, nodeIdx + leftDiff);
    }
    if (rightDiff) {
        tree->Right = GetTree(trees, leafMultiplier, streamLeafWeights, nodeIdx + rightDiff);
    }
    if (leftDiff || rightDiff) {
        const auto& binFeatures = trees.GetBinFeatures();
        tree->SplitCondition = binFeatures[data->GetTreeSplits()[nodeIdx]];
    }

    const auto& leafIdx = data->GetNonSymmetricNodeIdToLeafId()[nodeIdx];
    if (leafIdx != (ui32)-1) {
        CB_ENSURE(!leftDiff || !rightDiff, "Got a corrupted non-symmetric tree");
        const auto dimensionCount = trees.GetDimensionsCount();
        CB_ENSURE(leafIdx % dimensionCount == 0, "Got a corrupted non-symmetric tree");
        auto leaf = MakeHolder<TNonSymmetricTreeNode>();
        const auto leafValues = data->GetLeafValues();
        if (dimensionCount == 1) {
            leaf->Value = leafValues[leafIdx] * leafMultiplier;
        } else {
            const auto begin = leafValues.begin() + leafIdx;
            const auto end = begin + dimensionCount;
            leaf->Value = TVector<double>{begin, end};
            for (auto& value : std::get<TVector<double>>(leaf->Value)) {
                value *= leafMultiplier;
            }
        }
        const auto leafWeights = data->GetLeafWeights();
        if (streamLeafWeights && !leafWeights.empty()) {
            leaf->NodeWeight = leafWeights[leafIdx / dimensionCount];
        }
        if (leftDiff) {
            tree->Right = std::move(leaf);
        } else if (rightDiff) {
            tree->Left = std::move(leaf);
        } else {
            tree = std::move(leaf);
        }
    }
    return tree;
}

// overload by type of builder
static void StreamModelTreesWithoutScaleAndBiasToBuilder(
    const TModelTrees& trees,
    double leafMultiplier,
    TNonSymmetricTreeModelBuilder* builder,
    bool streamLeafWeights,
    ECtrTableMergePolicy ctrTableMergePolicy,
    THashMap<TModelCtrBaseMergeKey, TCtrTablesMergeStatus>* ctrTablesIndices
) {
    CB_ENSURE(ctrTableMergePolicy != ECtrTableMergePolicy::KeepAllTables, "KeepAllTables CTR merge policy in not yet supported for non-symmetric trees");

    Y_UNUSED(ctrTablesIndices);
    const auto& data = trees.GetModelTreeData();
    for (size_t treeIdx = 0; treeIdx < trees.GetTreeCount(); ++treeIdx) {
        builder->AddTree(
            GetTree(
                trees,
                leafMultiplier,
                streamLeafWeights,
                data->GetTreeStartOffsets()[treeIdx]));
    }
}

static void SumModelsParams(
    const TVector<const TFullModel*> modelVector,
    THashMap<TString, TString>* modelInfo
) {
    TMaybe<TString> classParams;

    auto dimensionsCount = modelVector.back()->GetDimensionsCount();

    for (auto modelIdx : xrange(modelVector.size())) {
        const auto& modelInfo = modelVector[modelIdx]->ModelInfo;
        bool paramFound = false;
        for (const auto& paramName : {"class_params", "multiclass_params"}) {
            if (modelInfo.contains(paramName)) {
                if (classParams) {
                    CB_ENSURE(
                        modelInfo.at(paramName) == *classParams,
                        "Cannot sum models with different class params"
                    );
                } else if ((modelIdx == 0) || (dimensionsCount == 1)) {
                    // it is ok only for 1-dimensional models to have only some classParams specified
                    classParams = modelInfo.at(paramName);
                } else {
                    CB_ENSURE(false, "Cannot sum multidimensional models with and without class params");
                }
                paramFound = true;
                break;
            }
        }
        if ((modelIdx != 0) && classParams && !paramFound && (dimensionsCount > 1)) {
            CB_ENSURE(false, "Cannot sum multidimensional models with and without class params");
        }
    }

    if (classParams) {
        (*modelInfo)["class_params"] = *classParams;
    } else {
        /* One-dimensional models.
         * If class labels for binary classification are present they must be the same
         */

        TMaybe<TVector<NJson::TJsonValue>> sumClassLabels;

        for (const TFullModel* model : modelVector) {
            TVector<NJson::TJsonValue> classLabels = model->GetModelClassLabels();
            if (classLabels) {
                CB_ENSURE(classLabels.size() == 2, "Expect exactly two class labels in binary classification");

                if (sumClassLabels) {
                    CB_ENSURE(classLabels == *sumClassLabels, "Cannot sum models with different class labels");
                } else {
                    sumClassLabels = std::move(classLabels);
                }
            }
        }

        if (sumClassLabels) {
            TString& paramsString = (*modelInfo)["params"];
            NJson::TJsonValue paramsJson;
            if (paramsString) {
                paramsJson = ReadTJsonValue(paramsString);
            }
            NJson::TJsonValue classNames;
            classNames.AppendValue((*sumClassLabels)[0]);
            classNames.AppendValue((*sumClassLabels)[1]);
            paramsJson["data_processing_options"]["class_names"] = std::move(classNames);
            paramsString = ToString(paramsJson);
        }
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

    if (!AllOf(modelVector, [&](const TFullModel* model) {
            return model->GetScaleAndBias().IsIdentity();
        }))
    {
        NJson::TJsonValue summandScaleAndBiases;
        for (const auto& model : modelVector) {
            NJson::TJsonValue scaleAndBias;
            scaleAndBias.InsertValue("scale", model->GetScaleAndBias().Scale);
            NJson::TJsonValue biasValue;
            auto bias = model->GetScaleAndBias().GetBiasRef();
            for (auto b : bias) {
                biasValue.AppendValue(b);
            }
            scaleAndBias.InsertValue("bias", biasValue);
            summandScaleAndBiases.AppendValue(scaleAndBias);
        }
        (*modelInfo)["summand_scale_and_biases"] = summandScaleAndBiases.GetStringRobust();
    }
}

static bool IsAllOblivious(const TVector<const TFullModel*>& modelVector) {
    return AllOf(modelVector, [] (const TFullModel* m) { return m->IsOblivious(); });
}

static bool IsAllNonSymmetric(const TVector<const TFullModel*>& modelVector) {
    return AllOf(modelVector, [] (const TFullModel* m) { return !m->IsOblivious(); });
}

template <typename TBuilderType>
static void SumModels(
    const TVector<const TFullModel*>& modelVector,
    const TVector<double>& weights,
    const TVector<TFloatFeature>& floatFeatures,
    const TVector<TCatFeature>& catFeatures,
    bool allModelsHaveLeafWeights,
    ECtrTableMergePolicy ctrMergePolicy,
    TFullModel* sum
) {
    const auto approxDimension = modelVector.back()->GetDimensionsCount();
    TBuilderType builder(floatFeatures, catFeatures, {}, {}, approxDimension);

    THashMap<TModelCtrBaseMergeKey, TCtrTablesMergeStatus> ctrTablesIndices;

    for (const auto modelId : xrange(modelVector.size())) {
        TScaleAndBias normer = modelVector[modelId]->GetScaleAndBias();
        StreamModelTreesWithoutScaleAndBiasToBuilder(
            *modelVector[modelId]->ModelTrees,
            weights[modelId] * normer.Scale,
            &builder,
            allModelsHaveLeafWeights,
            ctrMergePolicy,
            &ctrTablesIndices
        );
    }
    builder.Build(sum->ModelTrees.GetMutable());
}

TFullModel SumModels(
    const TVector<const TFullModel*> modelVector,
    const TVector<double>& weights,
    const TVector<TString>& modelParamsPrefixes,
    ECtrTableMergePolicy ctrMergePolicy
) {
    CB_ENSURE(!modelVector.empty(), "empty model vector unexpected");
    CB_ENSURE(modelVector.size() == weights.size());
    CB_ENSURE(modelParamsPrefixes.empty() || (modelVector.size() == modelParamsPrefixes.size()));

    CB_ENSURE(
        IsAllOblivious(modelVector) || IsAllNonSymmetric(modelVector),
        "Summation of symmetric and non-symmetric models is not supported [for now]");

    const auto approxDimension = modelVector.back()->GetDimensionsCount();
    size_t maxFlatFeatureVectorSize = 0;
    TVector<TIntrusivePtr<ICtrProvider>> ctrProviders;
    bool allModelsHaveLeafWeights = true;
    bool someModelHasLeafWeights = false;
    for (const auto& model : modelVector) {
        Y_ASSERT(model != nullptr);
        CB_ENSURE(
            model->ModelTrees->GetTextFeatures().empty(),
            "Models summation is not supported for models with text features"
        );
        CB_ENSURE(
            model->ModelTrees->GetEmbeddingFeatures().empty(),
            "Models summation is not supported for models with embedding features"
        );
        CB_ENSURE(
            model->GetDimensionsCount() == approxDimension,
            "Approx dimensions don't match: " << model->GetDimensionsCount() << " != "
            << approxDimension
        );
        maxFlatFeatureVectorSize = Max(
            maxFlatFeatureVectorSize,
            model->ModelTrees->GetFlatFeatureVectorExpectedSize()
        );
        ctrProviders.push_back(model->CtrProvider);
        // empty model does not disable LeafWeights:
        if (model->ModelTrees->GetModelTreeData()->GetLeafWeights().size() < model->GetTreeCount()) {
            allModelsHaveLeafWeights = false;
        }
        if (!model->ModelTrees->GetModelTreeData()->GetLeafWeights().empty()) {
            someModelHasLeafWeights = true;
        }
    }
    if (!allModelsHaveLeafWeights && someModelHasLeafWeights) {
        CATBOOST_WARNING_LOG << "Leaf weights for some models are ignored " <<
        "because not all models have leaf weights" << Endl;
    }
    TVector<TFlatFeature> flatFeatureInfoVector(maxFlatFeatureVectorSize);
    for (const auto& model : modelVector) {
        for (const auto& floatFeature : model->ModelTrees->GetFloatFeatures()) {
            flatFeatureInfoVector[floatFeature.Position.FlatIndex].SetOrCheck(floatFeature);
        }
        for (const auto& catFeature : model->ModelTrees->GetCatFeatures()) {
            flatFeatureInfoVector[catFeature.Position.FlatIndex].SetOrCheck(catFeature);
        }
    }
    TFlatFeatureMergerVisitor merger;
    for (auto& flatFeature: flatFeatureInfoVector) {
        std::visit(merger, flatFeature.FeatureVariant);
    }
    TVector<double> totalBias(approxDimension);
    for (const auto modelId : xrange(modelVector.size())) {
        TScaleAndBias normer = modelVector[modelId]->GetScaleAndBias();
        auto normerBias = normer.GetBiasRef();
        if (!normerBias.empty()) {
            CB_ENSURE(totalBias.size() == normerBias.size(), "Bias dimensions missmatch");
            for (auto dim : xrange(totalBias.size())) {
                totalBias[dim] += weights[modelId] * normerBias[dim];
            }
        }
    }
    TFullModel result;
    if (IsAllOblivious(modelVector)) {
        SumModels<TObliviousTreeBuilder>(
            modelVector,
            weights,
            merger.MergedFloatFeatures,
            merger.MergedCatFeatures,
            allModelsHaveLeafWeights,
            ctrMergePolicy,
            &result);
    } else if (IsAllNonSymmetric(modelVector)) {
        SumModels<TNonSymmetricTreeModelBuilder>(
            modelVector,
            weights,
            merger.MergedFloatFeatures,
            merger.MergedCatFeatures,
            allModelsHaveLeafWeights,
            ctrMergePolicy,
            &result);
    } else {
        CB_ENSURE_INTERNAL(false, "This should be unreachable");
    }

    for (const auto modelIdx : xrange(modelVector.size())) {
        TStringBuilder keyPrefix;
        if (modelParamsPrefixes.empty()) {
            keyPrefix << "model" << modelIdx << ":";
        } else {
            keyPrefix << modelParamsPrefixes[modelIdx];
        }
        for (const auto& [key, value]: modelVector[modelIdx]->ModelInfo) {
            result.ModelInfo[keyPrefix + key] = value;
        }
    }
    result.CtrProvider = MergeCtrProvidersData(ctrProviders, ctrMergePolicy);
    result.UpdateDynamicData();
    result.ModelInfo["model_guid"] = CreateGuidAsString();
    result.SetScaleAndBias({1, totalBias});
    SumModelsParams(modelVector, &result.ModelInfo);
    return result;
}

void SaveModelBorders(
    const TString& file,
    const TFullModel& model) {

    TOFStream out(file);

    for (const auto& feature : model.ModelTrees->GetFloatFeatures()) {
        NCB::OutputFeatureBorders(
            feature.Position.FlatIndex,
            feature.Borders,
            NanValueTreatmentToNanMode(feature.NanValueTreatment),
            &out
        );
    }
}

THashMap<int, TFloatFeature::ENanValueTreatment> GetNanTreatments(const TFullModel& model) {
    THashMap<int, TFloatFeature::ENanValueTreatment> nanTreatments;
    for (const auto& feature : model.ModelTrees->GetFloatFeatures()) {
        nanTreatments[feature.Position.FlatIndex] = feature.NanValueTreatment;
    }
    return nanTreatments;
}

DEFINE_DUMPER(TRepackedBin, FeatureIndex, XorMask, SplitIdx)

DEFINE_DUMPER(TNonSymmetricTreeStepNode, LeftSubtreeDiff, RightSubtreeDiff)

DEFINE_DUMPER(
    TModelTrees::TRuntimeData,
    BinFeatures,
    EffectiveBinFeaturesBucketCount
)

DEFINE_DUMPER(
    TModelTrees::TForApplyData,
    UsedFloatFeaturesCount,
    UsedCatFeaturesCount,
    MinimalSufficientFloatFeaturesVectorSize,
    MinimalSufficientCatFeaturesVectorSize,
    UsedModelCtrs,
    TreeFirstLeafOffsets
)

//DEFINE_DUMPER(TModelTrees),
//    TreeSplits, TreeSizes,
//    TreeStartOffsets, NonSymmetricStepNodes,
//    NonSymmetricNodeIdToLeafId, LeafValues);

TNonSymmetricTreeStepNode& TNonSymmetricTreeStepNode::operator=(const NCatBoostFbs::TNonSymmetricTreeStepNode* stepNode) {
    LeftSubtreeDiff = stepNode->LeftSubtreeDiff();
    RightSubtreeDiff = stepNode->RightSubtreeDiff();
    return *this;
}

TRepackedBin& TRepackedBin::operator=(const NCatBoostFbs::TRepackedBin* repackedBin) {
    std::tie(
        FeatureIndex,
        XorMask,
        SplitIdx
    ) = std::forward_as_tuple(
        repackedBin->FeatureIndex(),
        repackedBin->XorMask(),
        repackedBin->SplitIdx()
    );
    return *this;
}
