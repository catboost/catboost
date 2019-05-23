#include "model.h"

#include "coreml_helpers.h"
#include "flatbuffers_serializer_helper.h"
#include "formula_evaluator.h"
#include "json_model_helpers.h"
#include "model_build_helper.h"
#include "model_export/model_exporter.h"
#include "onnx_helpers.h"
#include "static_ctr_provider.h"

#include <catboost/libs/cat_feature/cat_feature.h>
#include <catboost/libs/helpers/borders_io.h>
#include <catboost/libs/logging/logging.h>
#include <catboost/libs/options/json_helper.h>
#include <catboost/libs/options/check_train_options.h>
#include <catboost/libs/options/output_file_options.h>

#include <contrib/libs/coreml/TreeEnsemble.pb.h>
#include <contrib/libs/coreml/Model.pb.h>

#include <library/json/json_reader.h>
#include <library/dbg_output/dump.h>

#include <util/generic/algorithm.h>
#include <util/generic/buffer.h>
#include <util/generic/fwd.h>
#include <util/generic/guid.h>
#include <util/generic/variant.h>
#include <util/generic/xrange.h>
#include <util/generic/ylimits.h>
#include <util/string/builder.h>
#include <util/stream/buffer.h>
#include <util/stream/file.h>
#include <util/system/fs.h>
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

static NJson::TJsonValue RemoveInvalidParams(const NJson::TJsonValue& params) {
    try {
        CheckFitParams(params);
        return params;
    } catch (...) {
        CATBOOST_WARNING_LOG << "There are invalid params and some of them will be ignored." << Endl;
    }
    NJson::TJsonValue result(NJson::JSON_MAP);
    // TODO(sergmiller): make proper validation for each parameter separately
    for (const auto& param : params.GetMap()) {
        result[param.first] = param.second;

        try {
            CheckFitParams(result);
        } catch (...) {
            result.EraseValue(param.first);

            NJson::TJsonValue badParam;
            badParam[param.first] = param.second;
            CATBOOST_WARNING_LOG << "Parameter " << ToString<NJson::TJsonValue>(badParam)
                << " is ignored, because it cannot be parsed." << Endl;
        }
    }
    return result;
}

TFullModel ReadModel(IInputStream* modelStream, EModelType format) {
    TFullModel model;
    if (format == EModelType::CatboostBinary) {
        Load(modelStream, model);
    } else if (format == EModelType::Json) {
        NJson::TJsonValue jsonModel = NJson::ReadJsonTree(modelStream);
        ConvertJsonToCatboostModel(jsonModel, &model);
    } else {
        CoreML::Specification::Model coreMLModel;
        CB_ENSURE(coreMLModel.ParseFromString(modelStream->ReadAll()), "coreml model deserialization failed");
        NCatboost::NCoreML::ConvertCoreMLToCatboostModel(coreMLModel, &model);
    }
    if (model.ModelInfo.contains("params")) {
        NJson::TJsonValue paramsJson = ReadTJsonValue(model.ModelInfo.at("params"));
        paramsJson["flat_params"] = RemoveInvalidParams(paramsJson["flat_params"]);
        model.ModelInfo["params"] = ToString<NJson::TJsonValue>(paramsJson);
    }
    return model;
}

TFullModel ReadModel(const TString& modelFile, EModelType format){
    CB_ENSURE(NFs::Exists(modelFile), "Model file doesn't exist: " << modelFile);
    TIFStream f(modelFile);
    return ReadModel(&f, format);
}

TFullModel ReadModel(const void* binaryBuffer, size_t binaryBufferSize, EModelType format)  {
    TBuffer buf((char*)binaryBuffer, binaryBufferSize);
    TBufferInput bs(buf);
    return ReadModel(&bs, format);
}

void OutputModelCoreML(
    const TFullModel& model,
    const TString& modelFile,
    const NJson::TJsonValue& userParameters,
    const THashMap<ui32, TString>* catFeaturesHashToString) {

    CoreML::Specification::Model treeModel;
    treeModel.set_specificationversion(1);

    auto regressor = treeModel.mutable_treeensembleregressor();
    auto ensemble = regressor->mutable_treeensemble();

    bool createPipelineModel;
    NCatboost::NCoreML::ConfigureTrees(model, ensemble, &createPipelineModel);

    TString data;
    if (createPipelineModel) {
        CoreML::Specification::Model pipelineModel;
        pipelineModel.set_specificationversion(1);

        auto* container = pipelineModel.mutable_pipeline()->mutable_models();
        NCatboost::NCoreML::ConfigureCategoricalMappings(model, catFeaturesHashToString, container);

        auto* contained = container->Add();
        auto treeDescription = treeModel.mutable_description();
        NCatboost::NCoreML::ConfigureTreeModelIO(model, userParameters, regressor, treeDescription);
        *contained = treeModel;

        auto pipelineDescription = pipelineModel.mutable_description();
        NCatboost::NCoreML::ConfigureMetadata(model, userParameters, pipelineDescription);
        NCatboost::NCoreML::ConfigurePipelineModelIO(model, pipelineDescription);

        pipelineModel.SerializeToString(&data);
    } else {
        auto description = treeModel.mutable_description();
        NCatboost::NCoreML::ConfigureMetadata(model, userParameters, description);
        NCatboost::NCoreML::ConfigureTreeModelIO(model, userParameters, regressor, description);
        treeModel.SerializeToString(&data);
    }

    TOFStream out(modelFile);
    out.Write(data);
}

void OutputModelOnnx(
    const TFullModel& model,
    const TString& modelFile,
    const NJson::TJsonValue& userParameters) {

    /* TODO(akhropov): the problem with OneHotFeatures is that raw 'float' values
     * could be interpreted as nans so that equality comparison won't work for such splits
     */
    CB_ENSURE(
        !model.HasCategoricalFeatures(),
        "ONNX-ML format export does yet not support categorical features"
    );

    onnx::ModelProto outModel;

    NCatboost::NOnnx::InitMetadata(model, userParameters, &outModel);

    TMaybe<TString> graphName;
    if (userParameters.Has("onnx_graph_name")) {
        graphName = userParameters["onnx_graph_name"].GetStringSafe();
    }

    NCatboost::NOnnx::ConvertTreeToOnnxGraph(model, graphName, outModel.mutable_graph());

    TString data;
    outModel.SerializeToString(&data);

    TOFStream out(modelFile);
    out.Write(data);
}

void ExportModel(
    const TFullModel& model,
    const TString& modelFile,
    const EModelType format,
    const TString& userParametersJson,
    bool addFileFormatExtension,
    const TVector<TString>* featureId,
    const THashMap<ui32, TString>* catFeaturesHashToString) {

    //TODO(eermishkina): support non symmetric trees
    CB_ENSURE(model.IsOblivious() || format == EModelType::CatboostBinary, "Can save non symmetric trees only in cbm format");
    const auto modelFileName = NCatboostOptions::AddExtension(format, modelFile, addFileFormatExtension);
    switch (format) {
        case EModelType::CatboostBinary:
            CB_ENSURE(
                userParametersJson.empty(),
                "JSON user params for CatBoost model export are not supported"
            );
            OutputModel(model, modelFileName);
            break;
        case EModelType::AppleCoreML:
            {
                TStringInput is(userParametersJson);
                NJson::TJsonValue params;
                NJson::ReadJsonTree(&is, &params);

                OutputModelCoreML(model, modelFileName, params, catFeaturesHashToString);
            }
            break;
        case EModelType::Json:
            {
                CB_ENSURE(
                    userParametersJson.empty(),
                    "JSON user params for CatBoost model export are not supported"
                );

                OutputModelJson(model, modelFileName, featureId, catFeaturesHashToString);
            }
            break;
        case EModelType::Onnx:
            {
                TStringInput is(userParametersJson);
                NJson::TJsonValue params;
                NJson::ReadJsonTree(&is, &params);

                OutputModelOnnx(model, modelFileName, params);
            }
            break;
        default:
            TIntrusivePtr<NCatboost::ICatboostModelExporter> modelExporter
                = NCatboost::CreateCatboostModelExporter(
                    modelFile,
                    format,
                    userParametersJson,
                    addFileFormatExtension
                );
            if (!modelExporter) {
                TStringBuilder err;
                err << "Export to " << format << " format is not supported";
                CB_ENSURE(false, err.c_str());
            }
            modelExporter->Write(model, catFeaturesHashToString);
            break;
    }
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

void TObliviousTrees::TruncateTrees(size_t begin, size_t end) {
    //TODO(eermishkina): support non symmetric trees
    CB_ENSURE(IsOblivious(), "Truncate support only symmetric trees");
    CB_ENSURE(begin <= end, "begin tree index should be not greater than end tree index.");
    CB_ENSURE(end <= TreeSplits.size(), "end tree index should be not greater than tree count.");
    TObliviousTreeBuilder builder(FloatFeatures, CatFeatures, ApproxDimension);
    const auto& leafOffsets = RuntimeData->TreeFirstLeafOffsets;
    for (size_t treeIdx = begin; treeIdx < end; ++treeIdx) {
        TVector<TModelSplit> modelSplits;
        for (int splitIdx = TreeStartOffsets[treeIdx];
             splitIdx < TreeStartOffsets[treeIdx] + TreeSizes[treeIdx];
             ++splitIdx)
        {
            modelSplits.push_back(RuntimeData->BinFeatures[TreeSplits[splitIdx]]);
        }
        TArrayRef<double> leafValuesRef(
            LeafValues.begin() + leafOffsets[treeIdx],
            LeafValues.begin() + leafOffsets[treeIdx] + ApproxDimension * (1u << TreeSizes[treeIdx])
        );
        builder.AddTree(modelSplits, leafValuesRef, LeafWeights.empty() ? TVector<double>() : LeafWeights[treeIdx]);
    }
    *this = builder.Build();
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
    std::vector<flatbuffers::Offset<NCatBoostFbs::TOneHotFeature>> oneHotFeaturesOffsets;
    for (const auto& oneHotFeature : OneHotFeatures) {
        oneHotFeaturesOffsets.push_back(oneHotFeature.FBSerialize(serializer.FlatbufBuilder));
    }
    std::vector<flatbuffers::Offset<NCatBoostFbs::TCtrFeature>> ctrFeaturesOffsets;
    for (const auto& ctrFeature : CtrFeatures) {
        ctrFeaturesOffsets.push_back(ctrFeature.FBSerialize(serializer));
    }
    TVector<double> flatLeafWeights;
    for (const auto& oneTreeLeafWeights: LeafWeights) {
        flatLeafWeights.insert(
            flatLeafWeights.end(),
            oneTreeLeafWeights.begin(),
            oneTreeLeafWeights.end()
        );
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
        &flatLeafWeights,
        &fbsNonSymmetricTreeStepNode,
        &NonSymmetricNodeIdToLeafId
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
    ref.MinimalSufficientFloatFeaturesVectorSize = 0;
    ref.MinimalSufficientCatFeaturesVectorSize = 0;
    for (const auto& feature : FloatFeatures) {
        if (!feature.UsedInModel()) {
            continue;
        }
        ++ref.UsedFloatFeaturesCount;
        ref.MinimalSufficientFloatFeaturesVectorSize = static_cast<size_t>(feature.FeatureIndex) + 1;
        for (int borderId = 0; borderId < feature.Borders.ysize(); ++borderId) {
            TFloatSplit fs{feature.FeatureIndex, feature.Borders[borderId]};
            ref.BinFeatures.emplace_back(fs);
            auto& bf = splitIds.emplace_back();
            bf.FeatureIdx = ref.EffectiveBinFeaturesBucketCount + borderId / MAX_VALUES_PER_BIN;
            bf.SplitIdx = (borderId % MAX_VALUES_PER_BIN) + 1;
        }
        ref.EffectiveBinFeaturesBucketCount
            += (feature.Borders.size() + MAX_VALUES_PER_BIN - 1) / MAX_VALUES_PER_BIN;
    }
    for (const auto& feature : CatFeatures) {
        if (!feature.UsedInModel) {
            continue;
        }
        ++ref.UsedCatFeaturesCount;
        ref.MinimalSufficientCatFeaturesVectorSize = static_cast<size_t>(feature.FeatureIndex) + 1;
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
    EraseIf(CatFeatures, [](const TCatFeature& feature) { return !feature.UsedInModel; });
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

void TFullModel::CalcFlat(
    TConstArrayRef<TConstArrayRef<float>> features,
    size_t treeStart,
    size_t treeEnd,
    TArrayRef<double> results) const {

    const auto expectedFlatVecSize = ObliviousTrees.GetFlatFeatureVectorExpectedSize();
    for (const auto& flatFeaturesVec : features) {
        CB_ENSURE(
            flatFeaturesVec.size() >= expectedFlatVecSize,
            "insufficient flat features vector size: " << flatFeaturesVec.size()
            << " expected: " << expectedFlatVecSize
        );
    }
    CalcGeneric(
        *this,
        [&features](const TFloatFeature& floatFeature, size_t index) -> float {
            return features[index][floatFeature.FlatFeatureIndex];
        },
        [&features](const TCatFeature& catFeature, size_t index) -> int {
            return ConvertFloatCatFeatureToIntHash(features[index][catFeature.FlatFeatureIndex]);
        },
        features.size(),
        treeStart,
        treeEnd,
        results
    );
}

void TFullModel::CalcFlatSingle(
    TConstArrayRef<float> features,
    size_t treeStart,
    size_t treeEnd,
    TArrayRef<double> results) const {

    CB_ENSURE(
        ObliviousTrees.GetFlatFeatureVectorExpectedSize() <= features.size(),
        "Not enough features provided"
    );
    CalcGeneric(
        *this,
        [&features](const TFloatFeature& floatFeature, size_t ) -> float {
            return features[floatFeature.FlatFeatureIndex];
        },
        [&features](const TCatFeature& catFeature, size_t ) -> int {
            return ConvertFloatCatFeatureToIntHash(features[catFeature.FlatFeatureIndex]);
        },
        1,
        treeStart,
        treeEnd,
        results
    );
}

void TFullModel::CalcFlatTransposed(
    TConstArrayRef<TConstArrayRef<float>> transposedFeatures,
    size_t treeStart,
    size_t treeEnd,
    TArrayRef<double> results) const {

    CB_ENSURE(
        ObliviousTrees.GetFlatFeatureVectorExpectedSize() <= transposedFeatures.size(),
        "Not enough features provided"
    );
    TMaybe<size_t> docCount;
    CB_ENSURE(!ObliviousTrees.FloatFeatures.empty() || !ObliviousTrees.CatFeatures.empty(),
        "Both float features and categorical features information are empty");
    if (!ObliviousTrees.FloatFeatures.empty()) {
        for (const auto& floatFeature : ObliviousTrees.FloatFeatures) {
            if (floatFeature.UsedInModel()) {
                docCount = transposedFeatures[floatFeature.FlatFeatureIndex].size();
                break;
            }
        }
    }
    if (!docCount.Defined() && !ObliviousTrees.CatFeatures.empty()) {
        for (const auto& catFeature : ObliviousTrees.CatFeatures) {
            if (catFeature.UsedInModel) {
                docCount = transposedFeatures[catFeature.FlatFeatureIndex].size();
                break;
            }
        }
    }
    CB_ENSURE(docCount.Defined(), "couldn't determine document count, something went wrong");
    CalcGeneric(
        *this,
        [&transposedFeatures](const TFloatFeature& floatFeature, size_t index) -> float {
            return transposedFeatures[floatFeature.FlatFeatureIndex][index];
        },
        [&transposedFeatures](const TCatFeature& catFeature, size_t index) -> int {
            return ConvertFloatCatFeatureToIntHash(transposedFeatures[catFeature.FlatFeatureIndex][index]);
        },
        *docCount,
        treeStart,
        treeEnd,
        results
    );
}

template <typename TCatFeatureContainer = TConstArrayRef<int>>
static void ValidateInputFeatures(
    const TFullModel& model,
    TConstArrayRef<TConstArrayRef<float>> floatFeatures,
    TConstArrayRef<TCatFeatureContainer> catFeatures
) {
    if (!floatFeatures.empty() && !catFeatures.empty()) {
        CB_ENSURE(catFeatures.size() == floatFeatures.size());
    }
    CB_ENSURE(
        model.ObliviousTrees.GetUsedFloatFeaturesCount() == 0 || !floatFeatures.empty(),
        "Model has float features but no float features provided"
    );
    CB_ENSURE(
        model.ObliviousTrees.GetUsedCatFeaturesCount() == 0 || !catFeatures.empty(),
        "Model has categorical features but no categorical features provided"
    );
    for (const auto& floatFeaturesVec : floatFeatures) {
        CB_ENSURE(
            floatFeaturesVec.size() >= model.ObliviousTrees.GetMinimalSufficientFloatFeaturesVectorSize(),
            "insufficient float features vector size: " << floatFeaturesVec.size() << " expected: " <<
            model.ObliviousTrees.GetMinimalSufficientFloatFeaturesVectorSize()
        );
    }
    for (const auto& catFeaturesVec : catFeatures) {
        CB_ENSURE(
            catFeaturesVec.size() >= model.ObliviousTrees.GetMinimalSufficientCatFeaturesVectorSize(),
            "insufficient cat features vector size: " << catFeaturesVec.size() << " expected: " <<
            model.ObliviousTrees.GetMinimalSufficientCatFeaturesVectorSize()
        );
    }
}

void TFullModel::Calc(
    TConstArrayRef<TConstArrayRef<float>> floatFeatures,
    TConstArrayRef<TConstArrayRef<int>> catFeatures,
    size_t treeStart,
    size_t treeEnd,
    TArrayRef<double> results
) const {
    ValidateInputFeatures(*this, floatFeatures, catFeatures);
    const size_t docCount = Max(catFeatures.size(), floatFeatures.size());
    CalcGeneric(
        *this,
        [&floatFeatures](const TFloatFeature& floatFeature, size_t index) -> float {
            return floatFeatures[index][floatFeature.FeatureIndex];
        },
        [&catFeatures](const TCatFeature& catFeature, size_t index) -> int {
            return catFeatures[index][catFeature.FeatureIndex];
        },
        docCount,
        treeStart,
        treeEnd,
        results
    );
}

void TFullModel::Calc(
    TConstArrayRef<TConstArrayRef<float>> floatFeatures,
    TConstArrayRef<TVector<TStringBuf>> catFeatures,
    size_t treeStart,
    size_t treeEnd,
    TArrayRef<double> results
) const {
    ValidateInputFeatures(*this, floatFeatures, catFeatures);
    const size_t docCount = Max(catFeatures.size(), floatFeatures.size());
    CalcGeneric(
        *this,
        [&floatFeatures](const TFloatFeature& floatFeature, size_t index) -> float {
            return floatFeatures[index][floatFeature.FeatureIndex];
        },
        [&catFeatures](const TCatFeature& catFeature, size_t index) -> int {
            return CalcCatFeatureHash(catFeatures[index][catFeature.FeatureIndex]);
        },
        docCount,
        treeStart,
        treeEnd,
        results
    );
}

TVector<TVector<double>> TFullModel::CalcTreeIntervals(
    TConstArrayRef<TConstArrayRef<float>> floatFeatures,
    TConstArrayRef<TConstArrayRef<int>> catFeatures,
    size_t incrementStep
) const {
    ValidateInputFeatures(*this, floatFeatures, catFeatures);
    const size_t docCount = Max(catFeatures.size(), floatFeatures.size());
    return CalcTreeIntervalsGeneric(
        *this,
        [&floatFeatures](const TFloatFeature& floatFeature, size_t index) -> float {
            return floatFeatures[index][floatFeature.FeatureIndex];
        },
        [&catFeatures](const TCatFeature& catFeature, size_t index) -> int {
            return catFeatures[index][catFeature.FeatureIndex];
        },
        docCount,
        incrementStep
    );
}
TVector<TVector<double>> TFullModel::CalcTreeIntervalsFlat(
    TConstArrayRef<TConstArrayRef<float>> features,
    size_t incrementStep
) const {
    const auto expectedFlatVecSize = ObliviousTrees.GetFlatFeatureVectorExpectedSize();
    for (const auto& flatFeaturesVec : features) {
        CB_ENSURE(
            flatFeaturesVec.size() >= expectedFlatVecSize,
            "insufficient flat features vector size: " << flatFeaturesVec.size()
            << " expected: " << expectedFlatVecSize
        );
    }
    return CalcTreeIntervalsGeneric(
        *this,
        [&features](const TFloatFeature& floatFeature, size_t index) -> float {
            return features[index][floatFeature.FlatFeatureIndex];
        },
        [&features](const TCatFeature& catFeature, size_t index) -> int {
            return ConvertFloatCatFeatureToIntHash(features[index][catFeature.FlatFeatureIndex]);
        },
        features.size(),
        incrementStep
    );
}

void TFullModel::CalcLeafIndexesSingle(
    TConstArrayRef<float> floatFeatures,
    TConstArrayRef<TStringBuf> catFeatures,
    size_t treeStart,
    size_t treeEnd,
    TArrayRef<ui32> indexes
) const {
    ValidateInputFeatures<TConstArrayRef<TStringBuf>>(*this, {floatFeatures}, {catFeatures});
    CalcLeafIndexesGeneric(
        *this,
        [&floatFeatures](const TFloatFeature& floatFeature, size_t) -> float {
            return floatFeatures[floatFeature.FeatureIndex];
        },
        [&catFeatures](const TCatFeature& catFeature, size_t) -> int {
            return CalcCatFeatureHash(catFeatures[catFeature.FeatureIndex]);
        },
        1,
        treeStart,
        treeEnd,
        indexes
    );
}

void TFullModel::CalcLeafIndexes(
    TConstArrayRef<TConstArrayRef<float>> floatFeatures,
    TConstArrayRef<TVector<TStringBuf>> catFeatures,
    size_t treeStart,
    size_t treeEnd,
    TArrayRef<ui32> indexes
) const {
    ValidateInputFeatures(*this, floatFeatures, catFeatures);
    const size_t docCount = Max(catFeatures.size(), floatFeatures.size());
    CB_ENSURE(docCount * (treeEnd - treeStart) == indexes.size(), LabeledOutput(docCount * (treeEnd - treeStart), indexes.size()));
    CalcLeafIndexesGeneric(
        *this,
        [&floatFeatures](const TFloatFeature& floatFeature, size_t index) -> float {
            return floatFeatures[index][floatFeature.FeatureIndex];
        },
        [&catFeatures](const TCatFeature& catFeature, size_t index) -> int {
            return CalcCatFeatureHash(catFeatures[index][catFeature.FeatureIndex]);
        },
        docCount,
        treeStart,
        treeEnd,
        indexes
    );
}


void TFullModel::Save(IOutputStream* s) const {
    using namespace flatbuffers;
    using namespace NCatBoostFbs;
    ::Save(s, GetModelFormatDescriptor());
    TModelPartsCachingSerializer serializer;
    auto obliviousTreesOffset = ObliviousTrees.FBSerialize(serializer);
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
        ObliviousTrees.FBDeserialize(fbModelCore->ObliviousTrees());
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
        CB_ENSURE(modelParts.size() == 1, "only single part model supported now");
        CtrProvider = new TStaticCtrProvider;
        CB_ENSURE(modelParts[0] == CtrProvider->ModelPartIdentifier(), "only static ctr models supported");
        CtrProvider->Load(s);
    }
    UpdateDynamicData();
}

TVector<TString> GetModelUsedFeaturesNames(const TFullModel& model) {
    TVector<int> featuresIdxs;
    TVector<TString> featuresNames;
    const TObliviousTrees& forest = model.ObliviousTrees;

    for (const TFloatFeature& feature : forest.FloatFeatures) {
        featuresIdxs.push_back(feature.FlatFeatureIndex);
        featuresNames.push_back(
            feature.FeatureId == "" ? ToString(feature.FlatFeatureIndex) : feature.FeatureId
        );
    }
    for (const TCatFeature& feature : forest.CatFeatures) {
        featuresIdxs.push_back(feature.FlatFeatureIndex);
        featuresNames.push_back(
            feature.FeatureId == "" ? ToString(feature.FlatFeatureIndex) : feature.FeatureId
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

inline TVector<TString> ExtractClassNamesFromJsonArray(const NJson::TJsonValue& arr) {
    TVector<TString> classNames;
    for (const auto& token : arr.GetArraySafe()) {
        classNames.push_back(token.GetStringSafe());
    }
    return classNames;
}

TVector<TString> GetModelClassNames(const TFullModel& model) {
    TVector<TString> classNames;

    if (model.ModelInfo.contains("multiclass_params")) {
        NJson::TJsonValue paramsJson = ReadTJsonValue(model.ModelInfo.at("multiclass_params"));
        classNames = ExtractClassNamesFromJsonArray(paramsJson["class_names"]);
    } else if (model.ModelInfo.contains("params")) {
        const TString& modelInfoParams = model.ModelInfo.at("params");
        NJson::TJsonValue paramsJson = ReadTJsonValue(modelInfoParams);
        if (paramsJson.Has("data_processing_options")
            && paramsJson["data_processing_options"].Has("class_names")) {
            classNames = ExtractClassNamesFromJsonArray(paramsJson["data_processing_options"]["class_names"]);
        }
    }
    return classNames;
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
                other.FlatFeatureIndex
            );
            TFeatureType& feature = Get<TFeatureType>(FeatureVariant);
            CB_ENSURE(feature.FlatFeatureIndex == other.FlatFeatureIndex);
            CB_ENSURE(
                feature.FeatureIndex == other.FeatureIndex,
                "Internal feature index mismatch: " << feature.FeatureIndex << " != "
                << other.FeatureIndex << " flat feature index: " << feature.FlatFeatureIndex
            );
            CB_ENSURE(
                feature.FeatureId.empty() || feature.FeatureId == other.FeatureId,
                "Feature name mismatch: " << feature.FeatureId << " != " << other.FeatureId
                << " flat feature index: " << feature.FlatFeatureIndex
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
    for (size_t treeIdx = 0; treeIdx < trees.TreeSizes.size(); ++treeIdx) {
        TVector<TModelSplit> modelSplits;
        for (int splitIdx = trees.TreeStartOffsets[treeIdx];
             splitIdx < trees.TreeStartOffsets[treeIdx] + trees.TreeSizes[treeIdx];
             ++splitIdx)
        {
            modelSplits.push_back(binFeatures[trees.TreeSplits[splitIdx]]);
        }
        TConstArrayRef<double> treeLeafWeights;
        if (streamLeafWeights) {
            treeLeafWeights = trees.LeafWeights[treeIdx];
        }
        if (leafMultiplier == 1.0) {
            TConstArrayRef<double> leafValuesRef(
                trees.LeafValues.begin() + leafOffsets[treeIdx],
                trees.LeafValues.begin() + leafOffsets[treeIdx]
                    + trees.ApproxDimension * (1u << trees.TreeSizes[treeIdx])
            );
            builder->AddTree(modelSplits, leafValuesRef, treeLeafWeights);
        } else {
            TVector<double> leafValues(
                trees.LeafValues.begin() + leafOffsets[treeIdx],
                trees.LeafValues.begin() + leafOffsets[treeIdx]
                    + trees.ApproxDimension * (1u << trees.TreeSizes[treeIdx])
            );
            for (auto& leafValue: leafValues) {
                leafValue *= leafMultiplier;
            }
            builder->AddTree(modelSplits, leafValues, treeLeafWeights);
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
    const auto approxDimension = modelVector.back()->ObliviousTrees.ApproxDimension;
    size_t maxFlatFeatureVectorSize = 0;
    TVector<TIntrusivePtr<ICtrProvider>> ctrProviders;
    bool allModelsHaveLeafWeights = true;
    bool someModelHasLeafWeights = false;
    for (const auto& model : modelVector) {
        Y_ASSERT(model != nullptr);
        //TODO(eermishkina): support non symmetric trees
        CB_ENSURE(model->IsOblivious(), "Models summation supported only for symmetric trees");
        CB_ENSURE(
            model->ObliviousTrees.ApproxDimension == approxDimension,
            "Approx dimensions don't match: " << model->ObliviousTrees.ApproxDimension << " != "
            << approxDimension
        );
        maxFlatFeatureVectorSize = Max(
            maxFlatFeatureVectorSize,
            model->ObliviousTrees.GetFlatFeatureVectorExpectedSize()
        );
        ctrProviders.push_back(model->CtrProvider);
        // empty model does not disable LeafWeights:
        if (model->ObliviousTrees.LeafWeights.size() < model->GetTreeCount()) {
            allModelsHaveLeafWeights = false;
        }
        if (!model->ObliviousTrees.LeafWeights.empty()) {
            someModelHasLeafWeights = true;
        }
    }
    if (!allModelsHaveLeafWeights && someModelHasLeafWeights) {
        CATBOOST_WARNING_LOG << "Leaf weights for some models are ignored " <<
        "because not all models have leaf weights" << Endl;
    }
    TVector<TFlatFeature> flatFeatureInfoVector(maxFlatFeatureVectorSize);
    for (const auto& model : modelVector) {
        for (const auto& floatFeature : model->ObliviousTrees.FloatFeatures) {
            flatFeatureInfoVector[floatFeature.FlatFeatureIndex].SetOrCheck(floatFeature);
        }
        for (const auto& catFeature : model->ObliviousTrees.CatFeatures) {
            flatFeatureInfoVector[catFeature.FlatFeatureIndex].SetOrCheck(catFeature);
        }
    }
    TFlatFeatureMergerVisitor merger;
    for (auto& flatFeature: flatFeatureInfoVector) {
        Visit(merger, flatFeature.FeatureVariant);
    }
    TObliviousTreeBuilder builder(merger.MergedFloatFeatures, merger.MergedCatFeatures, approxDimension);
    for (const auto modelId : xrange(modelVector.size())) {
        StreamModelTreesToBuilder(
            modelVector[modelId]->ObliviousTrees,
            weights[modelId],
            &builder,
            allModelsHaveLeafWeights
        );
    }
    TFullModel result;
    result.ObliviousTrees = builder.Build();
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
    return result;
}

void SaveModelBorders(
    const TString& file,
    const TFullModel& model) {

    TOFStream out(file);

    for (const auto& feature : model.ObliviousTrees.FloatFeatures) {
        NCB::OutputFeatureBorders(feature.FlatFeatureIndex, feature.Borders, NanValueTreatmentToNanMode(feature.NanValueTreatment), out);
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

DEFINE_DUMPER(TObliviousTrees,
    TreeSplits, TreeSizes,
    TreeStartOffsets, NonSymmetricStepNodes,
    NonSymmetricNodeIdToLeafId, LeafValues);
