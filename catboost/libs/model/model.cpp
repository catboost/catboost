#include "model.h"
#include "coreml_helpers.h"
#include "formula_evaluator.h"
#include "static_ctr_provider.h"
#include "flatbuffers_serializer_helper.h"
#include "model_export/model_exporter.h"

#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/options/json_helper.h>
#include <catboost/libs/options/check_train_options.h>
#include <catboost/libs/logging/logging.h>

#include <contrib/libs/coreml/TreeEnsemble.pb.h>
#include <contrib/libs/coreml/Model.pb.h>

#include <library/json/json_reader.h>

#include <util/string/builder.h>
#include <util/stream/buffer.h>
#include <util/stream/str.h>
#include <util/stream/file.h>

static const char MODEL_FILE_DESCRIPTOR_CHARS[4] = {'C', 'B', 'M', '1'};

ui32 GetModelFormatDescriptor() {
    return *reinterpret_cast<const ui32*>(MODEL_FILE_DESCRIPTOR_CHARS);
}

static const char* CURRENT_CORE_FORMAT_STRING = "FlabuffersModel_v1";

void OutputModel(const TFullModel& model, const TString& modelFile) {
    TOFStream f(modelFile);
    Save(&f, model);
}

static NJson::TJsonValue RemoveInvalidParams(const NJson::TJsonValue& params) {
    NJson::TJsonValue result(NJson::JSON_MAP);
    for (const auto& param : params.GetMap()) {
        NJson::TJsonValue paramToTest;
        paramToTest[param.first] = param.second;

        try {
            CheckFitParams(paramToTest);
            result[param.first] = param.second;
        }
        catch (...) {
            MATRIXNET_WARNING_LOG << "Parameter " << ToString<NJson::TJsonValue>(paramToTest) << " is ignored, because it cannot be parsed." << Endl;
        }
    }
    return result;
}

TFullModel ReadModel(IInputStream* modelStream, EModelType format) {
    TFullModel model;
    if (format == EModelType::CatboostBinary) {
        Load(modelStream, model);
        NJson::TJsonValue paramsJson = ReadTJsonValue(model.ModelInfo.at("params"));
        paramsJson["flat_params"] = RemoveInvalidParams(paramsJson["flat_params"]);
        model.ModelInfo["params"] = ToString<NJson::TJsonValue>(paramsJson);
    } else {
        CoreML::Specification::Model coreMLModel;
        CB_ENSURE(coreMLModel.ParseFromString(modelStream->ReadAll()), "coreml model deserialization failed");
        NCatboost::NCoreML::ConvertCoreMLToCatboostModel(coreMLModel, &model);
    }
    return model;
}

TFullModel ReadModel(const TString& modelFile, EModelType format) {
    TIFStream f(modelFile);
    return ReadModel(&f, format);
}

TFullModel ReadModel(const void* binaryBuffer, size_t binaryBufferSize, EModelType format)  {
    TBuffer buf((char*)binaryBuffer, binaryBufferSize);
    TBufferInput bs(buf);
    return ReadModel(&bs, format);
}

void OutputModelCoreML(const TFullModel& model, const TString& modelFile, const NJson::TJsonValue& userParameters) {
    CoreML::Specification::Model outModel;
    outModel.set_specificationversion(1);

    auto regressor = outModel.mutable_treeensembleregressor();
    auto ensemble = regressor->mutable_treeensemble();
    auto description = outModel.mutable_description();

    NCatboost::NCoreML::ConfigureMetadata(model, userParameters, description);
    NCatboost::NCoreML::ConfigureTrees(model, ensemble);
    NCatboost::NCoreML::ConfigureIO(model, userParameters, regressor, description);

    TString data;
    outModel.SerializeToString(&data);

    TOFStream out(modelFile);
    out.Write(data);
}

void ExportModel(const TFullModel& model, const TString& modelFile, const EModelType format, const TString& userParametersJSON, bool addFileFormatExtension) {
    switch (format) {
        case EModelType::CatboostBinary:
            CB_ENSURE(userParametersJSON.empty(), "JSON user params for CatBoost model export are not supported");
            OutputModel(model, addFileFormatExtension ? modelFile + ".bin" : modelFile);
            break;
        case EModelType::AppleCoreML:
            {
                TStringInput is(userParametersJSON);
                NJson::TJsonValue params;
                NJson::ReadJsonTree(&is, &params);

                OutputModelCoreML(model, addFileFormatExtension ? modelFile + ".mlmodel" : modelFile, params);
            }
            break;
        default:
            TIntrusivePtr<NCatboost::ICatboostModelExporter> modelExporter = NCatboost::CreateCatboostModelExporter(modelFile, format, userParametersJSON, addFileFormatExtension);
            if (!modelExporter) {
                TStringBuilder err;
                err << "Export to " << format << " format is not supported";
                CB_ENSURE(false, err.c_str());
            }
            modelExporter->Write(model);
            break;
    }
}

TString SerializeModel(const TFullModel& model) {
    TStringStream ss;
    Save(&ss, model);
    return ss.Str();
}

TFullModel DeserializeModel(const TString& serializeModelString) {
    TStringStream ss(serializeModelString);
    TFullModel model;
    Load(&ss, model);
    return model;
}

namespace {
    template<typename T>
    void TruncateVector(const size_t begin, const size_t end, TVector<T>* vector) {
        CB_ENSURE(begin <= end);
        CB_ENSURE(begin <= vector->size());
        CB_ENSURE(end <= vector->size());
        vector->erase(vector->begin(), vector->begin() + begin);
        vector->erase(vector->begin() + (end - begin), vector->end());
    }
}

void TObliviousTrees::Truncate(size_t begin, size_t end) {
    auto treeBinStart = TreeSplits.begin() + TreeStartOffsets[begin];
    TreeSplits.erase(TreeSplits.begin(), treeBinStart);
    TruncateVector(begin, end, &TreeSizes);
    TreeStartOffsets.resize(TreeSizes.size());
    if (!TreeSizes.empty()) {
        TreeStartOffsets[0] = 0;
        for (size_t i = 1; i < TreeSizes.size(); ++i) {
            TreeStartOffsets[i] = TreeStartOffsets[i - 1] + TreeSizes[i - 1];
        }
    }
    TruncateVector(begin, end, &LeafValues);
    UpdateMetadata();
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
    TVector<double> flatLeafValues;
    for (const auto& oneTreeLeafValues: LeafValues) {
        flatLeafValues.insert(
            flatLeafValues.end(),
            oneTreeLeafValues.begin(),
            oneTreeLeafValues.end()
        );
    }
    TVector<double> flatLeafWeights;
    for (const auto& oneTreeLeafWeights: LeafWeights) {
        flatLeafWeights.insert(
                flatLeafWeights.end(),
                oneTreeLeafWeights.begin(),
                oneTreeLeafWeights.end()
        );
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
        &flatLeafValues,
        &flatLeafWeights
    );
}

void TObliviousTrees::UpdateMetadata() const {
    struct TFeatureSplitId {
        ui32 FeatureIdx = 0;
        ui32 SplitIdx = 0;
    };
    MetaData = TMetaData{}; // reset metadata
    TVector<TFeatureSplitId> splitIds;
    auto& ref = MetaData.GetRef();
    for (const auto& ctrFeature : CtrFeatures) {
        ref.UsedModelCtrs.push_back(ctrFeature.Ctr);
    }
    ref.EffectiveBinFeaturesBucketCount = 0;
    for (size_t i = 0; i < FloatFeatures.size(); ++i) {
        const auto& feature = FloatFeatures[i];
        for (int borderId = 0; borderId < feature.Borders.ysize(); ++borderId) {
            TFloatSplit fs{feature.FeatureIndex, feature.Borders[borderId]};
            ref.BinFeatures.emplace_back(fs);
            auto& bf = splitIds.emplace_back();
            bf.FeatureIdx = ref.EffectiveBinFeaturesBucketCount;
            bf.SplitIdx = borderId + 1;
        }
        ++ref.EffectiveBinFeaturesBucketCount;
    }
    for (size_t i = 0; i < OneHotFeatures.size(); ++i) {
        const auto& feature = OneHotFeatures[i];
        for (int valueId = 0; valueId < feature.Values.ysize(); ++valueId) {
            TOneHotSplit oh{feature.CatFeatureIndex, feature.Values[valueId]};
            ref.BinFeatures.emplace_back(oh);
            auto& bf = splitIds.emplace_back();
            bf.FeatureIdx = ref.EffectiveBinFeaturesBucketCount;
            bf.SplitIdx = valueId + 1;
        }
        ++ref.EffectiveBinFeaturesBucketCount;
    }
    for (size_t i = 0; i < CtrFeatures.size(); ++i) {
        const auto& feature = CtrFeatures[i];
        for (int borderId = 0; borderId < feature.Borders.ysize(); ++borderId) {
            TModelCtrSplit ctrSplit;
            ctrSplit.Ctr = feature.Ctr;
            ctrSplit.Border = feature.Borders[borderId];
            ref.BinFeatures.emplace_back(std::move(ctrSplit));
            auto& bf = splitIds.emplace_back();
            bf.FeatureIdx = ref.EffectiveBinFeaturesBucketCount;
            bf.SplitIdx = borderId + 1;
        }
        ++ref.EffectiveBinFeaturesBucketCount;
    }
    for (const auto& binSplit : TreeSplits) {
        const auto& feature = ref.BinFeatures[binSplit];
        const auto& featureIndex = splitIds[binSplit];
        Y_ENSURE(featureIndex.FeatureIdx <= 0xffff, "To many features in model, ask catboost team for support");
        Y_ENSURE(featureIndex.SplitIdx <= 254, "To many splits in feature, ask catboost team for support");
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

void TFullModel::CalcFlat(const TVector<TConstArrayRef<float>>& features,
                                 size_t treeStart,
                                 size_t treeEnd,
                                 TArrayRef<double> results) const {
    const auto expectedFlatVecSize = ObliviousTrees.GetFlatFeatureVectorExpectedSize();
    for (const auto& flatFeaturesVec : features) {
        CB_ENSURE(flatFeaturesVec.size() >= expectedFlatVecSize,
                  "insufficient flat features vector size: " << flatFeaturesVec.size()
                                                             << " expected: " << expectedFlatVecSize);
    }
    CalcGeneric(
        *this,
        [&](const TFloatFeature& floatFeature, size_t index) {
            return features[index][floatFeature.FlatFeatureIndex];
        },
        [&](size_t catFeatureIdx, size_t index) {
            return ConvertFloatCatFeatureToIntHash(features[index][ObliviousTrees.CatFeatures[catFeatureIdx].FlatFeatureIndex]);
        },
        features.size(),
        treeStart,
        treeEnd,
        results
    );
}

void TFullModel::CalcFlatSingle(const TConstArrayRef<float>& features, size_t treeStart, size_t treeEnd, TArrayRef<double> results) const {
    CalcGeneric(
        *this,
        [&](const TFloatFeature& floatFeature, size_t ) {
            return features[floatFeature.FlatFeatureIndex];
        },
        [&](size_t catFeatureIdx, size_t ) {
            return ConvertFloatCatFeatureToIntHash(features[ObliviousTrees.CatFeatures[catFeatureIdx].FlatFeatureIndex]);
        },
        1,
        treeStart,
        treeEnd,
        results
    );
}

void TFullModel::CalcFlatTransposed(const TVector<TConstArrayRef<float>>& transposedFeatures,
                                           size_t treeStart,
                                           size_t treeEnd,
                                           TArrayRef<double> results) const {
    CB_ENSURE(!transposedFeatures.empty(), "Features should not be empty");
    CalcGeneric(
        *this,
        [&](const TFloatFeature& floatFeature, size_t index) {
            return transposedFeatures[floatFeature.FlatFeatureIndex][index];
        },
        [&](size_t catFeatureIdx, size_t index) {
            return ConvertFloatCatFeatureToIntHash(transposedFeatures[ObliviousTrees.CatFeatures[catFeatureIdx].FlatFeatureIndex][index]);
        },
        transposedFeatures[0].Size(),
        treeStart,
        treeEnd,
        results
    );
}

void TFullModel::Calc(const TVector<TConstArrayRef<float>>& floatFeatures,
                      const TVector<TConstArrayRef<int>>& catFeatures,
                      size_t treeStart,
                      size_t treeEnd,
                      TArrayRef<double> results) const {
    if (!floatFeatures.empty() && !catFeatures.empty()) {
        CB_ENSURE(catFeatures.size() == floatFeatures.size());
    }
    for (const auto& floatFeaturesVec : floatFeatures) {
        CB_ENSURE(floatFeaturesVec.size() >= ObliviousTrees.GetNumFloatFeatures(),
                  "insufficient float features vector size: " << floatFeaturesVec.size()
                                                              << " expected: " << ObliviousTrees.GetNumFloatFeatures());
    }
    for (const auto& catFeaturesVec : catFeatures) {
        CB_ENSURE(catFeaturesVec.size() >= ObliviousTrees.GetNumCatFeatures(),
                  "insufficient cat features vector size: " << catFeaturesVec.size()
                                                            << " expected: " << ObliviousTrees.GetNumCatFeatures());
    }
    CalcGeneric(
        *this,
        [&](const TFloatFeature& floatFeature, size_t index) {
            return floatFeatures[index][floatFeature.FeatureIndex];
        },
        [&](size_t catFeatureIdx, size_t index) {
            return catFeatures[index][catFeatureIdx];
        },
        floatFeatures.size(),
        treeStart,
        treeEnd,
        results
    );
}

void TFullModel::Calc(const TVector<TConstArrayRef<float>>& floatFeatures,
                             const TVector<TVector<TStringBuf>>& catFeatures, size_t treeStart, size_t treeEnd,
                             TArrayRef<double> results) const {
    if (!floatFeatures.empty() && !catFeatures.empty()) {
        CB_ENSURE(catFeatures.size() == floatFeatures.size());
    }
    for (const auto& floatFeaturesVec : floatFeatures) {
        CB_ENSURE(floatFeaturesVec.size() >= ObliviousTrees.GetNumFloatFeatures(),
                  "insufficient float features vector size: " << floatFeaturesVec.size()
                                                              << " expected: " << ObliviousTrees.GetNumFloatFeatures());
    }
    for (const auto& catFeaturesVec : catFeatures) {
        CB_ENSURE(catFeaturesVec.size() >= ObliviousTrees.GetNumCatFeatures(),
                  "insufficient cat features vector size: " << catFeaturesVec.size()
                                                            << " expected: " << ObliviousTrees.GetNumCatFeatures());
    }
    CalcGeneric(
        *this,
        [&](const TFloatFeature& floatFeature, size_t index) {
            return floatFeatures[index][floatFeature.FeatureIndex];
        },
        [&](size_t catFeatureIdx, size_t index) {
            return CalcCatFeatureHash(catFeatures[index][catFeatureIdx]);
        },
        floatFeatures.size(),
        treeStart,
        treeEnd,
        results
    );
}

TVector<TVector<double>> TFullModel::CalcTreeIntervals(
    const TVector<TConstArrayRef<float>>& floatFeatures,
    const TVector<TConstArrayRef<int>>& catFeatures,
    size_t incrementStep) const {
    if (!floatFeatures.empty() && !catFeatures.empty()) {
        CB_ENSURE(catFeatures.size() == floatFeatures.size());
    }
    for (const auto& floatFeaturesVec : floatFeatures) {
        CB_ENSURE(floatFeaturesVec.size() >= ObliviousTrees.GetNumFloatFeatures(),
                  "insufficient float features vector size: " << floatFeaturesVec.size()
                                                              << " expected: " << ObliviousTrees.GetNumFloatFeatures());
    }
    for (const auto& catFeaturesVec : catFeatures) {
        CB_ENSURE(catFeaturesVec.size() >= ObliviousTrees.GetNumCatFeatures(),
                  "insufficient cat features vector size: " << catFeaturesVec.size()
                                                            << " expected: " << ObliviousTrees.GetNumCatFeatures());
    }
    return CalcTreeIntervalsGeneric(
        *this,
        [&](const TFloatFeature& floatFeature, size_t index) {
            return floatFeatures[index][floatFeature.FeatureIndex];
        },
        [&](size_t catFeatureIdx, size_t index) {
            return catFeatures[index][catFeatureIdx];
        },
        floatFeatures.size(),
        incrementStep
    );
}
TVector<TVector<double>> TFullModel::CalcTreeIntervalsFlat(
    const TVector<TConstArrayRef<float>>& features,
    size_t incrementStep) const {
    const auto expectedFlatVecSize = ObliviousTrees.GetFlatFeatureVectorExpectedSize();
    for (const auto& flatFeaturesVec : features) {
        CB_ENSURE(flatFeaturesVec.size() >= expectedFlatVecSize,
                  "insufficient flat features vector size: " << flatFeaturesVec.size()
                                                             << " expected: " << expectedFlatVecSize);
    }
    return CalcTreeIntervalsGeneric(
        *this,
        [&](const TFloatFeature& floatFeature, size_t index) {
            return features[index][floatFeature.FlatFeatureIndex];
        },
        [&](size_t catFeatureIdx, size_t index) {
            return ConvertFloatCatFeatureToIntHash(features[index][ObliviousTrees.CatFeatures[catFeatureIdx].FlatFeatureIndex]);
        },
        features.size(),
        incrementStep
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
                key_value.first.size()),
            serializer.FlatbufBuilder.CreateString(
                key_value.second.c_str(),
                key_value.second.size())
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
