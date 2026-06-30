#include "json_model_helpers.h"

#include <catboost/libs/helpers/json_helpers.h>
#include <catboost/libs/model/ctr_helpers.h>
#include <catboost/libs/model/static_ctr_provider.h>

#include <catboost/libs/model/flatbuffers/model.fbs.h>

#include <library/cpp/json/json_reader.h>
#include <library/cpp/json/json_writer.h>

#include <util/generic/set.h>
#include <util/string/builder.h>
#include <util/string/cast.h>
#include <util/stream/file.h>
#include <util/stream/str.h>
#include <util/system/yassert.h>


using namespace NJson;


static TJsonValue ToJson(const TFloatSplit& floatSplit) {
    TJsonValue jsonValue;
    jsonValue.InsertValue("float_feature_index", floatSplit.FloatFeature);
    jsonValue.InsertValue("border", floatSplit.Split);
    return jsonValue;
}

static TJsonValue ToJson(const TOneHotSplit& oneHotSplit) {
    TJsonValue jsonValue;
    jsonValue.InsertValue("value", oneHotSplit.Value);
    jsonValue.InsertValue("cat_feature_index", oneHotSplit.CatFeatureIdx);
    return jsonValue;
}

static TJsonValue ToJson(const TFeatureCombination& featureCombination) {
    TJsonValue jsonValue;
    for (const auto& feature: featureCombination.CatFeatures) {
        jsonValue.AppendValue(TJsonValue());
        jsonValue.Back().InsertValue("cat_feature_index", feature);
        jsonValue.Back().InsertValue("combination_element", "cat_feature_value");
    }
    for (const auto& feature: featureCombination.BinFeatures) {
        jsonValue.AppendValue(ToJson(feature));
        jsonValue.Back().InsertValue("combination_element", "float_feature");
    }
    for (const auto& feature: featureCombination.OneHotFeatures) {
        jsonValue.AppendValue(ToJson(feature));
        jsonValue.Back().InsertValue("combination_element", "cat_feature_exact_value");
    }
    return jsonValue;
}

static TFloatSplit FloatSplitFromJson(const TJsonValue& value) {
    return TFloatSplit(
            value["float_feature_index"].GetInteger(), FromJson<float>(value["border"])
    );
}

static TOneHotSplit OneHotSplitFromJson(const TJsonValue& value) {
    return TOneHotSplit(value["cat_feature_index"].GetInteger(), value["value"].GetInteger());
}

static TFeatureCombination FeatureCombinationFromJson(const TJsonValue& jsonValue) {
    TFeatureCombination featureCombination;
    for (const auto& value: jsonValue.GetArray()) {
        auto& combinationElement = value["combination_element"];
        if (combinationElement == "cat_feature_value") {
            featureCombination.CatFeatures.push_back(value["cat_feature_index"].GetInteger());
        } else if (combinationElement == "float_feature") {
            featureCombination.BinFeatures.push_back(FloatSplitFromJson(value));
        } else {
            featureCombination.OneHotFeatures.push_back(OneHotSplitFromJson(value));
        }
    }
    return featureCombination;
}

static TJsonValue ToJson(const TModelCtrBase& modelCtrBase) {
    TJsonValue jsonValue;
    jsonValue.InsertValue("type", ToString<ECtrType>(modelCtrBase.CtrType));
    jsonValue.InsertValue("identifier", ToJson(modelCtrBase.Projection));
    return jsonValue;
}

TString ModelCtrBaseToStr(const TModelCtrBase& modelCtrBase) {
    return WriteJsonWithCatBoostPrecision(ToJson(modelCtrBase), false);
}

static TModelCtrBase ModelCtrBaseFromJson(const TJsonValue& jsonValue) {
    TModelCtrBase modelCtrBase;
    modelCtrBase.CtrType = FromString<ECtrType>(jsonValue["type"].GetString());
    modelCtrBase.Projection = FeatureCombinationFromJson(jsonValue["identifier"]);
    return modelCtrBase;
}

static TModelCtrBase ModelCtrBaseFromString(const TString& modelCtrBaseDescription) {
    TStringInput ss(modelCtrBaseDescription);
    TJsonValue tree;
    CB_ENSURE(ReadJsonTree(&ss, &tree), "can't parse params file");
    return ModelCtrBaseFromJson(tree);
}

static TJsonValue ToJson(const TCtrFeature& ctrFeature) {
    TJsonValue jsonValue;
    jsonValue.InsertValue("identifier", ModelCtrBaseToStr(ctrFeature.Ctr.Base));
    jsonValue.InsertValue("elements", ToJson(ctrFeature.Ctr.Base.Projection));
    jsonValue.InsertValue("ctr_type", ToString<ECtrType>(ctrFeature.Ctr.Base.CtrType));
    jsonValue.InsertValue("prior_numerator", ctrFeature.Ctr.PriorNum);
    jsonValue.InsertValue("prior_denomerator", ctrFeature.Ctr.PriorDenom);
    jsonValue.InsertValue("shift", ctrFeature.Ctr.Shift);
    jsonValue.InsertValue("scale", ctrFeature.Ctr.Scale);
    jsonValue.InsertValue("target_border_idx", ctrFeature.Ctr.TargetBorderIdx);
    jsonValue.InsertValue("borders", VectorToJson(ctrFeature.Borders));
    return jsonValue;
}

static TModelCtr ModelCtrFromJson(const TJsonValue& value) {
    TModelCtr modelCtr;
    modelCtr.Base.Projection = FeatureCombinationFromJson(value["elements"]);
    modelCtr.Base.CtrType = FromString<ECtrType>(value["ctr_type"].GetString());
    modelCtr.TargetBorderIdx = FromJson<int>(value["target_border_idx"]);
    modelCtr.PriorNum = FromJson<float>(value["prior_numerator"]);
    modelCtr.PriorDenom = FromJson<float>(value["prior_denomerator"]);
    modelCtr.Shift = FromJson<float>(value["shift"]);
    modelCtr.Scale = FromJson<float>(value["scale"]);
    return modelCtr;
}

static TCtrFeature CtrFeatureFromJson(const TJsonValue& value) {
    TCtrFeature ctrFeature;
    ctrFeature.Borders = JsonToVector<float>(value["borders"]);
    ctrFeature.Ctr = ModelCtrFromJson(value);
    return ctrFeature;
}

static TJsonValue ToJson(const TModelCtrSplit& modelCtrSplit) {
    TJsonValue jsonValue;
    jsonValue.InsertValue("ctr_target_border_idx", modelCtrSplit.Ctr.TargetBorderIdx);
    jsonValue.InsertValue("border", modelCtrSplit.Border);
    return jsonValue;
}

static TJsonValue ToJson(const TModelSplit& modelSplit) {
    TJsonValue jsonValue;
    if (modelSplit.Type == ESplitType::FloatFeature) {
        jsonValue = ToJson(modelSplit.FloatFeature);
    } else if (modelSplit.Type == ESplitType::OnlineCtr) {
        jsonValue = ToJson(modelSplit.OnlineCtr);
    } else {
        Y_ASSERT(modelSplit.Type == ESplitType::OneHotFeature);
        jsonValue = ToJson(modelSplit.OneHotFeature);
    }
    jsonValue.InsertValue("split_type", ToString<ESplitType>(modelSplit.Type));
    return jsonValue;
}

static TJsonValue ToJson(const TFloatFeature& floatFeature) {
    TJsonValue jsonValue;
    jsonValue.InsertValue("has_nans", floatFeature.HasNans);
    jsonValue.InsertValue("feature_index", floatFeature.Position.Index);
    jsonValue.InsertValue("flat_feature_index", floatFeature.Position.FlatIndex);
    jsonValue.InsertValue("borders", VectorToJson(floatFeature.Borders));
    jsonValue.InsertValue("feature_id", floatFeature.FeatureId);
    switch (floatFeature.NanValueTreatment) {
        case TFloatFeature::ENanValueTreatment::AsIs:
            jsonValue.InsertValue("nan_value_treatment", "AsIs");
            break;
        case TFloatFeature::ENanValueTreatment::AsFalse:
            jsonValue.InsertValue("nan_value_treatment", "AsFalse");
            break;
        case TFloatFeature::ENanValueTreatment::AsTrue:
            jsonValue.InsertValue("nan_value_treatment", "AsTrue");
            break;
    }
    return jsonValue;
}

static TFloatFeature FloatFeatureFromJson(const TJsonValue& value) {
    auto feature = TFloatFeature(
        value["has_nans"].GetBoolean(),
        value["feature_index"].GetInteger(),
        value["flat_feature_index"].GetInteger(),
        JsonToVector<float>(value["borders"]),
        value["feature_id"].GetString());
    feature.NanValueTreatment = FromString<TFloatFeature::ENanValueTreatment>(value["nan_value_treatment"].GetString());
    return feature;
}

static TJsonValue ToJson(const TCatFeature& catFeature) {
    TJsonValue jsonValue;
    jsonValue.InsertValue("feature_index", catFeature.Position.Index);
    jsonValue.InsertValue("flat_feature_index", catFeature.Position.FlatIndex);
    jsonValue.InsertValue("feature_id", catFeature.FeatureId);
    return jsonValue;
}

static TCatFeature CatFeatureFromJson(const TJsonValue& value) {
    TCatFeature catFeature;
    catFeature.Position.Index = value["feature_index"].GetInteger();
    catFeature.Position.FlatIndex = value["flat_feature_index"].GetInteger();
    catFeature.FeatureId = value["feature_id"].GetString();
    return catFeature;
}

static TJsonValue GetFeaturesInfoJson(
        const TModelTrees& modelTrees,
        const TVector<TString>* featureId,
        const THashMap<ui32, TString>* catFeaturesHashToString
) {
    TJsonValue jsonValue;
    if (!modelTrees.GetFloatFeatures().empty()) {
        jsonValue.InsertValue("float_features", TJsonValue());
    }
    for (const auto& floatFeature: modelTrees.GetFloatFeatures()) {
        jsonValue["float_features"].AppendValue(ToJson(floatFeature));
        if (featureId && !featureId->empty()) {
            const auto& name = (*featureId)[floatFeature.Position.FlatIndex];
            if (!name.empty()) {
                jsonValue["float_features"].Back().InsertValue("feature_name", name);
            }
        }
    }
    if (!modelTrees.GetCatFeatures().empty()) {
        jsonValue.InsertValue("categorical_features", TJsonValue());
        THashMap<int, int> oneHotIndexes;
        for (int idx = 0; idx < modelTrees.GetOneHotFeatures().ysize(); ++idx) {
            oneHotIndexes[modelTrees.GetOneHotFeatures()[idx].CatFeatureIndex] = idx;
        }
        for (const auto &catFeature: modelTrees.GetCatFeatures()) {
            auto catFeatureJsonValue = ToJson(catFeature);
            if (oneHotIndexes.contains(catFeature.Position.Index)) {
                auto &ohFeauture = modelTrees.GetOneHotFeatures()[oneHotIndexes[catFeature.Position.Index]];
                catFeatureJsonValue.InsertValue("values", VectorToJson(ohFeauture.Values));
                if (!ohFeauture.StringValues.empty()) {
                    catFeatureJsonValue.InsertValue("string_values", VectorToJson(ohFeauture.StringValues));
                }
            }
            if (featureId && !featureId->empty()) {
                const auto &name = (*featureId)[catFeature.Position.FlatIndex];
                if (!name.empty()) {
                    catFeatureJsonValue.InsertValue("feature_name", name);
                }
            }
            jsonValue["categorical_features"].AppendValue(catFeatureJsonValue);
        }
    }
    if (!modelTrees.GetCtrFeatures().empty()) {
        jsonValue.InsertValue("ctrs", TJsonValue());
    }
    for (const auto& ctr: modelTrees.GetCtrFeatures()) {
        jsonValue["ctrs"].AppendValue(ToJson(ctr));
    }
    if (catFeaturesHashToString) {
        jsonValue.InsertValue("cat_features_hash", TJsonValue());
        for (const auto& key_value: *catFeaturesHashToString) {
            TJsonValue catFeaturesHash;
            catFeaturesHash.InsertValue("hash", key_value.first);
            catFeaturesHash.InsertValue("value", key_value.second);
            jsonValue["cat_features_hash"].AppendValue(catFeaturesHash);
        }
    }
    return jsonValue;
}

static void GetFeaturesInfo(const TJsonValue& jsonValue, TModelTrees* modelTrees) {
    if (jsonValue.Has("float_features")) {
        for (const auto &value: jsonValue["float_features"].GetArray()) {
            modelTrees->AddFloatFeature(FloatFeatureFromJson(value));
        }
    }
    if (jsonValue.Has("categorical_features")) {
        for (const auto &value: jsonValue["categorical_features"].GetArray()) {
            auto catFeature = CatFeatureFromJson(value);
            modelTrees->AddCatFeature(catFeature);
            if (value.Has("values")) {
                TOneHotFeature ohFeature;
                ohFeature.CatFeatureIndex = catFeature.Position.Index;
                ohFeature.Values = JsonToVector<int>(value["values"]);
                ohFeature.StringValues = JsonToVector<TString>(value["string_values"]);
                modelTrees->AddOneHotFeature(ohFeature);
            }
        }
    }
    if (jsonValue.Has("ctrs")) {
        for (const auto &value: jsonValue["ctrs"].GetArray()) {
            modelTrees->AddCtrFeature(CtrFeatureFromJson(value));
        }
    }
}

static TJsonValue GetObliviousModelTreesJson(const TModelTrees& modelTrees) {
    int leafValuesOffset = 0;
    int leafWeightsOffset = 0;
    TJsonValue jsonValue;
    const auto& binFeatures = modelTrees.GetBinFeatures();
    for (int treeIdx = 0; treeIdx < modelTrees.GetModelTreeData()->GetTreeSizes().ysize(); ++treeIdx) {
        TJsonValue tree;
        const size_t treeLeafCount = (1uLL << modelTrees.GetModelTreeData()->GetTreeSizes()[treeIdx]) * modelTrees.GetDimensionsCount();
        const size_t treeWeightsCount = (1uLL << modelTrees.GetModelTreeData()->GetTreeSizes()[treeIdx]);
        if (!modelTrees.GetModelTreeData()->GetLeafWeights().empty()) {
            for (size_t idx = 0; idx < treeWeightsCount; ++idx) {
                tree["leaf_weights"].AppendValue(modelTrees.GetModelTreeData()->GetLeafWeights()[leafWeightsOffset + idx]);
            }
        }
        tree.InsertValue("leaf_values", TJsonValue());
        for (size_t idx = 0; idx < treeLeafCount; ++idx) {
            tree["leaf_values"].AppendValue(modelTrees.GetModelTreeData()->GetLeafValues()[leafValuesOffset + idx]);
        }
        leafValuesOffset += treeLeafCount;
        leafWeightsOffset += treeWeightsCount;
        int treeSplitEnd;
        if (treeIdx + 1 < modelTrees.GetModelTreeData()->GetTreeStartOffsets().ysize()) {
            treeSplitEnd = modelTrees.GetModelTreeData()->GetTreeStartOffsets()[treeIdx + 1];
        } else {
            treeSplitEnd = modelTrees.GetModelTreeData()->GetTreeSplits().ysize();
        }
        tree.InsertValue("splits", TJsonValue());
        for (int idx = modelTrees.GetModelTreeData()->GetTreeStartOffsets()[treeIdx]; idx < treeSplitEnd; ++idx) {
            tree["splits"].AppendValue(ToJson(binFeatures[modelTrees.GetModelTreeData()->GetTreeSplits()[idx]]));
            tree["splits"].Back().InsertValue("split_index", modelTrees.GetModelTreeData()->GetTreeSplits()[idx]);
        }
        jsonValue.AppendValue(tree);
    }
    return jsonValue;
}

static TJsonValue BuildLeafJson(const TModelTrees& modelTrees, ui32 nodeIdx) {
    ui32 leafIdx = modelTrees.GetModelTreeData()->GetNonSymmetricNodeIdToLeafId()[nodeIdx];
    TJsonValue leafJson;
    leafJson.InsertValue("weight", modelTrees.GetModelTreeData()->GetLeafWeights()[leafIdx / modelTrees.GetDimensionsCount()]);
    if (modelTrees.GetDimensionsCount() == 1) {
        leafJson.InsertValue("value", modelTrees.GetModelTreeData()->GetLeafValues()[leafIdx]);
    } else {
        TConstArrayRef<double> valueRef(modelTrees.GetModelTreeData()->GetLeafValues().begin() + leafIdx, modelTrees.GetDimensionsCount());
        leafJson.InsertValue("value", VectorToJson<double>({valueRef.begin(), valueRef.end()}));
    }
    return leafJson;
}

static TJsonValue BuildTreeJson(const TModelTrees& modelTrees, ui32 nodeIdx) {
    TJsonValue tree;
    const TNonSymmetricTreeStepNode& node = modelTrees.GetModelTreeData()->GetNonSymmetricStepNodes()[nodeIdx];
    if (node.LeftSubtreeDiff == 0 && node.RightSubtreeDiff == 0) {
        return BuildLeafJson(modelTrees, nodeIdx);
    } else {
        tree.InsertValue("split", ToJson(modelTrees.GetBinFeatures()[modelTrees.GetModelTreeData()->GetTreeSplits()[nodeIdx]]));
        tree["split"].InsertValue("split_index", modelTrees.GetModelTreeData()->GetTreeSplits()[nodeIdx]);
        tree.InsertValue("left",
            node.LeftSubtreeDiff ? BuildTreeJson(modelTrees, nodeIdx + node.LeftSubtreeDiff) : BuildLeafJson(modelTrees, nodeIdx));
        tree.InsertValue("right",
            node.RightSubtreeDiff ? BuildTreeJson(modelTrees, nodeIdx + node.RightSubtreeDiff) : BuildLeafJson(modelTrees, nodeIdx));
    }
    return tree;
}

static TJsonValue GetNonSymmetricModelTreesJson(const TModelTrees& modelTrees) {
    TJsonValue jsonValue(JSON_ARRAY);
    for (int treeIdx = 0; treeIdx < modelTrees.GetModelTreeData()->GetTreeSizes().ysize(); ++treeIdx) {
        jsonValue.AppendValue(BuildTreeJson(modelTrees, modelTrees.GetModelTreeData()->GetTreeStartOffsets()[treeIdx]));
    }
    return jsonValue;
}

static TJsonValue GetModelTreesJson(const TModelTrees& modelTrees) {
    if (modelTrees.IsOblivious()) {
        return GetObliviousModelTreesJson(modelTrees);
    } else {
        return GetNonSymmetricModelTreesJson(modelTrees);
    }
}

static void GetObliviousModelTrees(const TJsonValue& jsonValue, TModelTrees* modelTrees) {
    for (const auto& value: jsonValue.GetArray()) {
        for (const auto& leaf: value["leaf_values"].GetArray()) {
            modelTrees->AddLeafValue(leaf.GetDouble());
        }
        int treeSize = value["splits"].GetArray().ysize();
        modelTrees->AddTreeSize(treeSize);
        modelTrees->SetApproxDimension(value["leaf_values"].GetArray().ysize() / (1uLL << treeSize));
        for (const auto& split: value["splits"].GetArray()) {
            modelTrees->AddTreeSplit(split["split_index"].GetInteger());
        }
        if (value.Has("leaf_weights")) {
            for (const auto& weight: value["leaf_weights"].GetArray()) {
                modelTrees->AddLeafWeight(weight.GetDouble());
            }
        }
    }
}

static void GetNonSymmetricModelTrees(const TJsonValue& jsonValue, TModelTrees* modelTrees) {
    TVector<TNonSymmetricTreeStepNode> nodes;
    TVector<ui32> nodeIdToLeafId;
    const std::function<int(const TJsonValue&)> readTreeFromJson = [modelTrees, &nodes, &nodeIdToLeafId, &readTreeFromJson](const TJsonValue& jsonNode) {
        int nodeIdx = nodes.size();
        nodes.emplace_back(TNonSymmetricTreeStepNode{0, 0});
        if (jsonNode.Has("value")) {
            const TJsonValue& value = jsonNode["value"];
            nodeIdToLeafId.push_back(modelTrees->GetModelTreeData()->GetLeafValues().size());
            modelTrees->AddTreeSplit(0);
            if (value.GetType() == EJsonValueType::JSON_ARRAY) {
                modelTrees->SetApproxDimension(value.GetArray().ysize());
                for (const auto& singleValue : value.GetArray()) {
                    modelTrees->AddLeafValue(singleValue.GetDouble());
                }
            } else {
                modelTrees->AddLeafValue(value.GetDouble());
            }
            if (jsonNode.Has("weight")) {
                modelTrees->AddLeafWeight(jsonNode["weight"].GetDouble());
            }
        } else {
            nodeIdToLeafId.push_back(Max<ui32>());
            modelTrees->AddTreeSplit(jsonNode["split"]["split_index"].GetInteger());
            nodes[nodeIdx].LeftSubtreeDiff = readTreeFromJson(jsonNode["left"]) - nodeIdx;
            nodes[nodeIdx].RightSubtreeDiff = readTreeFromJson(jsonNode["right"]) - nodeIdx;
        }
        return nodeIdx;
    };
    for (const auto& treeJson : jsonValue.GetArray()) {
        int oldNodesCount = nodes.size();
        readTreeFromJson(treeJson);
        modelTrees->AddTreeSize(nodes.size() - oldNodesCount);
    }
    modelTrees->SetNonSymmetricStepNodes(std::move(nodes));
    modelTrees->SetNonSymmetricNodeIdToLeafId(std::move(nodeIdToLeafId));
}

static NJson::TJsonValue ConvertCtrsToJson(const TStaticCtrProvider* ctrProvider, const TConstArrayRef<TModelCtr> neededCtrs) {
    NJson::TJsonValue jsonValue;
    if (neededCtrs.empty()) {
        return jsonValue;
    }
    auto compressedModelCtrs = NCB::CompressModelCtrs(neededCtrs);
    for (size_t idx = 0; idx < compressedModelCtrs.size(); ++idx) {
        auto& proj = *compressedModelCtrs[idx].Projection;
        for (const auto& ctr: compressedModelCtrs[idx].ModelCtrs) {
            NJson::TJsonValue hashValue;
            auto& learnCtr = ctrProvider->CtrData.LearnCtrs.at(ctr->Base);
            auto hashIndexResolver = learnCtr.GetIndexHashViewer();
            const ECtrType ctrType = ctr->Base.CtrType;
            THashSet<ui64> hashIndexes;
            for (const auto& bucket: hashIndexResolver.GetBuckets()) {
                auto value = bucket.IndexValue;
                if (value == NCatboost::TDenseIndexHashView::NotFoundIndex) {
                    continue;
                }
                // make a copy because we can't pass a reference to an unaligned struct member to 'hashIndexes' methods
                auto bucketHash = bucket.Hash;
                if (!hashIndexes.insert(bucketHash).second) {
                    continue;
                }
                hashValue.AppendValue(ToString(bucketHash));
                if (ctrType == ECtrType::BinarizedTargetMeanValue || ctrType == ECtrType::FloatTargetMeanValue) {
                    if (value != NCatboost::TDenseIndexHashView::NotFoundIndex) {
                        auto ctrMean = learnCtr.GetTypedArrayRefForBlobData<TCtrMeanHistory>();
                        const TCtrMeanHistory& ctrMeanHistory = ctrMean[value];
                        hashValue.AppendValue(ctrMeanHistory.Sum);
                        hashValue.AppendValue(ctrMeanHistory.Count);
                    }
                } else  if (ctrType == ECtrType::Counter || ctrType == ECtrType::FeatureFreq) {
                    TConstArrayRef<int> ctrTotal = learnCtr.GetTypedArrayRefForBlobData<int>();
                    hashValue.AppendValue(ctrTotal[value]);
                } else {
                    auto ctrIntArray = learnCtr.GetTypedArrayRefForBlobData<int>();
                    const int targetClassesCount = learnCtr.TargetClassesCount;
                    auto ctrHistory = MakeArrayRef(ctrIntArray.data() + value * targetClassesCount, targetClassesCount);
                    for (int classId = 0; classId < targetClassesCount; ++classId) {
                        hashValue.AppendValue(ctrHistory[classId]);
                    }
                }
            }
            NJson::TJsonValue hash;
            hash["hash_map"] = hashValue;
            hash["hash_stride"] =  hashValue.GetArray().ysize() / hashIndexes.size();
            hash["counter_denominator"] = learnCtr.CounterDenominator;
            TModelCtrBase modelCtrBase;
            modelCtrBase.Projection = proj;
            modelCtrBase.CtrType = ctrType;
            jsonValue.InsertValue(ModelCtrBaseToStr(modelCtrBase), hash);
        }
    }
    return jsonValue;
}

static TJsonValue GetScaleAndBiasJson(const TFullModel& model) {
    TJsonValue jsonValue;
    jsonValue.AppendValue(model.GetScaleAndBias().Scale);
    jsonValue.AppendValue(TJsonValue());
    auto bias = model.GetScaleAndBias().GetBiasRef();
    for (auto b : bias) {
        jsonValue[1].AppendValue(b);
    }
    return jsonValue;
}

TJsonValue ConvertModelToJson(const TFullModel& model, const TVector<TString>* featureId, const THashMap<ui32, TString>* catFeaturesHashToString) {
    TJsonValue jsonModel;
    TJsonValue modelInfo;
    for (const auto& key_value : model.ModelInfo) {
        if (key_value.first.EndsWith("params")) {
            TJsonValue tree;
            if (!key_value.second.empty()) {
                TStringStream ss(key_value.second);
                CB_ENSURE(ReadJsonTree(&ss, &tree), "can't parse params file");
                modelInfo.InsertValue(key_value.first, tree);
            }
        } else {
            modelInfo.InsertValue(key_value.first, key_value.second);
        }
    }
    jsonModel.InsertValue("model_info", modelInfo);
    if (model.IsOblivious()) {
        jsonModel.InsertValue("oblivious_trees", GetModelTreesJson(*model.ModelTrees));
    } else {
        jsonModel.InsertValue("trees", GetModelTreesJson(*model.ModelTrees));
    }
    jsonModel.InsertValue("features_info", GetFeaturesInfoJson(*model.ModelTrees, featureId, catFeaturesHashToString));
    const TStaticCtrProvider* ctrProvider = dynamic_cast<TStaticCtrProvider*>(model.CtrProvider.Get());
    if (ctrProvider) {
        auto applyData = model.ModelTrees->GetApplyData();
        jsonModel.InsertValue("ctr_data", ConvertCtrsToJson(ctrProvider, applyData->UsedModelCtrs));
    }
    jsonModel.InsertValue("scale_and_bias", GetScaleAndBiasJson(model));
    return jsonModel;
}

static TCtrData CtrDataFromJson(const TJsonValue& jsonValue) {
    TCtrData ctrData;
    for (const auto& key: jsonValue.GetMap()) {
        TModelCtrBase ctrBase = ModelCtrBaseFromString(key.first);
        TCtrValueTable learnCtr;
        learnCtr.ModelCtrBase = ctrBase;
        auto& ctrType = ctrBase.CtrType;
        const auto& hashJson = key.second;
        int hashStride = hashJson["hash_stride"].GetInteger();
        const auto& hashMap = hashJson["hash_map"].GetArray();
        auto blobSize = hashMap.ysize() / hashStride;
        auto indexHashBuilder = learnCtr.GetIndexHashBuilder(blobSize);

        size_t targetClassesCount = hashStride - 1;

        TArrayRef<int> ctrIntArray;
        TArrayRef<TCtrMeanHistory> ctrMean;
        if (ctrType == ECtrType::BinarizedTargetMeanValue || ctrType == ECtrType::FloatTargetMeanValue) {
            ctrMean = learnCtr.AllocateBlobAndGetArrayRef<TCtrMeanHistory>(blobSize);
        } else if (ctrType == ECtrType::Counter || ctrType == ECtrType::FeatureFreq) {
            ctrIntArray = learnCtr.AllocateBlobAndGetArrayRef<int>(blobSize);
            learnCtr.CounterDenominator = hashJson["counter_denominator"].GetInteger();
        } else {
            ctrIntArray = learnCtr.AllocateBlobAndGetArrayRef<int>(blobSize * targetClassesCount);
            learnCtr.TargetClassesCount = targetClassesCount;
        }

        for(auto hashPtr = hashMap.begin(); hashPtr != hashMap.end();) {
            ui64 hashValue = FromString<ui64>(hashPtr->GetString());
            hashPtr++;
            auto index = indexHashBuilder.AddIndex(hashValue);

            if (ctrType == ECtrType::BinarizedTargetMeanValue || ctrType == ECtrType::FloatTargetMeanValue) {
                ctrMean[index].Sum = hashPtr->GetInteger();
                hashPtr++;
                ctrMean[index].Count = hashPtr->GetInteger();
                hashPtr++;
            } else if (ctrType == ECtrType::Counter || ctrType == ECtrType::FeatureFreq) {
                ctrIntArray[index] = hashPtr->GetInteger();
                hashPtr++;
            } else {
                for (size_t idx = index * targetClassesCount; idx < (index + 1) * targetClassesCount; ++idx) {
                    ctrIntArray[idx] = hashPtr->GetInteger();
                    hashPtr++;
                }
            }
        }

        ctrData.LearnCtrs[ctrBase] = learnCtr;
    }
    return ctrData;
}

void ConvertJsonToCatboostModel(const TJsonValue& jsonModel, TFullModel* fullModel) {
    for (const auto& key_value : jsonModel["model_info"].GetMap()) {
        fullModel->ModelInfo[key_value.first] = key_value.second.GetStringRobust();
    }
    if (jsonModel.Has("oblivious_trees")) {
        GetObliviousModelTrees(jsonModel["oblivious_trees"], fullModel->ModelTrees.GetMutable());
    } else {
        GetNonSymmetricModelTrees(jsonModel["trees"], fullModel->ModelTrees.GetMutable());
    }
    GetFeaturesInfo(jsonModel["features_info"], fullModel->ModelTrees.GetMutable());
    if (jsonModel.Has("ctr_data")) {
        auto ctrData = CtrDataFromJson(jsonModel["ctr_data"]);
        fullModel->CtrProvider = new TStaticCtrProvider(ctrData);
    }
    if (jsonModel.Has("scale_and_bias")) {
        const auto& scaleAndBias = jsonModel["scale_and_bias"].GetArray();
        double scale = scaleAndBias[0].GetDouble();
        TVector<double> bias;
        for (const auto& biasValue : scaleAndBias[1].GetArray()) {
            bias.push_back(biasValue.GetDouble());
        }
        fullModel->SetScaleAndBias({scale, bias});
    }

    fullModel->UpdateDynamicData();
}

void OutputModelJson(const TFullModel& model, const TString& outputPath, const TVector<TString>* featureId, const THashMap<ui32, TString>* catFeaturesHashToString) {
    TOFStream out(outputPath);
    auto jsonModel = ConvertModelToJson(model, featureId, catFeaturesHashToString);
    WriteJsonWithCatBoostPrecision(jsonModel, true, &out);
}
