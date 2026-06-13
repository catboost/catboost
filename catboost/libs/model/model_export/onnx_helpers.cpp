#include "onnx_helpers.h"

#include <catboost/libs/model/model_build_helper.h>
#include <catboost/libs/cat_feature/cat_feature.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/private/libs/labels/helpers.h>
#include <catboost/private/libs/options/enum_helpers.h>
#include <catboost/private/libs/options/json_helper.h>
#include <catboost/private/libs/options/loss_description.h>
#include <catboost/private/libs/options/class_label_options.h>

#include <library/cpp/svnversion/svnversion.h>

#include <contrib/libs/onnx/onnx/common/constants.h>
#include <google/protobuf/repeated_field.h>

#include <util/generic/array_ref.h>
#include <util/generic/hash_set.h>
#include <util/generic/mapfindptr.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/generic/xrange.h>
#include <util/string/cast.h>
#include <util/string/join.h>
#include <util/system/compiler.h>
#include <util/system/yassert.h>

#include <numeric>


using ENanValueTreatment = TFloatFeature::ENanValueTreatment;
using EType = NCB::NOnnx::TOnnxNode::EType;
using TModeNode = NCB::NOnnx::TOnnxNode::TTreeNodeBehaviorToOnnxTreeNodeMode;

const TString TModeNode::BRANCH_LEQ = "BRANCH_LEQ";
const TString TModeNode::BRANCH_LT = "BRANCH_LT";
const TString TModeNode::BRANCH_GTE = "BRANCH_GTE";
const TString TModeNode::BRANCH_GT = "BRANCH_GT";
const TString TModeNode::BRANCH_EQ = "BRANCH_EQ";
const TString TModeNode::BRANCH_NEQ = "BRANCH_NEQ";
const TString TModeNode::LEAF = "LEAF";

void NCB::NOnnx::InitMetadata(
    const TFullModel& model,
    const NJson::TJsonValue& userParameters,
    onnx::ModelProto* onnxModel) {

    // Versions 4 and above essentially just introduce new types we don't need.
    onnxModel->set_ir_version(3);

    onnx::OperatorSetIdProto* opset = onnxModel->add_opset_import();
    opset->set_domain(onnx::AI_ONNX_ML_DOMAIN);
    opset->set_version(2);

    onnxModel->set_producer_name("CatBoost");
    onnxModel->set_producer_version(PROGRAM_VERSION);

    if (userParameters.Has("onnx_domain")) {
        onnxModel->set_domain(userParameters["onnx_domain"].GetStringSafe());
    }
    if (userParameters.Has("onnx_model_version")) {
        onnxModel->set_model_version(userParameters["onnx_model_version"].GetIntegerSafe());
    }
    if (userParameters.Has("onnx_doc_string")) {
        onnxModel->set_doc_string(userParameters["onnx_doc_string"].GetStringSafe());
    }

    for (const auto& [key, value] : model.ModelInfo) {
        CB_ENSURE_INTERNAL(
            key != "cat_features",
            "model metadata contains 'cat_features' key, but it is reserved for categorical features indices"
        );

        onnx::StringStringEntryProto* metadata_prop = onnxModel->add_metadata_props();
        metadata_prop->set_key(key);
        metadata_prop->set_value(value);
    }

    // If categorical features are present save cat_features to metadata_props as well
    if (!model.ModelTrees->GetCatFeatures().empty()) {
        TVector<int> catFeaturesIndices;
        for (const auto& catFeature : model.ModelTrees->GetCatFeatures()) {
            catFeaturesIndices.push_back(catFeature.Position.FlatIndex);
        }

        onnx::StringStringEntryProto* catFeaturesProp = onnxModel->add_metadata_props();
        catFeaturesProp->set_key("cat_features");
        catFeaturesProp->set_value(JoinSeq(",", catFeaturesIndices));
    }
}


static bool IsClassifierModel(const TFullModel& model) {
    if (model.ModelTrees->GetDimensionsCount() > 1) { // multiclass
        return true;
    }

    if (const auto* modelInfoParams = MapFindPtr(model.ModelInfo, "params")) {
        NJson::TJsonValue paramsJson = ReadTJsonValue(*modelInfoParams);

        if (paramsJson.Has("loss_function")) {
            NCatboostOptions::TLossDescription modelLossDescription;
            modelLossDescription.Load(paramsJson["loss_function"]);

            if (IsClassificationObjective(modelLossDescription.LossFunction)) {
                return true;
            }
        }
    }

    return false;
}


/*
 * TLabelContainer is either TVector<TJsonValue> or TJsonArray
 * call with non-empty classLabels only
 * only one of classLabelsInt64 or classLabelsString returned nonempty
 */
template <class TLabelContainer>
static void GetClassLabelsImpl(
    const TLabelContainer& classLabels,
    TVector<i64>* classLabelsInt64,
    TVector<TString>* classLabelsString) {

    CB_ENSURE(!classLabels.empty(), "Class labels are missing");

    classLabelsInt64->clear();
    classLabelsString->clear();

    switch (classLabels.begin()->GetType()) {
        case NJson::JSON_INTEGER:
            classLabelsInt64->reserve(classLabels.size());
            for (const NJson::TJsonValue& classLabel : classLabels) {
                classLabelsInt64->push_back(classLabel.GetInteger());
            }
            break;
        case NJson::JSON_DOUBLE:
            CB_ENSURE(false, "ONNX format does not support floating-point labels");
        case NJson::JSON_STRING:
            classLabelsString->reserve(classLabels.size());
            for (const NJson::TJsonValue& classLabel : classLabels) {
                classLabelsString->push_back(NCB::ClassLabelToString(classLabel));
            }
            break;
        default:
            CB_ENSURE(false, "Unexpected label type");
    }
}

// only one of classLabelsInt64 or classLabelsString returned nonempty
static void GetClassLabels(
    const TFullModel& model,
    TVector<i64>* classLabelsInt64,
    TVector<TString>* classLabelsString) {

    classLabelsInt64->clear();
    classLabelsString->clear();

    const TVector<NJson::TJsonValue> classLabels = model.GetModelClassLabels();
    if (!classLabels.empty()) {
        GetClassLabelsImpl(classLabels, classLabelsInt64, classLabelsString);
    }
}


static void InitValueInfo(
    const TString& name,
    google::protobuf::int32 elemType,
    TMaybe<google::protobuf::int64> secondDim,
    onnx::ValueInfoProto* valueInfo) {

    valueInfo->set_name(name);

    onnx::TypeProto* featuresType = valueInfo->mutable_type();
    onnx::TypeProto_Tensor* tensorType = featuresType->mutable_tensor_type();
    tensorType->set_elem_type(elemType);
    onnx::TensorShapeProto* tensorShape = tensorType->mutable_shape();
    tensorShape->add_dim()->set_dim_param("N");
    if (secondDim) {
        tensorShape->add_dim()->set_dim_value(*secondDim);
    }
}


inline void SetAttributeValue(float f, onnx::AttributeProto* attribute) {
    attribute->set_type(onnx::AttributeProto_AttributeType_FLOAT);
    attribute->set_f(f);
}

inline void SetAttributeValue(google::protobuf::int64 i, onnx::AttributeProto* attribute) {
    attribute->set_type(onnx::AttributeProto_AttributeType_INT);
    attribute->set_i(i);
}

inline void SetAttributeValue(const TString& s, onnx::AttributeProto* attribute) {
    attribute->set_type(onnx::AttributeProto_AttributeType_STRING);
    attribute->set_s(s);
}

inline void SetAttributeValue(TConstArrayRef<google::protobuf::int64> ints, onnx::AttributeProto* attribute) {
    attribute->set_type(onnx::AttributeProto_AttributeType_INTS);
    for (auto i : ints) {
        attribute->add_ints(i);
    }
}

inline void SetAttributeValue(TConstArrayRef<TString> strings, onnx::AttributeProto* attribute) {
    attribute->set_type(onnx::AttributeProto_AttributeType_STRINGS);
    for (const auto& s : strings) {
        attribute->add_strings(s);
    }
}

inline void SetAttributeValue(TConstArrayRef<float> floats, onnx::AttributeProto* attribute) {
    attribute->set_type(onnx::AttributeProto_AttributeType_FLOATS);
    for (auto f : floats) {
        attribute->add_floats(f);
    }
}


template <class T>
static void AddAttribute(
    const TString& name,
    const T& value,
    onnx::NodeProto* node) {

    onnx::AttributeProto* attribute = node->add_attribute();
    attribute->set_name(name);
    SetAttributeValue(value, attribute);
}


static void AddClassLabelsAttribute(
    const TVector<i64>& classLabelsInt64,
    const TVector<TString>& classLabelsString,
    onnx::NodeProto* node) {

    if (!classLabelsInt64.empty()) {
        AddAttribute("classlabels_int64s", classLabelsInt64, node);
    } else {
        AddAttribute("classlabels_strings", classLabelsString, node);
    }
}

static void InitProbabilitiesOutput(
    const TString& name,
    google::protobuf::int32 mapKeysType,
    onnx::ValueInfoProto* output) {

    output->set_name(name);

    onnx::TypeProto* featuresType = output->mutable_type();
    onnx::TypeProto_Sequence* sequenceType = featuresType->mutable_sequence_type();
    onnx::TypeProto* sequenceElementType = sequenceType->mutable_elem_type();

    onnx::TypeProto_Map* mapElement = sequenceElementType->mutable_map_type();
    mapElement->set_key_type(mapKeysType);
    mapElement->mutable_value_type()->mutable_tensor_type()->set_elem_type(onnx::TensorProto_DataType_FLOAT);
}


struct TTreesAttributes {
    // TreeEnsembleClassifier only
    onnx::AttributeProto* class_ids;
    onnx::AttributeProto* class_nodeids;
    onnx::AttributeProto* class_treeids;
    onnx::AttributeProto* class_weights;

    // TreeEnsembleRegressor only
    onnx::AttributeProto* target_ids;
    onnx::AttributeProto* target_nodeids;
    onnx::AttributeProto* target_treeids;
    onnx::AttributeProto* target_weights;

    onnx::AttributeProto* base_values;  // can be nullptr if model has no bias.
    onnx::AttributeProto* nodes_falsenodeids;
    onnx::AttributeProto* nodes_featureids;
    onnx::AttributeProto* nodes_hitrates;
    onnx::AttributeProto* nodes_missing_value_tracks_true;
    onnx::AttributeProto* nodes_modes;
    onnx::AttributeProto* nodes_nodeids;
    onnx::AttributeProto* nodes_treeids;
    onnx::AttributeProto* nodes_truenodeids;
    onnx::AttributeProto* nodes_values;

public:
    TTreesAttributes(
        bool isClassifierModel,
        bool hasBias,
        google::protobuf::RepeatedPtrField<onnx::AttributeProto>* treesNodeAttributes) {

#define GET_ATTR(attr, attr_type_suffix) \
        attr = treesNodeAttributes->Add(); \
        attr->set_name(#attr); \
        attr->set_type(onnx::AttributeProto_AttributeType_##attr_type_suffix);

        if (isClassifierModel) {
            GET_ATTR(class_ids, INTS);
            GET_ATTR(class_nodeids, INTS);
            GET_ATTR(class_treeids, INTS);
            GET_ATTR(class_weights, FLOATS);

            target_ids = nullptr;
            target_nodeids = nullptr;
            target_treeids = nullptr;
            target_weights = nullptr;
        } else {
            class_ids = nullptr;
            class_nodeids = nullptr;
            class_treeids = nullptr;
            class_weights = nullptr;

            GET_ATTR(target_ids, INTS);
            GET_ATTR(target_nodeids, INTS);
            GET_ATTR(target_treeids, INTS);
            GET_ATTR(target_weights, FLOATS);
        }

        if (hasBias) {
            GET_ATTR(base_values, FLOATS);
        }
        GET_ATTR(nodes_falsenodeids, INTS);
        GET_ATTR(nodes_featureids, INTS);
        GET_ATTR(nodes_hitrates, FLOATS);
        GET_ATTR(nodes_missing_value_tracks_true, INTS);
        GET_ATTR(nodes_modes, STRINGS);
        GET_ATTR(nodes_nodeids, INTS);
        GET_ATTR(nodes_treeids, INTS);
        GET_ATTR(nodes_truenodeids, INTS);
        GET_ATTR(nodes_values, FLOATS);

#undef GET_ATTR
    }

    TTreesAttributes(
        const bool isClassifierModel,
        google::protobuf::RepeatedPtrField<onnx::AttributeProto>& attributes) {

#define SET_ATTR(attr, new_attr) \
        if (new_attr.name() == #attr) { \
            attr = &new_attr; \
        }
        if (isClassifierModel) {
            target_ids = nullptr;
            target_nodeids = nullptr;
            target_treeids = nullptr;
            target_weights = nullptr;
        } else {
            class_ids = nullptr;
            class_nodeids = nullptr;
            class_treeids = nullptr;
            class_weights = nullptr;
        }
        base_values = nullptr;

        for (auto& attribute : attributes) {
            if (isClassifierModel) {
                SET_ATTR(class_ids, attribute);
                SET_ATTR(class_nodeids, attribute);
                SET_ATTR(class_treeids, attribute);
                SET_ATTR(class_weights, attribute);
            } else {
                SET_ATTR(target_ids, attribute);
                SET_ATTR(target_nodeids, attribute);
                SET_ATTR(target_treeids, attribute);
                SET_ATTR(target_weights, attribute);
            }

            SET_ATTR(base_values, attribute);
            SET_ATTR(nodes_falsenodeids, attribute);
            SET_ATTR(nodes_featureids, attribute);
            SET_ATTR(nodes_hitrates, attribute);
            SET_ATTR(nodes_missing_value_tracks_true, attribute);
            SET_ATTR(nodes_modes, attribute);
            SET_ATTR(nodes_nodeids, attribute);
            SET_ATTR(nodes_treeids, attribute);
            SET_ATTR(nodes_truenodeids, attribute);
            SET_ATTR(nodes_values, attribute);
        }
#undef SET_ATTR
    }
};


static void AddTree(
    const TModelTrees& trees,
    i64 treeIdx,
    bool isClassifierModel,
    const TVector<THashMap<int, ui32>>& oneHotValuesToIdx,
    int numFloatFeatures,
    TTreesAttributes* treesAttributes) {

    i64 nodeIdx = 0;

    // Process splits
    for (auto depth : xrange(trees.GetModelTreeData()->GetTreeSizes()[treeIdx])) {
        const auto& split = trees.GetBinFeatures()[
            trees.GetModelTreeData()->GetTreeSplits()[trees.GetModelTreeData()->GetTreeStartOffsets()[treeIdx] + (trees.GetModelTreeData()->GetTreeSizes()[treeIdx] - 1 - depth)]];

        int splitFlatFeatureIdx = 0;
        TString nodeMode;
        i64 missingValueTracksTrue = 0;
        float splitValue = 0.0f;

        if (split.Type == ESplitType::FloatFeature) {
            const auto& floatFeature = trees.GetFloatFeatures()[split.FloatFeature.FloatFeature];
            nodeMode = TModeNode::BRANCH_GT;
            if (floatFeature.NanValueTreatment == TFloatFeature::ENanValueTreatment::AsTrue) {
                missingValueTracksTrue = 1;
            }
            splitValue = split.FloatFeature.Split;
            // When there are no one-hot features, the input tensor uses original flat indices.
            // When one-hot features are present, float features are concatenated first
            // and renumbered starting from 0.
            if (trees.GetOneHotFeatures().empty()) {
                splitFlatFeatureIdx = floatFeature.Position.FlatIndex;
            } else {
                splitFlatFeatureIdx = split.FloatFeature.FloatFeature;
            }
        } else if (split.Type == ESplitType::OneHotFeature) {
            splitFlatFeatureIdx = numFloatFeatures + split.OneHotFeature.CatFeatureIdx;
            nodeMode = TModeNode::BRANCH_EQ;
            CB_ENSURE(
                split.OneHotFeature.CatFeatureIdx < (int)oneHotValuesToIdx.size(),
                "Invalid CatFeatureIdx in OneHotFeature split");
            const auto& valueMap = oneHotValuesToIdx[split.OneHotFeature.CatFeatureIdx];
            auto it = valueMap.find(split.OneHotFeature.Value);
            CB_ENSURE(
                it != valueMap.end(),
                "OneHotFeature value not found in categorical mapping for feature " << split.OneHotFeature.CatFeatureIdx);
            splitValue = static_cast<float>(it->second);
            missingValueTracksTrue = 0;
        } else {
            CB_ENSURE(
                false,
                "Only FloatFeature and OneHotFeature splits are supported in ONNX-ML format export"
            );
        }

        i64 endNodeIdx = 2*nodeIdx + 1;
        for (; nodeIdx < endNodeIdx; ++nodeIdx) {
            treesAttributes->nodes_treeids->add_ints(treeIdx);
            treesAttributes->nodes_nodeids->add_ints(nodeIdx);

            treesAttributes->nodes_modes->add_strings(nodeMode);

            treesAttributes->nodes_featureids->add_ints((i64)splitFlatFeatureIdx);
            treesAttributes->nodes_values->add_floats(splitValue);
            treesAttributes->nodes_falsenodeids->add_ints(2*nodeIdx + 1);
            treesAttributes->nodes_truenodeids->add_ints(2*nodeIdx + 2);
            treesAttributes->nodes_missing_value_tracks_true->add_ints(missingValueTracksTrue);
            treesAttributes->nodes_hitrates->add_floats(1.0f);
        }
    }

    // Process leafs
    auto applyData = trees.GetApplyData();
    const double* leafValue = trees.GetModelTreeData()->GetLeafValues().begin() + applyData->TreeFirstLeafOffsets[treeIdx];

    for (i64 endNodeIdx = 2*nodeIdx + 1; nodeIdx < endNodeIdx; ++nodeIdx) {
        treesAttributes->nodes_treeids->add_ints(treeIdx);
        treesAttributes->nodes_nodeids->add_ints(nodeIdx);

        treesAttributes->nodes_modes->add_strings(TModeNode::LEAF);

        // add dummy values because nodes_* must have equal length
        treesAttributes->nodes_featureids->add_ints(0);
        treesAttributes->nodes_values->add_floats(0.0f);
        treesAttributes->nodes_falsenodeids->add_ints(0);
        treesAttributes->nodes_truenodeids->add_ints(0);
        treesAttributes->nodes_missing_value_tracks_true->add_ints(0);
        treesAttributes->nodes_hitrates->add_floats(1.0f);

        if (isClassifierModel) {
            if (trees.GetDimensionsCount() > 1) {
                for (auto approxIdx : xrange(trees.GetDimensionsCount())) {
                    treesAttributes->class_treeids->add_ints(treeIdx);
                    treesAttributes->class_nodeids->add_ints(nodeIdx);

                    treesAttributes->class_ids->add_ints(approxIdx);
                    treesAttributes->class_weights->add_floats((float)*leafValue);
                    ++leafValue;
                }
            } else {
                treesAttributes->class_treeids->add_ints(treeIdx);
                treesAttributes->class_nodeids->add_ints(nodeIdx);
                treesAttributes->class_ids->add_ints(0);
                treesAttributes->class_weights->add_floats(-(float)*leafValue);

                treesAttributes->class_treeids->add_ints(treeIdx);
                treesAttributes->class_nodeids->add_ints(nodeIdx);
                treesAttributes->class_ids->add_ints(1);
                treesAttributes->class_weights->add_floats((float)*leafValue);

                ++leafValue;
            }
        } else {
            Y_ASSERT(trees.GetDimensionsCount() == 1);

            treesAttributes->target_treeids->add_ints(treeIdx);
            treesAttributes->target_nodeids->add_ints(nodeIdx);

            treesAttributes->target_ids->add_ints(0);
            treesAttributes->target_weights->add_floats((float)*leafValue);
            ++leafValue;
        }
    }
}


void NCB::NOnnx::ConvertTreeToOnnxGraph(
    const TFullModel& model,
    const TMaybe<TString>& onnxGraphName,
    onnx::GraphProto* onnxGraph,
    const THashMap<ui32, TString>* catFeaturesHashToString) {

    const bool isClassifierModel = IsClassifierModel(model);

    const TModelTrees& trees = *model.ModelTrees;

    onnxGraph->set_name(onnxGraphName.GetOrElse("CatBoostModel"));

    const int numFloatFeatures = (int)trees.GetFloatFeatures().size();

    // Build oneHotValuesToIdx mapping for categorical features
    TVector<THashMap<int, ui32>> oneHotValuesToIdx;
    if (!trees.GetOneHotFeatures().empty()) {
        CB_ENSURE_INTERNAL(
            catFeaturesHashToString,
            "catFeaturesHashToString must be provided for models with one-hot features");
        oneHotValuesToIdx.resize(trees.GetCatFeatures().size());
        for (const auto& oneHotFeature : trees.GetOneHotFeatures()) {
            auto& oneHotValuesToIdxMap = oneHotValuesToIdx[oneHotFeature.CatFeatureIndex];
            for (auto i : xrange(oneHotFeature.Values.size())) {
                oneHotValuesToIdxMap.emplace(oneHotFeature.Values[i], (ui32)i);
            }
        }
    }

    TString treesInputName;
    if (!trees.GetOneHotFeatures().empty()) {
        // Float features input
        InitValueInfo(
            "features",
            onnx::TensorProto_DataType_FLOAT,
            numFloatFeatures,
            onnxGraph->add_input());

        // String inputs and LabelEncoders for categorical features
        TVector<TString> catFeatureEncodedNames;
        for (const auto& catFeature : trees.GetCatFeatures()) {
            if (!catFeature.UsedInModel()) {
                continue;
            }
            TString inputName = catFeature.FeatureId.empty()
                ? "cat_feature_" + ToString(catFeature.Position.FlatIndex)
                : catFeature.FeatureId;
            TString encodedName = inputName + "_encoded";
            catFeatureEncodedNames.push_back(encodedName);

            InitValueInfo(
                inputName,
                onnx::TensorProto_DataType_STRING,
                /*secondDim*/ 1,
                onnxGraph->add_input());

            // Find corresponding OneHotFeature
            const TOneHotFeature* oneHotFeature = nullptr;
            for (const auto& oh : trees.GetOneHotFeatures()) {
                if (oh.CatFeatureIndex == catFeature.Position.Index) {
                    oneHotFeature = &oh;
                    break;
                }
            }
            CB_ENSURE_INTERNAL(oneHotFeature, "No OneHotFeature found for categorical feature " << catFeature.Position.Index);

            onnx::NodeProto* labelEncoderNode = onnxGraph->add_node();
            labelEncoderNode->set_domain(onnx::AI_ONNX_ML_DOMAIN);
            labelEncoderNode->set_op_type("LabelEncoder");
            labelEncoderNode->add_input(inputName);
            labelEncoderNode->add_output(encodedName);

            TVector<TString> keysStrings;
            TVector<float> valuesFloats;
            for (auto i : xrange(oneHotFeature->Values.size())) {
                keysStrings.push_back(catFeaturesHashToString->at((ui32)oneHotFeature->Values[i]));
                valuesFloats.push_back((float)i);
            }

            AddAttribute("keys_strings", keysStrings, labelEncoderNode);
            AddAttribute("values_floats", valuesFloats, labelEncoderNode);
            AddAttribute("default_float", -1.0f, labelEncoderNode);
        }

        // Concatenate float features and encoded categorical features
        onnx::NodeProto* concatNode = onnxGraph->add_node();
        concatNode->set_op_type("Concat");
        AddAttribute("axis", (i64)1, concatNode);
        concatNode->add_input("features");
        for (const auto& encodedName : catFeatureEncodedNames) {
            concatNode->add_input(encodedName);
        }
        treesInputName = "all_features";
        concatNode->add_output(treesInputName);
    } else {
        InitValueInfo(
            "features",
            onnx::TensorProto_DataType_FLOAT,
            trees.GetFlatFeatureVectorExpectedSize(),
            onnxGraph->add_input());
        treesInputName = "features";
    }

    onnx::NodeProto* treesNode = onnxGraph->add_node();
    treesNode->set_domain(onnx::AI_ONNX_ML_DOMAIN);
    treesNode->add_input(treesInputName);

    if (isClassifierModel) {
        treesNode->set_op_type("TreeEnsembleClassifier");

        TVector<i64> classLabelsInt64;
        TVector<TString> classLabelsString;
        GetClassLabels(model, &classLabelsInt64, &classLabelsString);

        AddClassLabelsAttribute(classLabelsInt64, classLabelsString, treesNode);
        AddAttribute(
            "post_transform",
            trees.GetDimensionsCount() == 1 ? "LOGISTIC" : "SOFTMAX",
            treesNode
        );

        InitValueInfo(
            "label",
            classLabelsString.empty() ? onnx::TensorProto_DataType_INT64 : onnx::TensorProto_DataType_STRING,
            /*secondDim*/ Nothing(),
            onnxGraph->add_output()
        );
        treesNode->add_output("label");

        InitValueInfo(
            "probability_tensor",
            onnx::TensorProto_DataType_FLOAT,
            trees.GetDimensionsCount() == 1 ? 2 : trees.GetDimensionsCount(),
            onnxGraph->add_value_info()
        );
        treesNode->add_output("probability_tensor");

        onnx::NodeProto* zipMapNode = onnxGraph->add_node();
        zipMapNode->set_domain(onnx::AI_ONNX_ML_DOMAIN);
        zipMapNode->set_op_type("ZipMap");

        zipMapNode->add_input("probability_tensor");

        InitProbabilitiesOutput(
            "probabilities",
            classLabelsString.empty() ? onnx::TensorProto_DataType_INT64 : onnx::TensorProto_DataType_STRING,
            onnxGraph->add_output());

        zipMapNode->add_output("probabilities");

        AddClassLabelsAttribute(classLabelsInt64, classLabelsString, zipMapNode);
    } else {
        treesNode->set_op_type("TreeEnsembleRegressor");

        AddAttribute("post_transform", "NONE", treesNode);
        AddAttribute("n_targets", i64(1), treesNode);

        InitValueInfo(
            "predictions",
            onnx::TensorProto_DataType_FLOAT,
            /*secondDim*/ Nothing(),
            onnxGraph->add_output()
        );
        treesNode->add_output("predictions");
    }
    auto scaleAndBias = model.GetScaleAndBias();
    TTreesAttributes treesAttributes(isClassifierModel, !scaleAndBias.IsZeroBias(), treesNode->mutable_attribute());

    if (!scaleAndBias.IsZeroBias()) {
        if (isClassifierModel && (trees.GetDimensionsCount() == 1)) {
            const float bias = float(scaleAndBias.GetOneDimensionalBias());
            treesAttributes.base_values->add_floats(-bias);
            treesAttributes.base_values->add_floats(bias);
        } else {
            auto bias = scaleAndBias.GetBiasRef();
            size_t biasSize = bias.size();
            CB_ENSURE_INTERNAL(
                biasSize == trees.GetDimensionsCount(),
            "Inappropraite dimension of bias, should be " << trees.GetDimensionsCount() << " or 0, found " << biasSize);
            for (auto b : bias) {
                treesAttributes.base_values->add_floats(b);
            }
        }
    }
    for (auto treeIdx : xrange(trees.GetTreeCount())) {
        AddTree(trees, treeIdx, isClassifierModel, oneHotValuesToIdx, numFloatFeatures, &treesAttributes);
    }
}


static void ConfigureMetaInfo(const onnx::ModelProto& onnxModel, TFullModel* fullModel) {
    THashMap<TString, TString> modelInfo;

    for (int idx = 0; idx < onnxModel.metadata_props_size(); ++idx) {
        auto& property = onnxModel.metadata_props(idx);
        TString key, value;
        CB_ENSURE(property.has_key(), "Missing key in value info");
        key = property.key();
        if (property.has_value()) {
            value = property.value();
        }
        modelInfo[key] = value;
    }

    fullModel->ModelInfo = modelInfo;
}


static void PrepareTrees(
    const TTreesAttributes& treesAttributes,
    const bool isClassifierModel,
    TVector<THashMap<int, NCB::NOnnx::TOnnxNode>>* trees,
    int* approxDimension,
    TVector<TFloatFeature>* floatFeatures,
    TVector<TCatFeature>* catFeatures,
    THashMap<int, int>* flatFeatureIndexToPerTypeIndex,
    const THashMap<int, THashMap<int, TString>>* catFeatureIdxToEnumIdToString = nullptr
) {
    // First pass: determine which features are categorical based on node modes
    THashSet<int> categoricalFeatureIds;
    for (auto idx = 0; idx < treesAttributes.nodes_treeids->ints_size(); ++idx) {
        const TString nodeMode = treesAttributes.nodes_modes->strings(idx);
        if (nodeMode != TModeNode::LEAF) {
            if (nodeMode == TModeNode::BRANCH_EQ || nodeMode == TModeNode::BRANCH_NEQ) {
                categoricalFeatureIds.insert(treesAttributes.nodes_featureids->ints(idx));
            }
        }
    }

    // Build mapping from flat feature index to per-type index
    THashSet<int> floatFeatureIds;
    THashSet<int> catFeatureIds;
    for (auto idx = 0; idx < treesAttributes.nodes_treeids->ints_size(); ++idx) {
        const TString nodeMode = treesAttributes.nodes_modes->strings(idx);
        if (nodeMode == TModeNode::LEAF) {
            continue;
        }
        int flatFeatureId = treesAttributes.nodes_featureids->ints(idx);
        if (categoricalFeatureIds.contains(flatFeatureId)) {
            catFeatureIds.insert(flatFeatureId);
        } else {
            floatFeatureIds.insert(flatFeatureId);
        }
    }

    // Sort features by flat index to satisfy CatBoost's sorted requirement
    TVector<int> sortedFloatFeatureIds(floatFeatureIds.begin(), floatFeatureIds.end());
    TVector<int> sortedCatFeatureIds(catFeatureIds.begin(), catFeatureIds.end());
    Sort(sortedFloatFeatureIds.begin(), sortedFloatFeatureIds.end());
    Sort(sortedCatFeatureIds.begin(), sortedCatFeatureIds.end());

    for (int i = 0; i < (int)sortedFloatFeatureIds.size(); ++i) {
        (*flatFeatureIndexToPerTypeIndex)[sortedFloatFeatureIds[i]] = i;
    }
    for (int i = 0; i < (int)sortedCatFeatureIds.size(); ++i) {
        (*flatFeatureIndexToPerTypeIndex)[sortedCatFeatureIds[i]] = i;
    }

    floatFeatures->resize(sortedFloatFeatureIds.size());
    catFeatures->resize(sortedCatFeatureIds.size());
    for (int i = 0; i < (int)sortedFloatFeatureIds.size(); ++i) {
        (*floatFeatures)[i].Position.Index = i;
        (*floatFeatures)[i].Position.FlatIndex = sortedFloatFeatureIds[i];
    }
    for (int i = 0; i < (int)sortedCatFeatureIds.size(); ++i) {
        (*catFeatures)[i].Position.Index = i;
        (*catFeatures)[i].Position.FlatIndex = sortedCatFeatureIds[i];
        (*catFeatures)[i].SetUsedInModel(true);
    }

    TVector<TSet<float>> floatFeatureBorders(floatFeatures->size());

    // Second pass: process all nodes
    for (auto idx = 0; idx < treesAttributes.nodes_treeids->ints_size(); ++idx) {
        NCB::NOnnx::TOnnxNode node;
        const size_t treeId = treesAttributes.nodes_treeids->ints(idx);
        const int nodeId = treesAttributes.nodes_nodeids->ints(idx);
        node.FalseNodeId = treesAttributes.nodes_falsenodeids->ints(idx);
        node.TrueNodeId = treesAttributes.nodes_truenodeids->ints(idx);

        const TString nodeMode = treesAttributes.nodes_modes->strings(idx);

        if (nodeMode == TModeNode::LEAF) {
            node.Type = EType::Leaf;
        } else {
            node.Type = EType::Inner;

            TModelSplit split;
            
            if (nodeMode == TModeNode::BRANCH_LEQ || nodeMode == TModeNode::BRANCH_LT) {
                std::swap(node.TrueNodeId, node.FalseNodeId);
            }

            if (nodeMode == TModeNode::BRANCH_LEQ || nodeMode == TModeNode::BRANCH_LT ||
                nodeMode == TModeNode::BRANCH_GTE || nodeMode == TModeNode::BRANCH_GT) {
                split.Type = ESplitType::FloatFeature;
                int perTypeIndex = flatFeatureIndexToPerTypeIndex->at(treesAttributes.nodes_featureids->ints(idx));
                split.FloatFeature.FloatFeature = perTypeIndex;
                split.FloatFeature.Split = treesAttributes.nodes_values->floats(idx);
                
                if (treesAttributes.nodes_missing_value_tracks_true->ints(idx) == 1) {
                    (*floatFeatures)[perTypeIndex].NanValueTreatment = ENanValueTreatment::AsTrue;
                }
                floatFeatureBorders[perTypeIndex].insert(split.FloatFeature.Split);
            } else if (nodeMode == TModeNode::BRANCH_EQ || nodeMode == TModeNode::BRANCH_NEQ) {
                split.Type = ESplitType::OneHotFeature;
                int perTypeIndex = flatFeatureIndexToPerTypeIndex->at(treesAttributes.nodes_featureids->ints(idx));
                split.OneHotFeature.CatFeatureIdx = perTypeIndex;
                
                float floatValue = treesAttributes.nodes_values->floats(idx);
                int intValue = (int)floatValue;
                CB_ENSURE((float)intValue == floatValue,
                    "Categorical feature value must be an integer, got " << floatValue);
                
                TString stringValue;
                if (catFeatureIdxToEnumIdToString && catFeatureIdxToEnumIdToString->contains(perTypeIndex)) {
                    const auto& enumIdToString = catFeatureIdxToEnumIdToString->at(perTypeIndex);
                    CB_ENSURE(enumIdToString.contains(intValue),
                        "Categorical feature value " << intValue << " not found in LabelEncoder for feature " << perTypeIndex);
                    stringValue = enumIdToString.at(intValue);
                } else {
                    stringValue = ToString(intValue);
                }
                split.OneHotFeature.Value = CalcCatFeatureHash(stringValue);
                
                if (nodeMode == TModeNode::BRANCH_NEQ) {
                    std::swap(node.TrueNodeId, node.FalseNodeId);
                }
            } else {
                CB_ENSURE(false, "Undefined mode of node " << nodeMode);
            }

            node.SplitCondition = split;
        }

        //add node to tree
        if (treeId >= trees->size()) {
            trees->resize(treeId + 1);
        }
        (*trees)[treeId][nodeId] = node;
    }

    //to set borders to floatfeatures
    for (auto floatFeatureIdx : xrange(floatFeatures->size())) {
        (*floatFeatures)[floatFeatureIdx].Borders.assign(
            floatFeatureBorders[floatFeatureIdx].begin(),
            floatFeatureBorders[floatFeatureIdx].end());
    }

    //consider leaves
    auto treatLeafNode = [trees](
        onnx::AttributeProto* treeIds,
        onnx::AttributeProto* nodeIds,
        onnx::AttributeProto* values
    ) {
        for (auto idx = 0 ; idx < treeIds->ints_size(); ++idx) {
            const size_t treeId = treeIds->ints(idx);
            const int nodeId = nodeIds->ints(idx);
            const double value = values->floats(idx);
            CB_ENSURE(treeId < trees->size(), "Invalid class_nodeId " << treeId);
            (*trees)[treeId][nodeId].Values.emplace_back(value);
        }
    };
    if (isClassifierModel) {
        treatLeafNode(treesAttributes.class_treeids, treesAttributes.class_nodeids, treesAttributes.class_weights);
        //setup approxDimension
        const auto anyIdxNodeIdLeaf = treesAttributes.class_nodeids->ints(0);
        *approxDimension = static_cast<int>((*trees)[0][anyIdxNodeIdLeaf].Values.size());
    } else {
        treatLeafNode(treesAttributes.target_treeids, treesAttributes.target_nodeids, treesAttributes.target_weights);
    }
}


static THolder<TNonSymmetricTreeNode> BuildNonSymmetricTree(
    const THashMap<int, NCB::NOnnx::TOnnxNode>& tree,
    const int nodeId
) {
    THolder<TNonSymmetricTreeNode> head = MakeHolder<TNonSymmetricTreeNode>();
    const auto node = tree.at(nodeId);

    switch (node.Type) {
        case EType::Leaf: {
            if (node.Values.size() == 1) {
                head->Value = node.Values[0];
            } else {
                head->Value = node.Values;
            }
            return head;
        }
        case EType::Inner: {
            head->Value = TNonSymmetricTreeNode::TEmptyValue();
            head->SplitCondition = node.SplitCondition;
            CB_ENSURE(tree.contains(node.FalseNodeId), "unexpected false node id");
            CB_ENSURE(tree.contains(node.TrueNodeId), "unexpected true node id");
            head->Left = BuildNonSymmetricTree(tree, node.FalseNodeId);
            head->Right = BuildNonSymmetricTree(tree, node.TrueNodeId);
            return head;
        }
        default:
            CB_ENSURE(false, "Unexpected ONNX node type");
    }
}


static void ConfigureSymmetricTrees(const onnx::GraphProto& onnxGraph, TFullModel* fullModel) {

    const auto& nodes = onnxGraph.node();
    // Find the TreeEnsemble node (skip preprocessing nodes like LabelEncoder, Concat)
    const onnx::NodeProto* treesNode = nullptr;
    for (const auto& node : nodes) {
        if (node.op_type() == "TreeEnsembleClassifier" || node.op_type() == "TreeEnsembleRegressor") {
            treesNode = &node;
            break;
        }
    }
    CB_ENSURE(treesNode != nullptr, "No TreeEnsembleClassifier or TreeEnsembleRegressor node found in ONNX graph");

    const bool isClassifierModel = (treesNode->op_type() == "TreeEnsembleClassifier");

    auto attributes = treesNode->attribute();
    TTreesAttributes treesAttributes(isClassifierModel, attributes);

    // Build mapping from LabelEncoder enumerated ids to original string values
    THashMap<int, THashMap<int, TString>> catFeatureIdxToEnumIdToString;
    {
        // Find Concat node to determine catFeatureIdx order
        const onnx::NodeProto* concatNode = nullptr;
        THashMap<TString, int> encodedNameToCatFeatureIdx;
        for (const auto& node : nodes) {
            if (node.op_type() == "Concat") {
                concatNode = &node;
                break;
            }
        }
        if (concatNode != nullptr) {
            int catFeatureIdx = 0;
            for (int i = 0; i < concatNode->input_size(); ++i) {
                TString inputName = concatNode->input(i);
                if (inputName != "features") {
                    encodedNameToCatFeatureIdx[inputName] = catFeatureIdx++;
                }
            }
        }

        // Find LabelEncoder nodes and build enum id -> string mapping
        for (const auto& node : nodes) {
            if (node.op_type() == "LabelEncoder" && node.domain() == onnx::AI_ONNX_ML_DOMAIN) {
                TString outputName = node.output(0);
                int catFeatureIdx = encodedNameToCatFeatureIdx.Value(outputName, -1);
                if (catFeatureIdx < 0) {
                    continue; // LabelEncoder not part of Concat (shouldn't happen)
                }

                TVector<TString> keysStrings;
                TVector<float> valuesFloats;
                for (const auto& attr : node.attribute()) {
                    if (attr.name() == "keys_strings") {
                        for (const auto& s : attr.strings()) {
                            keysStrings.push_back(TString(s));
                        }
                    } else if (attr.name() == "values_floats") {
                        for (auto f : attr.floats()) {
                            valuesFloats.push_back(f);
                        }
                    }
                }
                CB_ENSURE(keysStrings.size() == valuesFloats.size(),
                    "LabelEncoder keys_strings and values_floats must have the same size");
                for (size_t i = 0; i < keysStrings.size(); ++i) {
                    int enumId = (int)valuesFloats[i];
                    catFeatureIdxToEnumIdToString[catFeatureIdx][enumId] = keysStrings[i];
                }
            }
        }
    }

    TVector<TFloatFeature> floatFeatures;
    TVector<TCatFeature> catFeatures;
    THashMap<int, int> flatFeatureIndexToPerTypeIndex;

    TVector<THashMap<int, NCB::NOnnx::TOnnxNode>> trees;
    int approxDimension = 1;
    PrepareTrees(treesAttributes, isClassifierModel, &trees, &approxDimension, &floatFeatures, &catFeatures, &flatFeatureIndexToPerTypeIndex,
                 catFeatureIdxToEnumIdToString.empty() ? nullptr : &catFeatureIdxToEnumIdToString);

    TNonSymmetricTreeModelBuilder treeBuilder(floatFeatures, catFeatures, {}, {}, approxDimension);

    for (const auto& tree : trees) {
        treeBuilder.AddTree(BuildNonSymmetricTree(tree, 0));
    }

    treeBuilder.Build(fullModel->ModelTrees.GetMutable());
    if (treesAttributes.base_values != nullptr) {
        TVector<double> bias;
        for (size_t idx: xrange(treesAttributes.base_values->floats_size())) {
            bias.push_back(treesAttributes.base_values->floats(idx));
        }
        fullModel->SetScaleAndBias({1., bias});
    }

    fullModel->UpdateDynamicData();
}



void NCB::NOnnx::ConvertOnnxToCatboostModel(const onnx::ModelProto& onnxModel, TFullModel* fullModel) {
    //InitMetaData
    ConfigureMetaInfo(onnxModel, fullModel);

    onnx::GraphProto onnxGraph = onnxModel.graph();
    ConfigureSymmetricTrees(onnxGraph, fullModel);
}
