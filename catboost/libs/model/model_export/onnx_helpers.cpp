#include "onnx_helpers.h"

#include <catboost/libs/model/model_build_helper.h>
#include <catboost/libs/model/static_ctr_provider.h>
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

#include <util/generic/algorithm.h>
#include <util/generic/array_ref.h>
#include <util/generic/hash_set.h>
#include <util/generic/mapfindptr.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/generic/utility.h>
#include <util/generic/xrange.h>
#include <util/generic/ymath.h>
#include <util/string/cast.h>
#include <util/string/join.h>
#include <util/system/compiler.h>
#include <util/system/yassert.h>

#include <cmath>
#include <limits>
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

static bool HasNonDefaultBinaryThreshold(const TFullModel& model);

void NCB::NOnnx::InitMetadata(
    const TFullModel& model,
    const NJson::TJsonValue& userParameters,
    onnx::ModelProto* onnxModel) {

    // Versions 4 and above essentially just introduce new types we don't need.
    onnxModel->set_ir_version(3);

    onnx::OperatorSetIdProto* opset = onnxModel->add_opset_import();
    opset->set_domain(onnx::AI_ONNX_ML_DOMAIN);
    opset->set_version(2);

    const bool hashedInputs = userParameters.Has("onnx_cat_features_input_type")
        && userParameters["onnx_cat_features_input_type"].GetStringSafe() == "hashed";
    if (hashedInputs || !model.ModelTrees->GetOneHotFeatures().empty() || HasNonDefaultBinaryThreshold(model)) {
        // Concat belongs to the standard ONNX operator set. ONNX Runtime guarantees
        // backward compatibility starting at standard-domain opset 7. Hashed
        // CTR inference also uses GatherElements, introduced in opset 11.
        auto* standardOpset = onnxModel->add_opset_import();
        standardOpset->set_domain("");
        standardOpset->set_version(hashedInputs ? 11 : 7);
    }

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


static bool HasNonDefaultBinaryThreshold(const TFullModel& model) {
    return model.ModelTrees->GetDimensionsCount() == 1
        && IsClassifierModel(model)
        && model.GetBinClassProbabilityThreshold() != DEFAULT_BINCLASS_PROBABILITY_THRESHOLD;
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
        case NJson::JSON_BOOLEAN:
            classLabelsInt64->reserve(classLabels.size());
            for (const NJson::TJsonValue& classLabel : classLabels) {
                classLabelsInt64->push_back(classLabel.GetBooleanSafe());
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


class TOnnxGraphBuilder {
public:
    explicit TOnnxGraphBuilder(onnx::GraphProto* graph)
        : Graph(graph)
    {
        for (const auto& input : graph->input()) {
            Names.insert(input.name());
        }
        for (const auto& node : graph->node()) {
            for (const auto& output : node.output()) {
                Names.insert(output);
            }
        }
        for (const auto& name : {"predictions", "label", "probabilities", "trees_input", "probability_tensor"}) {
            Names.insert(name);
        }
    }

    TString Node(const TString& op, const TVector<TString>& inputs, const TString& output = {}) {
        auto* node = Graph->add_node();
        node->set_op_type(op);
        for (const auto& input : inputs) {
            node->add_input(input);
        }
        const TString result = output.empty() ? NewName() : output;
        node->add_output(result);
        return result;
    }

    onnx::NodeProto* LastNode() {
        return Graph->mutable_node(Graph->node_size() - 1);
    }

    TString Constant(const onnx::TensorProto& tensor) {
        const auto result = Node("Constant", {});
        auto* attr = LastNode()->add_attribute();
        attr->set_name("value");
        attr->set_type(onnx::AttributeProto_AttributeType_TENSOR);
        *attr->mutable_t() = tensor;
        return result;
    }

    TString Float(float value) {
        onnx::TensorProto tensor;
        tensor.set_data_type(onnx::TensorProto_DataType_FLOAT);
        tensor.add_float_data(value);
        return Constant(tensor);
    }

    TString CastToInt(const TString& value) {
        const auto result = Node("Cast", {value});
        AddAttribute("to", i64(onnx::TensorProto_DataType_INT64), LastNode());
        return result;
    }

    TString NewName() {
        TString name;
        do {
            name = "__catboost_output_" + ToString(NextName++);
        } while (!Names.insert(name).second);
        return name;
    }

private:
    onnx::GraphProto* Graph;
    THashSet<TString> Names;
    size_t NextName = 0;
};

// Compare raw margins, rather than rounded probabilities: finite margins must
// still select the positive class at threshold 0 and the negative class at 1.
static void AddBinaryClassifierOutputs(
    const TFullModel& model,
    const TString& margin,
    onnx::GraphProto* graph) {

    TVector<i64> labelsInt;
    TVector<TString> labelsString;
    GetClassLabels(model, &labelsInt, &labelsString);
    CB_ENSURE(labelsInt.size() + labelsString.size() == 2,
        "ONNX binary classification requires exactly two class labels");
    TOnnxGraphBuilder builder(graph);
    const auto probability = builder.Node("Sigmoid", {margin});
    const auto probabilities = builder.Node("Concat", {builder.Node("Sub", {builder.Float(1), probability}), probability});
    AddAttribute("axis", i64(1), builder.LastNode());
    builder.Node("ZipMap", {probabilities}, "probabilities");
    builder.LastNode()->set_domain(onnx::AI_ONNX_ML_DOMAIN);
    AddClassLabelsAttribute(labelsInt, labelsString, builder.LastNode());

    // Rounding the threshold to float could turn a strictly greater margin into
    // a tie. Avoid the double cast when the threshold is exactly representable.
    const double logitThreshold = model.GetBinClassLogitThreshold();
    onnx::TensorProto threshold;
    auto comparedMargin = margin;
    if (double(float(logitThreshold)) == logitThreshold) {
        threshold.set_data_type(onnx::TensorProto_DataType_FLOAT);
        threshold.add_float_data(logitThreshold);
    } else {
        threshold.set_data_type(onnx::TensorProto_DataType_DOUBLE);
        threshold.add_double_data(logitThreshold);
        comparedMargin = builder.Node("Cast", {margin});
        AddAttribute("to", i64(onnx::TensorProto_DataType_DOUBLE), builder.LastNode());
    }
    const auto positive = builder.CastToInt(builder.Node("Greater", {comparedMargin, builder.Constant(threshold)}));
    const auto indices = builder.Node("Squeeze", {positive});
    AddAttribute("axes", TVector<i64>{1}, builder.LastNode());

    const auto labelType = labelsString.empty() ? onnx::TensorProto_DataType_INT64 : onnx::TensorProto_DataType_STRING;
    onnx::TensorProto labels;
    labels.set_data_type(labelType);
    labels.add_dims(2);
    for (auto value : labelsInt) {
        labels.add_int64_data(value);
    }
    for (const auto& value : labelsString) {
        labels.add_string_data(value);
    }
    builder.Node("Gather", {builder.Constant(labels), indices}, "label");
    InitValueInfo("label", labelType, Nothing(), graph->add_output());
    InitProbabilitiesOutput("probabilities", labelType, graph->add_output());
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


/*
 * Categorical features in ONNX-ML export.
 *
 * ONNX-ML TreeEnsemble* operators accept a single float tensor as input, so categorical features are
 * handled in the same way as in PMML export:
 *  - each categorical feature used in one-hot splits gets its own string input;
 *  - a LabelEncoder node maps its string values to indices in TOneHotFeature::Values (as floats),
 *    values not seen during training are mapped to -1 so they do not match any split;
 *  - a Concat node joins the numerical features tensor and the encoded categorical features
 *    (numerical features first), its output is the input of the TreeEnsemble* node.
 *
 * Feature indices in TreeEnsemble* nodes are:
 *  - index among float features for float features (the "num_features" input contains only float features
 *    in this case);
 *  - (float features count + position among exported categorical features) for categorical features.
 *
 * Models without one-hot features keep the original layout: the single "features" input has
 * GetFlatFeatureVectorExpectedSize() columns and float features are addressed by their flat indices.
 */

struct TOnnxFeaturesLayout {
    struct TFloatInput {
        int Index;  // flat index in ONNX model
        bool MissingValueTracksTrue;
    };

    struct TCatInput {
        int Index;  // flat index in ONNX model
        THashMap<int, ui32> ValuesToIdx; // hashed value -> index in TOneHotFeature::Values
    };

    TString TreesInputName;

    // Per-type indices can have gaps after DropUnusedFeatures().
    THashMap<int, TFloatInput> FloatInputs;     // [floatFeatureIdx]
    THashMap<int, TCatInput> CatInputs;         // [catFeatureIdx]
    THashMap<TModelCtr, int> CtrInputs;
};


static TString GetCatFeatureInputName(const TCatFeature& catFeature) {
    if (catFeature.FeatureId) {
        return catFeature.FeatureId;
    }
    return "cat_feature_" + ToString(catFeature.Position.FlatIndex);
}


static TOnnxFeaturesLayout InitInputs(
    const TModelTrees& trees,
    const THashMap<ui32, TString>* catFeaturesHashToString,
    onnx::GraphProto* onnxGraph) {

    TOnnxFeaturesLayout layout;
    const bool hasOneHotFeatures = !trees.GetOneHotFeatures().empty();
    int floatInputIndex = 0;
    for (const auto& floatFeature : trees.GetFloatFeatures()) {
        layout.FloatInputs.emplace(floatFeature.Position.Index, TOnnxFeaturesLayout::TFloatInput{
            hasOneHotFeatures ? floatInputIndex++ : floatFeature.Position.FlatIndex,
            floatFeature.NanValueTreatment == TFloatFeature::ENanValueTreatment::AsTrue
        });
    }

    if (!hasOneHotFeatures) {
        InitValueInfo(
            "features",
            onnx::TensorProto_DataType_FLOAT,
            trees.GetFlatFeatureVectorExpectedSize(),
            onnxGraph->add_input());
        layout.TreesInputName = "features";

        return layout;
    }

    CB_ENSURE(
        catFeaturesHashToString,
        "Categorical features hash to string mapping is required for ONNX-ML export of models with"
        " categorical features (pass the training dataset to the export function)"
    );

    const int floatFeaturesCount = (int)trees.GetFloatFeatures().size();

    TVector<TString> concatInputs;
    if (floatFeaturesCount > 0) {
        InitValueInfo(
            "num_features",
            onnx::TensorProto_DataType_FLOAT,
            floatFeaturesCount,
            onnxGraph->add_input());
        concatInputs.push_back("num_features");
    }

    THashSet<TString> usedNames = {
        "num_features", "trees_input", "label", "probability_tensor", "probabilities", "predictions"
    };

    THashMap<int, const TCatFeature*> catFeaturesByIndex;
    for (const auto& catFeature : trees.GetCatFeatures()) {
        catFeaturesByIndex.emplace(catFeature.Position.Index, &catFeature);
    }

    // maximum integer that can be represented as float w/o loss
    constexpr size_t MAX_EXACT_FLOAT_INT = size_t(1) << std::numeric_limits<float>::digits;

    int exportedCatFeaturesCount = 0;
    for (const auto& oneHotFeature : trees.GetOneHotFeatures()) {
        const int catFeatureIdx = oneHotFeature.CatFeatureIndex;
        const auto& catFeature = *catFeaturesByIndex.at(catFeatureIdx);

        const TString inputName = GetCatFeatureInputName(catFeature);
        const TString encodedName = inputName + "_encoded";
        for (const auto& name : {inputName, encodedName}) {
            CB_ENSURE(
                usedNames.insert(name).second,
                "Cannot export model to ONNX-ML: feature name \"" << name << "\" is not unique"
            );
        }

        CB_ENSURE(
            oneHotFeature.Values.size() < MAX_EXACT_FLOAT_INT,
            "Cannot export model to ONNX-ML: categorical feature \"" << inputName << "\" has too many unique"
            " values to be represented as 32 bit float in LabedEncoder node values_floats"
        );

        InitValueInfo(
            inputName,
            onnx::TensorProto_DataType_STRING,
            /*secondDim*/ 1,
            onnxGraph->add_input());

        TVector<TString> keysStrings;
        TVector<float> valuesFloats;
        keysStrings.reserve(oneHotFeature.Values.size());
        valuesFloats.reserve(oneHotFeature.Values.size());
        auto& catInput = layout.CatInputs[catFeatureIdx];
        catInput.Index = floatFeaturesCount + exportedCatFeaturesCount;
        for (auto i : xrange(oneHotFeature.Values.size())) {
            const int hashedValue = oneHotFeature.Values[i];
            const TString* stringValue = catFeaturesHashToString->FindPtr((ui32)hashedValue);
            CB_ENSURE(
                stringValue,
                "Cannot find string value for categorical feature \"" << inputName << "\" hash "
                << (ui32)hashedValue << ", the dataset passed to the export must contain all categorical"
                " values used by the model"
            );
            keysStrings.push_back(*stringValue);
            valuesFloats.push_back((float)i);
            catInput.ValuesToIdx.emplace(hashedValue, (ui32)i);
        }

        onnx::NodeProto* labelEncoderNode = onnxGraph->add_node();
        labelEncoderNode->set_domain(onnx::AI_ONNX_ML_DOMAIN);
        labelEncoderNode->set_op_type("LabelEncoder");
        labelEncoderNode->add_input(inputName);
        labelEncoderNode->add_output(encodedName);
        AddAttribute("keys_strings", keysStrings, labelEncoderNode);
        AddAttribute("values_floats", valuesFloats, labelEncoderNode);
        AddAttribute("default_float", -1.0f, labelEncoderNode);

        concatInputs.push_back(encodedName);
        ++exportedCatFeaturesCount;
    }

    layout.TreesInputName = "trees_input";
    // ONNX requires nodes to be in topological order: encoders must precede Concat.
    onnx::NodeProto* concatNode = onnxGraph->add_node();
    concatNode->set_op_type("Concat");
    AddAttribute("axis", i64(1), concatNode);
    for (const auto& input : concatInputs) {
        concatNode->add_input(input);
    }
    concatNode->add_output(layout.TreesInputName);

    return layout;
}


// CTR projection keys are evaluated modulo 2^64 using unsigned ONNX arithmetic.
// Dividing both saved keys and runtime hashes by the common odd multiplier M^2
// preserves every key and collision, and makes single-category lookup direct.
class TOnnxCtrGraphBuilder {
public:
    explicit TOnnxCtrGraphBuilder(onnx::GraphProto* graph)
        : Graph(graph)
    {
        for (const auto& input : graph->input()) {
            Names.insert(input.name());
        }
        for (const auto& node : graph->node()) {
            for (const auto& output : node.output()) {
                Names.insert(output);
            }
        }
        for (const auto& name : {"predictions", "label", "probabilities", "trees_input"}) {
            Names.insert(name);
        }
    }

    TString Node(const TString& op, const TVector<TString>& inputs, const TString& output = {}) {
        auto* node = Graph->add_node();
        node->set_op_type(op);
        for (const auto& input : inputs) {
            node->add_input(input);
        }
        const TString result = output.empty() ? NewName() : output;
        node->add_output(result);
        return result;
    }

    onnx::NodeProto* LastNode() {
        return Graph->mutable_node(Graph->node_size() - 1);
    }

    TString Constant(const onnx::TensorProto& tensor) {
        const auto result = Node("Constant", {});
        auto* attr = LastNode()->add_attribute();
        attr->set_name("value");
        attr->set_type(onnx::AttributeProto_AttributeType_TENSOR);
        *attr->mutable_t() = tensor;
        return result;
    }

    TString Int(i64 value) {
        if (const auto* name = IntConstants.FindPtr(value)) {
            return *name;
        }
        onnx::TensorProto tensor;
        tensor.set_data_type(onnx::TensorProto_DataType_INT64);
        tensor.add_int64_data(value);
        return IntConstants[value] = Constant(tensor);
    }

    TString Float(float value) {
        onnx::TensorProto tensor;
        tensor.set_data_type(onnx::TensorProto_DataType_FLOAT);
        tensor.add_float_data(value);
        return Constant(tensor);
    }

    TString CastToInt(const TString& value) {
        const auto result = Node("Cast", {value});
        AddAttribute("to", i64(onnx::TensorProto_DataType_INT64), LastNode());
        return result;
    }

    TString FloatColumn(int index) {
        if (const auto* cached = FloatColumns.FindPtr(index)) {
            return *cached;
        }
        onnx::TensorProto indices;
        indices.set_data_type(onnx::TensorProto_DataType_INT64);
        indices.add_dims(1);
        indices.add_int64_data(index);
        const auto result = Node("ArrayFeatureExtractor", {"num_features", Constant(indices)});
        LastNode()->set_domain(onnx::AI_ONNX_ML_DOMAIN);
        FloatColumns.emplace(index, result);
        return result;
    }

    TString Projection(const TVector<TString>& terms) {
        CB_ENSURE(!terms.empty(), "Cannot export an empty CTR projection to ONNX");
        if (terms.size() == 1) {
            return terms.front();
        }
        // Native h <- M * (h + M * x). Factoring out M^2 gives k <- M * k + x.
        auto result = ToUnsigned(terms.front());
        for (size_t i = 1; i < terms.size(); ++i) {
            const auto key = std::make_pair(result, terms[i]);
            if (const auto* cached = ProjectionPrefixes.FindPtr(key)) {
                result = *cached;
            } else {
                const auto product = Node("Mul", {result, HashMultiplier()});
                result = Node("Add", {product, ToUnsigned(terms[i])});
                ProjectionPrefixes.emplace(key, result);
            }
        }
        return CastToInt(result);
    }

    TString Lookup(const TString& input, const TVector<i64>& keys, const TVector<float>& values, float missing) {
        if (keys.empty()) {
            onnx::TensorProto tensor;
            tensor.set_data_type(onnx::TensorProto_DataType_FLOAT);
            tensor.add_dims(1);
            tensor.add_float_data(missing);
            const auto result = Node("ConstantOfShape", {Node("Shape", {input})});
            auto* attr = LastNode()->add_attribute();
            attr->set_name("value");
            attr->set_type(onnx::AttributeProto_AttributeType_TENSOR);
            *attr->mutable_t() = tensor;
            return result;
        }
        const auto result = Node("LabelEncoder", {input});
        auto* node = LastNode();
        node->set_domain(onnx::AI_ONNX_ML_DOMAIN);
        AddAttribute("keys_int64s", keys, node);
        AddAttribute("values_floats", values, node);
        AddAttribute("default_float", missing, node);
        return result;
    }

private:
    TString ToUnsigned(const TString& input) {
        if (const auto* cached = UnsignedInputs.FindPtr(input)) {
            return *cached;
        }
        const auto result = Node("Cast", {input});
        AddAttribute("to", i64(onnx::TensorProto_DataType_UINT64), LastNode());
        UnsignedInputs.emplace(input, result);
        return result;
    }

    TString HashMultiplier() {
        if (Multiplier.empty()) {
            onnx::TensorProto tensor;
            tensor.set_data_type(onnx::TensorProto_DataType_UINT64);
            tensor.add_uint64_data(CalcHash(1, 0));
            Multiplier = Constant(tensor);
        }
        return Multiplier;
    }

    TString NewName() {
        TString name;
        do {
            name = "__catboost_ctr_" + ToString(NextName++);
        } while (!Names.insert(name).second);
        return name;
    }

    onnx::GraphProto* Graph;
    THashSet<TString> Names;
    THashMap<i64, TString> IntConstants;
    THashMap<int, TString> FloatColumns;
    THashMap<TString, TString> UnsignedInputs;
    THashMap<std::pair<TString, TString>, TString> ProjectionPrefixes;
    TString Multiplier;
    size_t NextName = 0;
};

struct TCtrLookupValues {
    TVector<i64> Keys;
    TVector<float> Values;
    float Missing;
};

static TCtrLookupValues GetCtrLookupValues(const TModelCtr& ctr, const TCtrValueTable& table) {

    TVector<std::pair<ui64, ui64>> entries;
    for (const auto& bucket : table.GetIndexHashViewer().GetBuckets()) {
        if (bucket.Hash != NCatboost::TBucket::InvalidHashValue) {
            // TBucket is packed; do not bind constructor arguments to its unaligned fields.
            const ui64 key = bucket.Hash;
            const ui64 index = bucket.IndexValue;
            entries.emplace_back(key, index);
        }
    }
    Sort(entries);
    size_t bucketCount;
    if (ctr.Base.CtrType == ECtrType::BinarizedTargetMeanValue || ctr.Base.CtrType == ECtrType::FloatTargetMeanValue) {
        bucketCount = table.GetTypedArrayRefForBlobData<TCtrMeanHistory>().size();
    } else {
        bucketCount = table.GetTypedArrayRefForBlobData<int>().size();
        if (ctr.Base.CtrType == ECtrType::Buckets || ctr.Base.CtrType == ECtrType::Borders) {
            CB_ENSURE(table.TargetClassesCount >= (ctr.Base.CtrType == ECtrType::Borders ? 2 : 1),
                "Invalid target classes count in CTR table");
            const int borderCount = table.TargetClassesCount - (ctr.Base.CtrType == ECtrType::Borders);
            CB_ENSURE(ctr.TargetBorderIdx >= 0 && ctr.TargetBorderIdx < borderCount, "Invalid CTR target border");
            CB_ENSURE(bucketCount % table.TargetClassesCount == 0, "Invalid CTR table size");
            bucketCount /= table.TargetClassesCount;
        }
    }
    // M^2 is odd, so its inverse exists modulo 2^64. Newton iteration doubles
    // the number of correct bits each step, starting with one correct bit.
    const ui64 factor = CalcHash(0, 1);
    ui64 inverse = 1;
    for (int i = 0; i < 6; ++i) {
        inverse *= 2 - factor * inverse;
    }
    TVector<i64> keys;
    TVector<ui64> buckets;
    keys.reserve(entries.size());
    buckets.reserve(entries.size() + 1);
    for (const auto& [key, bucket] : entries) {
        CB_ENSURE(bucket < bucketCount, "Invalid CTR table bucket index");
        const ui64 normalized = key * inverse;
        keys.push_back(normalized <= ui64(std::numeric_limits<i64>::max()) ? i64(normalized) : -1 - i64(~normalized));
        buckets.push_back(bucket);
    }
    buckets.push_back(NCatboost::TDenseIndexHashView::NotFoundIndex);
    TVector<float> values(buckets.size());
    CalcCtrValues(ctr, table, buckets, values);
    const float missing = values.back();
    values.pop_back();
    return {std::move(keys), std::move(values), missing};
}

static TString ExportCtrLookup(
    const TModelCtr& ctr,
    const TCtrValueTable& table,
    const TString& hash,
    TOnnxCtrGraphBuilder* builder) {

    const auto values = GetCtrLookupValues(ctr, table);
    return builder->Lookup(hash, values.Keys, values.Values, values.Missing);
}

static TOnnxFeaturesLayout InitHashedInputs(const TFullModel& model, onnx::GraphProto* graph) {
    const auto& trees = *model.ModelTrees;
    TOnnxFeaturesLayout layout;
    TVector<TString> columns;
    THashMap<int, TString> catInputs;
    THashSet<TString> names = {"num_features", "trees_input", "predictions", "label", "probabilities"};
    if (!trees.GetFloatFeatures().empty()) {
        InitValueInfo("num_features", onnx::TensorProto_DataType_FLOAT, trees.GetFloatFeatures().size(), graph->add_input());
        columns.push_back("num_features");
    }
    int nextColumn = 0;
    for (const auto& feature : trees.GetFloatFeatures()) {
        layout.FloatInputs.emplace(feature.Position.Index, TOnnxFeaturesLayout::TFloatInput{
            nextColumn++, feature.HasNans && feature.NanValueTreatment == ENanValueTreatment::AsTrue});
    }
    const bool needsBatchInput = trees.GetFloatFeatures().empty()
        && trees.GetOneHotFeatures().empty() && trees.GetCtrFeatures().empty();
    for (const auto& feature : trees.GetCatFeatures()) {
        // A constant categorical model still needs an input carrying the batch dimension.
        if (feature.UsedInModel() || (needsBatchInput && catInputs.empty())) {
            const auto name = GetCatFeatureInputName(feature);
            CB_ENSURE(names.insert(name).second, "Cannot export model to ONNX-ML: feature name \"" << name << "\" is not unique");
            catInputs.emplace(feature.Position.Index, name);
            auto* input = graph->add_input();
            InitValueInfo(name, onnx::TensorProto_DataType_INT64, 1, input);
            input->set_doc_string("Signed 32-bit CatBoost category hashes stored as int64; use catboost.utils.get_cat_feature_hashes.");
        }
    }
    TOnnxCtrGraphBuilder builder(graph);
    for (const auto& feature : trees.GetOneHotFeatures()) {
        CB_ENSURE(feature.Values.size() < (size_t(1) << std::numeric_limits<float>::digits), "Too many one-hot values for ONNX float encoding");
        TVector<i64> keys;
        TVector<float> values;
        auto& input = layout.CatInputs[feature.CatFeatureIndex];
        input.Index = nextColumn++;
        for (size_t i = 0; i < feature.Values.size(); ++i) {
            keys.push_back(feature.Values[i]);
            values.push_back(float(i));
            input.ValuesToIdx.emplace(feature.Values[i], i);
        }
        columns.push_back(builder.Lookup(catInputs.at(feature.CatFeatureIndex), keys, values, -1.f));
    }
    const auto* provider = dynamic_cast<const TStaticCtrProvider*>(model.CtrProvider.Get());
    CB_ENSURE(trees.GetCtrFeatures().empty() || provider, "ONNX CTR export requires a static CTR provider with saved tables");
    THashMap<TFeatureCombination, TString> projectionHashes;
    THashMap<TFloatSplit, TString> floatConditions;
    THashMap<TOneHotSplit, TString> oneHotConditions;
    for (const auto& feature : trees.GetCtrFeatures()) {
        const auto& ctr = feature.Ctr;
        const auto& projection = ctr.Base.Projection;
        auto* hash = projectionHashes.FindPtr(projection);
        if (!hash) {
            TVector<TString> terms;
            terms.reserve(projection.CatFeatures.size() + projection.BinFeatures.size() + projection.OneHotFeatures.size());
            for (int cat : projection.CatFeatures) {
                terms.push_back(catInputs.at(cat));
            }
            for (const auto& split : projection.BinFeatures) {
                if (!floatConditions.contains(split)) {
                    const auto& input = layout.FloatInputs.at(split.FloatFeature);
                    const auto column = builder.FloatColumn(input.Index);
                    auto condition = builder.Node("Greater", {column, builder.Float(split.Split)});
                    if (input.MissingValueTracksTrue && split.Split < std::numeric_limits<float>::infinity()) {
                        condition = builder.Node("Or", {condition, builder.Node("IsNaN", {column})});
                    }
                    floatConditions.emplace(split, builder.CastToInt(condition));
                }
                terms.push_back(floatConditions.at(split));
            }
            for (const auto& split : projection.OneHotFeatures) {
                if (!oneHotConditions.contains(split)) {
                    oneHotConditions.emplace(split, builder.CastToInt(
                        builder.Node("Equal", {catInputs.at(split.CatFeatureIdx), builder.Int(split.Value)})));
                }
                terms.push_back(oneHotConditions.at(split));
            }
            hash = &projectionHashes.emplace(projection, builder.Projection(terms)).first->second;
        }
        const auto* table = provider->CtrData.LearnCtrs.FindPtr(ctr.Base);
        CB_ENSURE(table, "ONNX CTR export requires all saved CTR tables");
        columns.push_back(ExportCtrLookup(ctr, *table, *hash, &builder));
        layout.CtrInputs.emplace(ctr, nextColumn++);
    }
    if (columns.empty() && !catInputs.empty()) {
        columns.push_back(builder.Lookup(catInputs.begin()->second, {}, {}, 0.f));
    }
    CB_ENSURE(!columns.empty(), "ONNX hashed categorical export requires at least one input feature");
    layout.TreesInputName = builder.Node("Concat", columns, "trees_input");
    AddAttribute("axis", i64(1), builder.LastNode());
    return layout;
}


static void AddTree(
    const TModelTrees& trees,
    i64 treeIdx,
    bool isClassifierModel,
    const TOnnxFeaturesLayout& layout,
    TTreesAttributes* treesAttributes) {

    i64 nodeIdx = 0;

    // Process splits
    for (auto depth : xrange(trees.GetModelTreeData()->GetTreeSizes()[treeIdx])) {
        const auto& split = trees.GetBinFeatures()[
            trees.GetModelTreeData()->GetTreeSplits()[trees.GetModelTreeData()->GetTreeStartOffsets()[treeIdx] + (trees.GetModelTreeData()->GetTreeSizes()[treeIdx] - 1 - depth)]];

        int splitFeatureIdx = 0;
        TString nodeMode;
        i64 missingValueTracksTrue = 0;
        float splitValue = 0.0f;

        if (split.Type == ESplitType::FloatFeature) {
            const auto& floatInput = layout.FloatInputs.at(split.FloatFeature.FloatFeature);
            splitFeatureIdx = floatInput.Index;
            nodeMode = TModeNode::BRANCH_GT;
            if (floatInput.MissingValueTracksTrue) {
                missingValueTracksTrue = 1;
            }
            splitValue = split.FloatFeature.Split;
        } else if (split.Type == ESplitType::OneHotFeature) {
            const auto& catInput = layout.CatInputs.at(split.OneHotFeature.CatFeatureIdx);
            splitFeatureIdx = catInput.Index;
            nodeMode = TModeNode::BRANCH_EQ;
            splitValue = (float)catInput.ValuesToIdx.at(split.OneHotFeature.Value);
        } else if (split.Type == ESplitType::OnlineCtr && layout.CtrInputs.contains(split.OnlineCtr.Ctr)) {
            splitFeatureIdx = layout.CtrInputs.at(split.OnlineCtr.Ctr);
            nodeMode = TModeNode::BRANCH_GT;
            splitValue = split.OnlineCtr.Border;
        } else {
            CB_ENSURE(
                false,
                "ONNX-ML format export supports only float features and one-hot encoded categorical features"
                " splits. Set one_hot_max_size parameter to a value greater than the maximum number of unique"
                " categorical feature values to make all categorical features one-hot encoded"
            );
        }

        i64 endNodeIdx = 2*nodeIdx + 1;
        for (; nodeIdx < endNodeIdx; ++nodeIdx) {
            treesAttributes->nodes_treeids->add_ints(treeIdx);
            treesAttributes->nodes_nodeids->add_ints(nodeIdx);

            treesAttributes->nodes_modes->add_strings(nodeMode);

            treesAttributes->nodes_featureids->add_ints((i64)splitFeatureIdx);
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


// With one categorical input and simple projections, all values absent from
// the saved tables take the same path. Compose the tables and forest at export
// time, keeping native prediction semantics (including collisions and bias).
static TMaybe<TString> TryExportSingleCategoryMargin(const TFullModel& model, onnx::GraphProto* graph) {
    const auto& trees = *model.ModelTrees;
    if (!trees.GetFloatFeatures().empty() || trees.GetCatFeatures().size() != 1) {
        return Nothing();
    }
    const auto& category = trees.GetCatFeatures().front();
    for (const auto& feature : trees.GetCtrFeatures()) {
        const auto& projection = feature.Ctr.Base.Projection;
        if (!projection.IsSingleCatFeature() || projection.CatFeatures.front() != category.Position.Index) {
            return Nothing();
        }
    }
    const auto* provider = dynamic_cast<const TStaticCtrProvider*>(model.CtrProvider.Get());
    CB_ENSURE(trees.GetCtrFeatures().empty() || provider, "ONNX CTR export requires a static CTR provider with saved tables");
    THashSet<i64> categories;
    for (const auto& feature : trees.GetCtrFeatures()) {
        const auto* table = provider->CtrData.LearnCtrs.FindPtr(feature.Ctr.Base);
        CB_ENSURE(table, "ONNX CTR export requires all saved CTR tables");
        // Validate the table before calling the native evaluator.
        for (i64 key : GetCtrLookupValues(feature.Ctr, *table).Keys) {
            if (key >= std::numeric_limits<i32>::min() && key <= std::numeric_limits<i32>::max()) {
                categories.insert(key);
            }
        }
    }
    for (const auto& feature : trees.GetOneHotFeatures()) {
        categories.insert(feature.Values.begin(), feature.Values.end());
    }
    TVector<i64> keys(categories.begin(), categories.end());
    Sort(keys);
    i64 unknown = std::numeric_limits<i32>::min();
    while (categories.contains(unknown)) {
        ++unknown;
    }
    CB_ENSURE(unknown <= std::numeric_limits<i32>::max(), "Too many categorical values for ONNX lookup export");
    keys.push_back(unknown);
    TVector<float> predictions;
    predictions.reserve(keys.size());
    auto evaluator = NCB::NModelEvaluation::CreateEvaluator(EFormulaEvaluatorType::CPU, model);
    evaluator->SetPredictionType(NCB::NModelEvaluation::EPredictionType::RawFormulaVal);
    const size_t catWidth = category.Position.Index + 1;
    const size_t blockSize = Max(size_t(1), size_t(65536) / catWidth);
    for (size_t start = 0; start < keys.size(); start += blockSize) {
        const size_t count = Min(blockSize, keys.size() - start);
        TVector<int> inputs(count * catWidth, 0);
        TVector<TConstArrayRef<int>> rows;
        rows.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            inputs[i * catWidth + category.Position.Index] = keys[start + i];
            rows.emplace_back(inputs.data() + i * catWidth, catWidth);
        }
        TVector<double> results(count);
        evaluator->Calc({}, rows, 0, model.GetTreeCount(), results, nullptr);
        for (double value : results) {
            predictions.push_back(value);
        }
    }
    const float missing = predictions.back();
    predictions.pop_back();
    keys.pop_back();
    const auto name = GetCatFeatureInputName(category);
    for (const TString reserved : {"num_features", "trees_input", "predictions", "label", "probabilities"}) {
        CB_ENSURE(name != reserved, "Cannot export model to ONNX-ML: feature name \"" << name << "\" is not unique");
    }
    auto* input = graph->add_input();
    InitValueInfo(name, onnx::TensorProto_DataType_INT64, 1, input);
    input->set_doc_string("Signed 32-bit CatBoost category hashes stored as int64; use catboost.utils.get_cat_feature_hashes.");
    TOnnxCtrGraphBuilder builder(graph);
    return builder.Lookup(name, keys, predictions, missing);
}

static TString AddIntVector(TConstArrayRef<i64> values, TOnnxCtrGraphBuilder* builder) {
    onnx::TensorProto tensor;
    tensor.set_data_type(onnx::TensorProto_DataType_INT64);
    tensor.add_dims(values.size());
    for (auto value : values) {
        tensor.add_int64_data(value);
    }
    return builder->Constant(tensor);
}

static TString AddFloatTensor(TConstArrayRef<float> values, TConstArrayRef<i64> shape, TOnnxCtrGraphBuilder* builder) {
    onnx::TensorProto tensor;
    tensor.set_data_type(onnx::TensorProto_DataType_FLOAT);
    for (auto dimension : shape) {
        tensor.add_dims(dimension);
    }
    for (auto value : values) {
        tensor.add_float_data(value);
    }
    return builder->Constant(tensor);
}

static TString AddBoolVector(TConstArrayRef<int> values, TOnnxCtrGraphBuilder* builder) {
    onnx::TensorProto tensor;
    tensor.set_data_type(onnx::TensorProto_DataType_BOOL);
    tensor.add_dims(values.size());
    for (auto value : values) {
        tensor.add_int32_data(value);
    }
    return builder->Constant(tensor);
}

// Every tree is oblivious: the truth values of its splits are the bits of its
// leaf index. Small blocks bound the dense matrices and temporary batch storage.
static TString ExportObliviousTreeMargin(
    const TFullModel& model,
    const TOnnxFeaturesLayout& layout,
    TOnnxCtrGraphBuilder* builder) {

    const auto& trees = *model.ModelTrees;
    const auto& data = trees.GetModelTreeData();
    const auto applyData = trees.GetApplyData();
    for (int depth : data->GetTreeSizes()) {
        if (depth > 24) {
            // Float indices cease to be exact beyond 24 bits. Retain the generic
            // tree operator for these exceptionally large trees.
            const auto result = builder->Node("TreeEnsembleRegressor", {layout.TreesInputName});
            auto* node = builder->LastNode();
            node->set_domain(onnx::AI_ONNX_ML_DOMAIN);
            AddAttribute("post_transform", "NONE", node);
            AddAttribute("n_targets", i64(1), node);
            const auto scaleAndBias = model.GetScaleAndBias();
            TTreesAttributes attributes(false, !scaleAndBias.IsZeroBias(), node->mutable_attribute());
            if (!scaleAndBias.IsZeroBias()) {
                attributes.base_values->add_floats(scaleAndBias.GetOneDimensionalBias());
            }
            for (auto tree : xrange(trees.GetTreeCount())) {
                AddTree(trees, tree, false, layout, &attributes);
            }
            return result;
        }
    }
    TVector<TVector<size_t>> treesByDepth(25);
    double bias = model.GetScaleAndBias().GetOneDimensionalBias();
    for (size_t tree = 0; tree < trees.GetTreeCount(); ++tree) {
        const int depth = data->GetTreeSizes()[tree];
        CB_ENSURE(depth <= 24, "Tensor ONNX export supports tree depth up to 24");
        if (depth == 0) {
            bias += data->GetLeafValues()[applyData->TreeFirstLeafOffsets[tree]];
        } else {
            treesByDepth[depth].push_back(tree);
        }
    }
    TVector<TString> contributions;
    for (size_t depth = 1; depth < treesByDepth.size(); ++depth) {
        const auto& group = treesByDepth[depth];
        const size_t leavesPerTree = size_t(1) << depth;
        const size_t blockSize = Min(size_t(32), Max(size_t(1), (size_t(1) << 20) / leavesPerTree));
        for (size_t start = 0; start < group.size(); start += blockSize) {
            const size_t treeCount = Min(blockSize, group.size() - start);
            THashMap<int, size_t> splitColumns;
            TVector<int> splits;
            for (size_t t = 0; t < treeCount; ++t) {
                const auto offset = data->GetTreeStartOffsets()[group[start + t]];
                for (size_t d = 0; d < depth; ++d) {
                    const int split = data->GetTreeSplits()[offset + d];
                    if (splitColumns.emplace(split, splits.size()).second) {
                        splits.push_back(split);
                    }
                }
            }
            TVector<i64> features;
            TVector<float> thresholds;
            TVector<int> equal, missingTrue;
            features.reserve(splits.size());
            thresholds.reserve(splits.size());
            equal.reserve(splits.size());
            missingTrue.reserve(splits.size());
            bool hasEqual = false;
            bool hasMissingTrue = false;
            for (int index : splits) {
                const auto& split = trees.GetBinFeatures()[index];
                int feature;
                float threshold;
                bool isEqual = false;
                bool missing = false;
                if (split.Type == ESplitType::FloatFeature) {
                    const auto& input = layout.FloatInputs.at(split.FloatFeature.FloatFeature);
                    feature = input.Index;
                    threshold = split.FloatFeature.Split;
                    missing = input.MissingValueTracksTrue && threshold < std::numeric_limits<float>::infinity();
                } else if (split.Type == ESplitType::OneHotFeature) {
                    const auto& input = layout.CatInputs.at(split.OneHotFeature.CatFeatureIdx);
                    feature = input.Index;
                    threshold = input.ValuesToIdx.at(split.OneHotFeature.Value);
                    isEqual = true;
                } else {
                    CB_ENSURE(split.Type == ESplitType::OnlineCtr, "Unsupported split in ONNX CTR export");
                    feature = layout.CtrInputs.at(split.OnlineCtr.Ctr);
                    threshold = split.OnlineCtr.Border;
                }
                features.push_back(feature);
                thresholds.push_back(threshold);
                equal.push_back(isEqual);
                missingTrue.push_back(missing);
                hasEqual |= isEqual;
                hasMissingTrue |= missing;
            }
            const auto values = builder->Node("ArrayFeatureExtractor", {layout.TreesInputName, AddIntVector(features, builder)});
            builder->LastNode()->set_domain(onnx::AI_ONNX_ML_DOMAIN);
            const auto borders = AddFloatTensor(thresholds, {i64(splits.size())}, builder);
            auto decisions = builder->Node("Greater", {values, borders});
            if (hasEqual) {
                const auto mask = AddBoolVector(equal, builder);
                decisions = builder->Node("Or", {
                    builder->Node("And", {mask, builder->Node("Equal", {values, borders})}),
                    builder->Node("And", {builder->Node("Not", {mask}), decisions})});
            }
            if (hasMissingTrue) {
                decisions = builder->Node("Or", {decisions, builder->Node("And", {
                    builder->Node("IsNaN", {values}), AddBoolVector(missingTrue, builder)})});
            }
            auto bits = builder->Node("Cast", {decisions});
            AddAttribute("to", i64(onnx::TensorProto_DataType_FLOAT), builder->LastNode());
            bits = builder->Node("Transpose", {bits});
            AddAttribute("perm", TVector<i64>{1, 0}, builder->LastNode());
            TString indices;
            // Ten-bit chunks keep even coefficients for repeated splits exact in
            // TF32 matmul. All partial sums and final indices are exact float32 integers.
            for (size_t bitStart = 0; bitStart < depth; bitStart += 10) {
                TVector<float> weights(treeCount * splits.size(), 0.f);
                for (size_t t = 0; t < treeCount; ++t) {
                    const auto offset = data->GetTreeStartOffsets()[group[start + t]];
                    for (size_t d = bitStart; d < Min(bitStart + 10, depth); ++d) {
                        const int split = data->GetTreeSplits()[offset + d];
                        weights[t * splits.size() + splitColumns.at(split)] += size_t(1) << (d - bitStart);
                    }
                }
                auto part = builder->Node("MatMul", {
                    AddFloatTensor(weights, {i64(treeCount), i64(splits.size())}, builder), bits});
                if (bitStart) {
                    part = builder->Node("Mul", {part, builder->Float(size_t(1) << bitStart)});
                }
                indices = indices.empty() ? part : builder->Node("Add", {indices, part});
            }
            indices = builder->Node("Cast", {indices});
            AddAttribute("to", i64(onnx::TensorProto_DataType_INT32), builder->LastNode());
            TVector<float> leaves;
            leaves.reserve(treeCount * leavesPerTree);
            for (size_t t = 0; t < treeCount; ++t) {
                const auto offset = applyData->TreeFirstLeafOffsets[group[start + t]];
                for (size_t i = 0; i < leavesPerTree; ++i) {
                    leaves.push_back(data->GetLeafValues()[offset + i]);
                }
            }
            const auto predictions = builder->Node("GatherElements", {
                AddFloatTensor(leaves, {i64(treeCount), i64(leavesPerTree)}, builder), indices});
            AddAttribute("axis", i64(1), builder->LastNode());
            auto sum = builder->Node("ReduceSum", {predictions});
            AddAttribute("axes", TVector<i64>{0}, builder->LastNode());
            AddAttribute("keepdims", i64(0), builder->LastNode());
            sum = builder->Node("Unsqueeze", {sum});
            AddAttribute("axes", TVector<i64>{1}, builder->LastNode());
            contributions.push_back(sum);
        }
    }
    if (contributions.empty()) {
        const auto column = builder->Node("ArrayFeatureExtractor", {layout.TreesInputName, AddIntVector({0}, builder)});
        builder->LastNode()->set_domain(onnx::AI_ONNX_ML_DOMAIN);
        return builder->Lookup(column, {}, {}, bias);
    }
    auto result = contributions.front();
    // Binary Add also handles empty batches on CUDA; the variadic Sum kernel
    // in ORT 1.24 does not support that case for three or more inputs.
    for (size_t i = 1; i < contributions.size(); ++i) {
        result = builder->Node("Add", {result, contributions[i]});
    }
    if (bias != 0.) {
        result = builder->Node("Add", {result, builder->Float(bias)});
    }
    return result;
}

static void ConvertHashedCatTreeToOnnxGraph(const TFullModel& model, onnx::GraphProto* graph) {
    const auto& trees = *model.ModelTrees;
    const TString loss = model.GetLossFunctionName();
    CB_ENSURE(trees.GetDimensionsCount() == 1 && (loss.empty() || !IsMultiTargetObjective(loss)),
        "ONNX hashed categorical export supports only binary classification and scalar regression");
    CB_ENSURE(trees.IsOblivious(), "ONNX export requires symmetric trees");
    const bool classifier = IsClassifierModel(model);
    TVector<i64> labelsInt;
    TVector<TString> labelsString;
    if (classifier) {
        GetClassLabels(model, &labelsInt, &labelsString);
        CB_ENSURE(labelsInt.size() + labelsString.size() == 2, "ONNX hashed categorical export requires exactly two class labels");
    }
    auto singleCategoryMargin = TryExportSingleCategoryMargin(model, graph);
    TOnnxFeaturesLayout layout;
    if (!singleCategoryMargin) {
        layout = InitHashedInputs(model, graph);
    }
    TOnnxCtrGraphBuilder builder(graph);
    const auto margin = singleCategoryMargin ? *singleCategoryMargin : ExportObliviousTreeMargin(model, layout, &builder);
    if (!classifier) {
        builder.Node("Identity", {margin}, "predictions");
        InitValueInfo("predictions", onnx::TensorProto_DataType_FLOAT, 1, graph->add_output());
        return;
    }
    const auto probability = builder.Node("Sigmoid", {margin});
    const auto probabilities = builder.Node("Concat", {builder.Node("Sub", {builder.Float(1), probability}), probability});
    AddAttribute("axis", i64(1), builder.LastNode());
    builder.Node("ZipMap", {probabilities}, "probabilities");
    builder.LastNode()->set_domain(onnx::AI_ONNX_ML_DOMAIN);
    AddClassLabelsAttribute(labelsInt, labelsString, builder.LastNode());
    const auto labelType = labelsString.empty() ? onnx::TensorProto_DataType_INT64 : onnx::TensorProto_DataType_STRING;
    const auto positive = builder.CastToInt(builder.Node("Greater", {margin, builder.Float(model.GetBinClassLogitThreshold())}));
    const auto indices = builder.Node("Squeeze", {positive});
    AddAttribute("axes", TVector<i64>{1}, builder.LastNode());
    onnx::TensorProto labels;
    labels.set_data_type(labelType);
    labels.add_dims(2);
    for (auto value : labelsInt) {
        labels.add_int64_data(value);
    }
    for (const auto& value : labelsString) {
        labels.add_string_data(value);
    }
    builder.Node("Gather", {builder.Constant(labels), indices}, "label");
    InitValueInfo("label", labelType, Nothing(), graph->add_output());
    InitProbabilitiesOutput("probabilities", labelType, graph->add_output());
}


void NCB::NOnnx::ConvertTreeToOnnxGraph(
    const TFullModel& model,
    const TMaybe<TString>& onnxGraphName,
    onnx::GraphProto* onnxGraph,
    const THashMap<ui32, TString>* catFeaturesHashToString,
    bool useHashedCatFeatures) {

    onnxGraph->set_name(onnxGraphName.GetOrElse("CatBoostModel"));
    if (useHashedCatFeatures) {
        ConvertHashedCatTreeToOnnxGraph(model, onnxGraph);
        return;
    }
    CB_ENSURE(model.ModelTrees->GetCtrFeatures().empty(),
        "ONNX export of CTR models requires onnx_cat_features_input_type='hashed' and prehashed categorical inputs");
    const bool explicitBinaryOutputs = HasNonDefaultBinaryThreshold(model);
    // A custom binary threshold uses a scalar margin followed by explicit classification.
    const bool isClassifierModel = IsClassifierModel(model) && !explicitBinaryOutputs;
    TString margin;

    const TModelTrees& trees = *model.ModelTrees;

    onnxGraph->set_name(onnxGraphName.GetOrElse("CatBoostModel"));

    const TOnnxFeaturesLayout layout = InitInputs(trees, catFeaturesHashToString, onnxGraph);

    onnx::NodeProto* treesNode = onnxGraph->add_node();
    treesNode->set_domain(onnx::AI_ONNX_ML_DOMAIN);

    treesNode->add_input(layout.TreesInputName);
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

        if (explicitBinaryOutputs) {
            margin = TOnnxGraphBuilder(onnxGraph).NewName();
            treesNode->add_output(margin);
        } else {
            InitValueInfo(
                "predictions",
                onnx::TensorProto_DataType_FLOAT,
                /*secondDim*/ Nothing(),
                onnxGraph->add_output()
            );
            treesNode->add_output("predictions");
        }
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
            "Inappropriate dimension of bias, should be " << trees.GetDimensionsCount() << " or 0, found " << biasSize);
            for (auto b : bias) {
                treesAttributes.base_values->add_floats(b);
            }
        }
    }
    for (auto treeIdx : xrange(trees.GetTreeCount())) {
        AddTree(trees, treeIdx, isClassifierModel, layout, &treesAttributes);
    }
    if (explicitBinaryOutputs) {
        AddBinaryClassifierOutputs(model, margin, onnxGraph);
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

    fullModel->ModelInfo = std::move(modelInfo);
}


struct TOnnxCatFeatureInputInfo {
    TString Name;
    THashMap<int, TString> EncodedValueToString; // from LabelEncoder attributes
};

// Description of the TreeEnsemble* node input tensor
struct TOnnxTreesInputInfo {
    // number of columns in the tree ensemble input
    int FeaturesCount = 0;

    // column index -> info about categorical feature preprocessed with LabelEncoder
    THashMap<int, TOnnxCatFeatureInputInfo> CatFeatureInputs;
};


static int GetSecondDim(const onnx::ValueInfoProto& valueInfo) {
    const auto& shape = valueInfo.type().tensor_type().shape();
    CB_ENSURE(
        shape.dim_size() == 2,
        "Dimension of input \"" << valueInfo.name() << "\" must have format [N, featuresCount]"
    );
    CB_ENSURE(shape.dim(1).has_dim_value(), "Second dimension of input \"" << valueInfo.name() << "\" is unknown");
    CB_ENSURE(
        shape.dim(1).dim_value() >= 0 && shape.dim(1).dim_value() <= Max<int>(),
        "Invalid feature count for input \"" << valueInfo.name() << "\""
    );
    return (int)shape.dim(1).dim_value();
}


static const onnx::NodeProto* FindTreesNode(const onnx::GraphProto& onnxGraph) {
    for (const auto& node : onnxGraph.node()) {
        if ((node.op_type() == "TreeEnsembleClassifier") || (node.op_type() == "TreeEnsembleRegressor")) {
            return &node;
        }
    }
    ythrow TCatBoostException() << "TreeEnsembleClassifier or TreeEnsembleRegressor node not found in ONNX graph";
}


static int GetIntegerCatFeatureValue(float value) {
    CB_ENSURE(
        std::isfinite(value) && value == std::floor(value) &&
        (double)value >= Min<int>() && (double)value <= Max<int>(),
        "Categorical feature value must be an integer in the int32 range, got " << value
    );
    return (int)value;
}


static THashMap<int, TString> GetLabelEncoderMapping(const onnx::NodeProto& labelEncoderNode) {
    const onnx::AttributeProto* keysStrings = nullptr;
    const onnx::AttributeProto* valuesFloats = nullptr;
    float defaultValue = -0.0f; // LabelEncoder's default_float default in ai.onnx.ml opset 2
    for (const auto& attribute : labelEncoderNode.attribute()) {
        if (attribute.name() == "keys_strings") {
            keysStrings = &attribute;
        } else if (attribute.name() == "values_floats") {
            valuesFloats = &attribute;
        } else if (attribute.name() == "default_float") {
            defaultValue = attribute.f();
        }
    }
    CB_ENSURE(
        keysStrings && valuesFloats,
        "Only LabelEncoder nodes with keys_strings and values_floats attributes are supported"
    );
    CB_ENSURE(
        keysStrings->strings_size() == valuesFloats->floats_size(),
        "LabelEncoder keys_strings and values_floats must have the same size"
    );
    CB_ENSURE(!std::isnan(defaultValue), "LabelEncoder default_float must not be NaN");

    THashMap<int, TString> result;
    THashSet<TString> keys;
    for (auto i : xrange(keysStrings->strings_size())) {
        const int value = GetIntegerCatFeatureValue(valuesFloats->floats(i));
        CB_ENSURE(
            valuesFloats->floats(i) != defaultValue,
            "LabelEncoder default_float must differ from all encoded values"
        );
        CB_ENSURE(keys.insert(keysStrings->strings(i)).second, "LabelEncoder keys_strings must be unique");
        CB_ENSURE(
            result.emplace(value, keysStrings->strings(i)).second,
            "LabelEncoder values_floats must be unique"
        );
    }
    return result;
}


/*
 * Supported inputs of the TreeEnsemble* node:
 *  - a graph input (float tensor with all features);
 *  - an output of a Concat node whose inputs are graph inputs or outputs of LabelEncoder nodes applied
 *    to graph inputs (this is how CatBoost exports models with categorical features);
 */
static TOnnxTreesInputInfo GetTreesInputInfo(const onnx::GraphProto& onnxGraph, const TString& treesInputName) {
    THashMap<TString, const onnx::ValueInfoProto*> graphInputs;
    for (const auto& input : onnxGraph.input()) {
        graphInputs[input.name()] = &input;
    }
    THashMap<TString, const onnx::NodeProto*> outputToNode;
    for (const auto& node : onnxGraph.node()) {
        for (const auto& output : node.output()) {
            outputToNode[output] = &node;
        }
    }

    TOnnxTreesInputInfo result;

    if (const auto* graphInput = graphInputs.Value(treesInputName, nullptr)) {
        result.FeaturesCount = GetSecondDim(*graphInput);
        return result;
    }

    const onnx::NodeProto* concatNode = outputToNode.Value(treesInputName, nullptr);
    CB_ENSURE(
        concatNode && concatNode->op_type() == "Concat" && concatNode->domain().empty(),
        "Unsupported preprocessing of TreeEnsemble input \"" << treesInputName << "\""
    );

    bool concatenatesFeatures = false;
    for (const auto& attribute : concatNode->attribute()) {
        if (attribute.name() == "axis") {
            concatenatesFeatures = attribute.i() == 1 || attribute.i() == -1;
        }
    }
    CB_ENSURE(concatenatesFeatures, "Concat must concatenate input columns (axis 1 or -1)");

    for (const auto& concatInputName : concatNode->input()) {
        if (const auto* graphInput = graphInputs.Value(concatInputName, nullptr)) {
            const int featureCount = GetSecondDim(*graphInput);
            CB_ENSURE(
                featureCount <= Max<int>() - result.FeaturesCount,
                "Too many features in Concat inputs"
            );
            result.FeaturesCount += featureCount;
            continue;
        }

        const onnx::NodeProto* labelEncoderNode = outputToNode.Value(concatInputName, nullptr);
        CB_ENSURE(
            labelEncoderNode && (labelEncoderNode->op_type() == "LabelEncoder"),
            "Inputs of Concat node must be either graph inputs or outputs of LabelEncoder nodes"
        );
        CB_ENSURE(labelEncoderNode->input_size() == 1, "LabelEncoder node must have a single input");
        const auto* labelEncoderInput = graphInputs.Value(labelEncoderNode->input(0), nullptr);
        CB_ENSURE(labelEncoderInput, "Input of LabelEncoder node must be a graph input");
        CB_ENSURE(
            GetSecondDim(*labelEncoderInput) == 1,
            "Input of LabelEncoder node must have a single column"
        );

        auto& catFeatureInput = result.CatFeatureInputs[result.FeaturesCount];
        catFeatureInput.Name = labelEncoderInput->name();
        catFeatureInput.EncodedValueToString = GetLabelEncoderMapping(*labelEncoderNode);
        CB_ENSURE(result.FeaturesCount < Max<int>(), "Too many features in Concat inputs");
        ++result.FeaturesCount;
    }

    return result;
}


static bool IsCatFeatureNodeMode(const TString& nodeMode) {
    return (nodeMode == TModeNode::BRANCH_EQ) || (nodeMode == TModeNode::BRANCH_NEQ);
}

static bool IsFloatFeatureNodeMode(const TString& nodeMode) {
    return (nodeMode == TModeNode::BRANCH_LEQ) || (nodeMode == TModeNode::BRANCH_LT) ||
        (nodeMode == TModeNode::BRANCH_GTE) || (nodeMode == TModeNode::BRANCH_GT);
}


/*
 * Features preprocessed with LabelEncoder are categorical and may only be used in BRANCH_EQ/BRANCH_NEQ
 * nodes. Other features are numerical and may only be used in ordered comparisons.
 * All columns of the input tensor are added to the features (even if unused in the model) to preserve
 * the input layout.
 */
static void PrepareFeatures(
    const TTreesAttributes& treesAttributes,
    const TOnnxTreesInputInfo& treesInputInfo,
    TVector<TFloatFeature>* floatFeatures,
    TVector<TCatFeature>* catFeatures,
    TVector<int>* flatFeatureIndexToPerTypeIndex
) {
    const int featuresCount = treesInputInfo.FeaturesCount;
    THashSet<int> catFeatureFlatIndices;
    THashSet<int> floatFeatureFlatIndices;
    for (const auto& catFeatureInput : treesInputInfo.CatFeatureInputs) {
        catFeatureFlatIndices.insert(catFeatureInput.first);
    }

    for (auto idx : xrange(treesAttributes.nodes_modes->strings_size())) {
        const TString& nodeMode = treesAttributes.nodes_modes->strings(idx);
        if (nodeMode == TModeNode::LEAF) {
            continue;
        }
        const auto flatFeatureIdx = treesAttributes.nodes_featureids->ints(idx);
        CB_ENSURE(
            flatFeatureIdx >= 0 && flatFeatureIdx < featuresCount,
            "ONNX node feature index " << flatFeatureIdx << " is outside the input tensor"
        );
        if (IsCatFeatureNodeMode(nodeMode)) {
            CB_ENSURE(
                !floatFeatureFlatIndices.contains(flatFeatureIdx),
                "ONNX feature # " << flatFeatureIdx << " cannot be used in both categorical and numerical splits"
            );
            catFeatureFlatIndices.insert(flatFeatureIdx);
        } else {
            CB_ENSURE(IsFloatFeatureNodeMode(nodeMode), "Undefined mode of node " << nodeMode);
            CB_ENSURE(
                !catFeatureFlatIndices.contains(flatFeatureIdx),
                "ONNX feature # " << flatFeatureIdx << " cannot be used in both categorical and numerical splits"
            );
            floatFeatureFlatIndices.insert(flatFeatureIdx);
        }
    }

    floatFeatures->clear();
    catFeatures->clear();
    flatFeatureIndexToPerTypeIndex->assign(featuresCount, -1);

    for (auto flatFeatureIdx : xrange(featuresCount)) {
        if (catFeatureFlatIndices.contains(flatFeatureIdx)) {
            TCatFeature catFeature;
            catFeature.Position = TFeaturePosition((int)catFeatures->size(), flatFeatureIdx);
            CB_ENSURE(
                treesInputInfo.CatFeatureInputs.contains(flatFeatureIdx),
                "Categorical feature # " << flatFeatureIdx << " must have a LabelEncoder input"
            );
            catFeature.FeatureId = treesInputInfo.CatFeatureInputs.at(flatFeatureIdx).Name;
            (*flatFeatureIndexToPerTypeIndex)[flatFeatureIdx] = catFeature.Position.Index;
            catFeatures->push_back(std::move(catFeature));
        } else {
            TFloatFeature floatFeature;
            floatFeature.Position = TFeaturePosition((int)floatFeatures->size(), flatFeatureIdx);
            (*flatFeatureIndexToPerTypeIndex)[flatFeatureIdx] = floatFeature.Position.Index;
            floatFeatures->push_back(std::move(floatFeature));
        }
    }
}


static int GetCatFeatureValue(
    const TOnnxTreesInputInfo& treesInputInfo,
    int flatFeatureIdx,
    float onnxValue
) {
    const int encodedValue = GetIntegerCatFeatureValue(onnxValue);

    const auto& catFeatureInput = treesInputInfo.CatFeatureInputs.at(flatFeatureIdx);
    const TString* mappedValue = catFeatureInput.EncodedValueToString.FindPtr(encodedValue);
    CB_ENSURE(
        mappedValue,
        "Categorical feature " << flatFeatureIdx << " split value " << encodedValue
        << " is not found in LabelEncoder mapping"
    );
    return CalcCatFeatureHash(*mappedValue);
}


static void PrepareTrees(
    const TTreesAttributes& treesAttributes,
    const bool isClassifierModel,
    const TOnnxTreesInputInfo& treesInputInfo,
    TVector<THashMap<int, NCB::NOnnx::TOnnxNode>>* trees,
    int* approxDimension,
    TVector<TFloatFeature>* floatFeatures, /* for adding nanModes and borders */
    TVector<TCatFeature>* catFeatures
) {
    TVector<int> flatFeatureIndexToPerTypeIndex;
    PrepareFeatures(treesAttributes, treesInputInfo, floatFeatures, catFeatures, &flatFeatureIndexToPerTypeIndex);

    TVector<TSet<float>> floatFeatureBorders(floatFeatures->size());
    //consider all nodes
    for (auto idx = 0; idx < treesAttributes.nodes_treeids->ints_size() ;++idx) {
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

            const int flatFeatureIdx = treesAttributes.nodes_featureids->ints(idx);
            const int perTypeFeatureIdx = flatFeatureIndexToPerTypeIndex.at(flatFeatureIdx);

            TModelSplit split;
            if (IsCatFeatureNodeMode(nodeMode)) {
                if (nodeMode == TModeNode::BRANCH_NEQ) {
                    std::swap(node.TrueNodeId, node.FalseNodeId);
                }

                split.Type = ESplitType::OneHotFeature;
                split.OneHotFeature.CatFeatureIdx = perTypeFeatureIdx;
                split.OneHotFeature.Value = GetCatFeatureValue(
                    treesInputInfo,
                    flatFeatureIdx,
                    treesAttributes.nodes_values->floats(idx)
                );
            } else {
                if (nodeMode == TModeNode::BRANCH_LEQ || nodeMode == TModeNode::BRANCH_LT) {
                    std::swap(node.TrueNodeId, node.FalseNodeId);
                } else {
                    CB_ENSURE(nodeMode == TModeNode::BRANCH_GTE || nodeMode == TModeNode::BRANCH_GT, "Undefined mode of node " << nodeMode);
                }

                split.Type = ESplitType::FloatFeature;
                split.FloatFeature.FloatFeature = perTypeFeatureIdx;
                split.FloatFeature.Split = treesAttributes.nodes_values->floats(idx);

                //update floatFeatures
                if (treesAttributes.nodes_missing_value_tracks_true->ints(idx) == 1) {
                    (*floatFeatures)[split.FloatFeature.FloatFeature].NanValueTreatment =
                    ENanValueTreatment::AsTrue;
                }
                floatFeatureBorders[split.FloatFeature.FloatFeature].insert(split.FloatFeature.Split);
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
    const onnx::NodeProto* treesNode = FindTreesNode(onnxGraph);
    const bool isClassifierModel = (treesNode->op_type() == "TreeEnsembleClassifier");

    auto attributes = treesNode->attribute();
    TTreesAttributes treesAttributes(isClassifierModel, attributes);

    CB_ENSURE(treesNode->input_size() == 1, "TreeEnsemble node must have a single input");
    const TOnnxTreesInputInfo treesInputInfo = GetTreesInputInfo(onnxGraph, treesNode->input(0));

    TVector<TFloatFeature> floatFeatures;
    TVector<TCatFeature> catFeatures;
    TVector<THashMap<int, NCB::NOnnx::TOnnxNode>> trees;
    int approxDimension = 1;
    PrepareTrees(
        treesAttributes,
        isClassifierModel,
        treesInputInfo,
        &trees,
        &approxDimension,
        &floatFeatures,
        &catFeatures
    );

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

    const onnx::GraphProto& onnxGraph = onnxModel.graph();
    ConfigureSymmetricTrees(onnxGraph, fullModel);
}
