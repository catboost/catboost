#include "pmml_helpers.h"

#include <catboost/libs/model/model.h>

#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/xml_output.h>
#include <catboost/private/libs/options/enum_helpers.h>

#include <library/cpp/json/json_value.h>
#include <library/cpp/svnversion/svnversion.h>

#include <util/datetime/base.h>
#include <util/generic/algorithm.h>
#include <util/generic/cast.h>
#include <util/generic/vector.h>
#include <util/generic/xrange.h>
#include <util/stream/file.h>
#include <util/string/builder.h>
#include <util/system/types.h>

#include <algorithm>

static void OutputHeader(
    const TFullModel& model,
    const NJson::TJsonValue& userParameters,
    TXmlOutputContext* xmlOut) {

    TXmlElementOutputContext header(xmlOut, "Header");
    if (userParameters.Has("pmml_copyright")) {
        xmlOut->AddAttr("copyright", userParameters["pmml_copyright"].GetStringSafe());
    }
    if (userParameters.Has("pmml_description")) {
        xmlOut->AddAttr("description", userParameters["pmml_description"].GetStringSafe());
    }
    if (userParameters.Has("pmml_model_version")) {
        xmlOut->AddAttr("modelVersion", userParameters["pmml_model_version"].GetStringSafe());
    }
    {
        TXmlElementOutputContext application(xmlOut, "Application");
        xmlOut->AddAttr("name", "CatBoost");
        if (const auto* catBoostVersionInfo = model.ModelInfo.FindPtr("catboost_version_info")) {
            xmlOut->AddAttr("version", *catBoostVersionInfo);
        } else {
            xmlOut->AddAttr("version", PROGRAM_VERSION);
        }
    }
    {
        TXmlElementOutputContext timestamp(xmlOut, "Timestamp");
        if (const auto* trainFinishTime = model.ModelInfo.FindPtr("train_finish_time")) {
            xmlOut->GetOutput() << *trainFinishTime;
        } else {
            xmlOut->GetOutput() << TInstant::Now().ToStringUpToSeconds();
        }
    }
}


template <class TFeature>
static TString CreateFeatureName(const TFeature& feature) {
    if (feature.FeatureId) {
        return feature.FeatureId;
    }
    TStringBuilder result;
    result << "feature_" << feature.Position.FlatIndex;
    return result;
}


static void OutputDataDictionary(
    const TFullModel& model,
    bool isClassification,
    TXmlOutputContext* xmlOut) {

    TXmlElementOutputContext dataDictionary(xmlOut, "DataDictionary");

    for (const auto& floatFeature : model.ModelTrees->GetFloatFeatures()) {
        TXmlElementOutputContext dataField(xmlOut, "DataField");
        xmlOut->AddAttr("name", CreateFeatureName(floatFeature))
            .AddAttr("optype", "continuous")
            .AddAttr("dataType", "float");
    }

    for (const auto& catFeature : model.ModelTrees->GetCatFeatures()) {
        {
            TXmlElementOutputContext dataField(xmlOut, "DataField");
            xmlOut->AddAttr("name", CreateFeatureName(catFeature))
                .AddAttr("optype", "categorical")
                .AddAttr("dataType", "string");
        }
    }

    // output
    {
        TXmlElementOutputContext dataField(xmlOut, "DataField");
        xmlOut->AddAttr("name", "prediction");
        if (isClassification) {
            xmlOut->AddAttr("optype", "categorical").AddAttr("dataType", "boolean");
        } else {
            xmlOut->AddAttr("optype", "continuous").AddAttr("dataType", "double");
        }
    }
    if (isClassification) {
        TXmlElementOutputContext dataField(xmlOut, "DataField");
        xmlOut->AddAttr("name", "approx").AddAttr("optype", "continuous").AddAttr("dataType", "double");
    }
}

static void OutputTargetsFields(
    const TFullModel& model,
    TXmlOutputContext* xmlOut) {

    CB_ENSURE(
        model.GetDimensionsCount() == 1,
        "PMML export currently supports only single-dimensional models");


    TXmlElementOutputContext targets(xmlOut, "Targets");
    {
        TXmlElementOutputContext target(xmlOut, "Target");
        xmlOut->AddAttr("rescaleConstant", model.GetScaleAndBias().GetOneDimensionalBias())
            .AddAttr("rescaleFactor", model.GetScaleAndBias().Scale)
            .AddAttr("field", "prediction");
    }


}

static void OutputMiningSchemaWithModelFeatures(
    const TFullModel& model,
    bool mappedCategoricalFeatures,
    TMaybe<TStringBuf> targetName, // if nothing - don't output it
    TXmlOutputContext* xmlOut) {

    TXmlElementOutputContext miningSchema(xmlOut, "MiningSchema");

    for (const auto& floatFeature : model.ModelTrees->GetFloatFeatures()) {
        TXmlElementOutputContext miningField(xmlOut, "MiningField");
        xmlOut->AddAttr("name", CreateFeatureName(floatFeature)).AddAttr("usageType", "active");
    }

    const auto& modelTrees = model.ModelTrees;

    if (mappedCategoricalFeatures) {
        for (const auto& oneHotFeature : modelTrees->GetOneHotFeatures()) {
            TXmlElementOutputContext miningField(xmlOut, "MiningField");
            xmlOut->AddAttr("name", CreateFeatureName(modelTrees->GetCatFeatures()[oneHotFeature.CatFeatureIndex]) + "_mapped")
                .AddAttr("usageType", "active");
        }
    } else {
        for (const auto& catFeature : modelTrees->GetCatFeatures()) {
            TXmlElementOutputContext miningField(xmlOut, "MiningField");
            xmlOut->AddAttr("name", CreateFeatureName(catFeature))
                .AddAttr("usageType", "active");
        }
    }

    if (targetName) {
        TXmlElementOutputContext miningField(xmlOut, "MiningField");
        xmlOut->AddAttr("name", *targetName).AddAttr("usageType", "target");
    }
}


using TOneHotValuesToIdx = TVector<THashMap<int, ui32>>; // [catFeatureIdx][oneHotValue] -> oneHotIndex


static void OutputCategoricalMapping(
    const TFullModel& model,
    const THashMap<ui32, TString>& catFeaturesHashToString,
    TOneHotValuesToIdx* oneHotValuesToIdx,
    TXmlOutputContext* xmlOut) {

    const auto& modelTrees = model.ModelTrees;

    oneHotValuesToIdx->clear();
    oneHotValuesToIdx->resize(model.ModelTrees->GetCatFeatures().size());

    TXmlElementOutputContext localTransformations(xmlOut, "LocalTransformations");

    for (const auto& oneHotFeature : modelTrees->GetOneHotFeatures()) {
        const auto featureName = CreateFeatureName(modelTrees->GetCatFeatures()[oneHotFeature.CatFeatureIndex]);
        auto& oneHotValuesToIdxMap = (*oneHotValuesToIdx)[oneHotFeature.CatFeatureIndex];

        TString mappedFeatureName = featureName + "_mapped";

        TXmlElementOutputContext derivedField(xmlOut, "DerivedField");
        xmlOut->AddAttr("name", mappedFeatureName)
            .AddAttr("optype", "categorical")
            .AddAttr("dataType", "integer");

        {
            TXmlElementOutputContext mapValues(xmlOut, "MapValues");

            const auto mapMissingTo = oneHotFeature.Values.size();
            xmlOut->AddAttr("mapMissingTo", mapMissingTo)
                .AddAttr("defaultValue", mapMissingTo)
                .AddAttr("outputColumn", "value")
                .AddAttr("dataType", "integer");

            {
                TXmlElementOutputContext fieldColumnPair(xmlOut, "FieldColumnPair");
                xmlOut->AddAttr("field", featureName).AddAttr("column", "key");
            }
            {
                TXmlElementOutputContext inlineTable(xmlOut, "InlineTable");

                for (auto i : xrange(oneHotFeature.Values.size())) {
                    oneHotValuesToIdxMap.emplace(oneHotFeature.Values[i], (ui32)i);

                    TXmlElementOutputContext row(xmlOut, "row");
                    {
                        TXmlElementOutputContext key(xmlOut, "key");
                        xmlOut->GetOutput() << catFeaturesHashToString.at((ui32)oneHotFeature.Values[i]);
                    }
                    {
                        TXmlElementOutputContext value(xmlOut, "value");
                        xmlOut->GetOutput() << i;
                    }
                }
            }
        }
    }
}

static void OutputPredicate(
    const TModelTrees& modelTrees,
    const TModelSplit& binFeature,
    bool isLeafNode,
    size_t leftChildIdx,
    size_t rightChildIdx,
    const TOneHotValuesToIdx& oneHotValuesToIdx,
    TXmlOutputContext* xmlOut) {

    auto featureType = binFeature.Type;

    if (featureType == ESplitType::FloatFeature) {
        int floatFeatureIdx = binFeature.FloatFeature.FloatFeature;
        const auto& floatFeature = modelTrees.GetFloatFeatures()[floatFeatureIdx];

        if (!isLeafNode) {
            if (floatFeature.HasNans &&
                (floatFeature.NanValueTreatment == TFloatFeature::ENanValueTreatment::AsTrue))
            {
                xmlOut->AddAttr("defaultChild", rightChildIdx);
            } else {
                xmlOut->AddAttr("defaultChild", leftChildIdx);
            }
        }

        TXmlElementOutputContext simplePredicate(xmlOut, "SimplePredicate");
        xmlOut->AddAttr("field", CreateFeatureName(floatFeature))
            .AddAttr("operator", "greaterThan")
            .AddAttr("value", binFeature.FloatFeature.Split);
    } else {
        Y_ASSERT(featureType == ESplitType::OneHotFeature);

        if (!isLeafNode) {
            xmlOut->AddAttr("defaultChild", leftChildIdx);
        }

        int catFeatureIdx = binFeature.OneHotFeature.CatFeatureIdx;
        const auto& catFeature = modelTrees.GetCatFeatures()[catFeatureIdx];

        TXmlElementOutputContext simplePredicate(xmlOut, "SimplePredicate");
        xmlOut->AddAttr("field", CreateFeatureName(catFeature) + "_mapped")
            .AddAttr("operator", "equal")
            .AddAttr("value", oneHotValuesToIdx[catFeatureIdx].at(binFeature.OneHotFeature.Value));
    }
}


static void OutputNodeSymmetric(
    const TModelTrees& modelTrees,
    size_t treeIdx,
    size_t treeFirstGlobalLeafIdx,
    size_t layer,
    size_t leafIdx,
    const TOneHotValuesToIdx& oneHotValuesToIdx,
    TConstArrayRef<double> nodeWeights,
    TXmlOutputContext* xmlOut) {

    TXmlElementOutputContext node(xmlOut, "Node");
    xmlOut->AddAttr("id", leafIdx);

    const bool isLeafNode = layer == SafeIntegerCast<size_t>(modelTrees.GetModelTreeData()->GetTreeSizes()[treeIdx]);

    if (isLeafNode) {
        xmlOut->AddAttr(
            "score",
            modelTrees.GetModelTreeData()->GetLeafValues()[treeFirstGlobalLeafIdx + leafIdx + 1 - (size_t(1) << layer)]);
    }
    xmlOut->AddAttr(
        "recordCount",
        nodeWeights[leafIdx]);

    // predicate
    if ((layer != 0) && (leafIdx % 2 == 0)) {
        const int splitIdx
            = modelTrees.GetModelTreeData()->GetTreeStartOffsets()[treeIdx] + modelTrees.GetModelTreeData()->GetTreeSizes()[treeIdx] - layer;
        const auto& binFeature = modelTrees.GetBinFeatures()[modelTrees.GetModelTreeData()->GetTreeSplits().at(splitIdx)];

        OutputPredicate(
            modelTrees,
            binFeature,
            isLeafNode,
            /*leftChildIdx*/ 2 * leafIdx + 1,
            /*rightChildIdx*/ 2 * leafIdx + 2,
            oneHotValuesToIdx,
            xmlOut);
    } else {
        TXmlElementOutputContext predicate(xmlOut, "True");
    }

    if (!isLeafNode) {
        for (auto childLeafIdx : {2 * leafIdx + 2, 2 * leafIdx + 1}) {
            OutputNodeSymmetric(
                modelTrees,
                treeIdx,
                treeFirstGlobalLeafIdx,
                layer + 1,
                childLeafIdx,
                oneHotValuesToIdx,
                nodeWeights,
                xmlOut);
        }
    }
}


static TVector<double> CalculateNodeWeights(
    const TModelTrees& modelTrees,
    size_t treeIdx,
    size_t treeFirstGlobalLeafIdx) {

    auto lastLayer = SafeIntegerCast<size_t>(modelTrees.GetModelTreeData()->GetTreeSizes()[treeIdx]);
    auto knownIdxBegin = (size_t(1) << lastLayer) - 1;

    TVector<double> weights(2 * knownIdxBegin + 1);
    std::copy_n(
        modelTrees.GetModelTreeData()->GetLeafWeights().begin() + treeFirstGlobalLeafIdx,
        size_t(1) << lastLayer,
        weights.begin() + knownIdxBegin);
    while (knownIdxBegin > 0) {
        auto prevLayerBegin = knownIdxBegin / 2;
        auto readIdx = knownIdxBegin;
        for (auto writeIdx = prevLayerBegin; writeIdx != knownIdxBegin; ++writeIdx) {
            weights[writeIdx] = weights[readIdx++];
            weights[writeIdx] += weights[readIdx++];
        }
        knownIdxBegin = prevLayerBegin;
    }
    return weights;
}

static void OutputNodeNonSymmetric(
    const TModelTrees& modelTrees,
    size_t treeIdx,
    size_t nodeIdx,
    size_t *outputNodeIdx,
    bool isRightChild,
    bool isDummyLeaf,
    size_t parentNodeIdx,
    const TOneHotValuesToIdx& oneHotValuesToIdx,
    TConstArrayRef<double> nodeWeights,
    TConstArrayRef<size_t> leftChildren,
    TXmlOutputContext* xmlOut) {

    size_t leftStep = 0;
    size_t rightStep = 0;
    if (!isDummyLeaf) {
        leftStep = modelTrees.GetModelTreeData()->GetNonSymmetricStepNodes()[nodeIdx].LeftSubtreeDiff;
        rightStep = modelTrees.GetModelTreeData()->GetNonSymmetricStepNodes()[nodeIdx].RightSubtreeDiff;
    }

    const bool isLeafNode = leftStep == 0 && rightStep == 0;

    TXmlElementOutputContext node(xmlOut, "Node");
    xmlOut->AddAttr("id", *outputNodeIdx);

    if (isLeafNode) {
        xmlOut->AddAttr(
            "score",
            modelTrees.GetModelTreeData()->GetLeafValues()[modelTrees.GetModelTreeData()->GetNonSymmetricNodeIdToLeafId()[nodeIdx]]);
    }
    xmlOut->AddAttr(
        "recordCount",
        nodeWeights[*outputNodeIdx]);

    // predicate
    if (isRightChild) {
        const auto& binFeature = modelTrees.GetBinFeatures()[modelTrees.GetModelTreeData()->GetTreeSplits().at(parentNodeIdx)];

        OutputPredicate(
            modelTrees,
            binFeature,
            isLeafNode,
            leftChildren[*outputNodeIdx],
            /*rightChildIdx*/ *outputNodeIdx + 1,
            oneHotValuesToIdx,
            xmlOut);
    } else {
        TXmlElementOutputContext predicate(xmlOut, "True");
    }

    ++*outputNodeIdx;

    if (!isLeafNode) {
        OutputNodeNonSymmetric(
            modelTrees,
            treeIdx,
            nodeIdx + rightStep,
            outputNodeIdx,
            /*isRightChild*/true,
            /*isDummyLeaf*/rightStep == 0,
            nodeIdx,
            oneHotValuesToIdx,
            nodeWeights,
            leftChildren,
            xmlOut);
        OutputNodeNonSymmetric(
            modelTrees,
            treeIdx,
            nodeIdx + leftStep,
            outputNodeIdx,
            /*isRightChild*/false,
            /*isDummyLeaf*/leftStep == 0,
            nodeIdx,
            oneHotValuesToIdx,
            nodeWeights,
            leftChildren,
            xmlOut);
    }
}


static void CalculateNodeWeightsAndLeftChildrenNonSymmetric(
    const TModelTrees& modelTrees,
    size_t treeIdx,
    size_t nodeIdx,
    TVector<double>* weights,
    TVector<size_t>* leftChildren) {

    const auto& stepNode = modelTrees.GetModelTreeData()->GetNonSymmetricStepNodes()[nodeIdx];
    const auto outputIdx = weights->size();
    weights->push_back(0.0);
    leftChildren->push_back(0);

    if (stepNode.LeftSubtreeDiff == 0 && stepNode.RightSubtreeDiff == 0) {
        (*weights)[outputIdx] = modelTrees.GetModelTreeData()->GetLeafWeights()[modelTrees.GetModelTreeData()->GetNonSymmetricNodeIdToLeafId()[nodeIdx]];
        return;
    }

    if (stepNode.RightSubtreeDiff != 0) {
        CalculateNodeWeightsAndLeftChildrenNonSymmetric(
            modelTrees,
            treeIdx,
            nodeIdx + stepNode.RightSubtreeDiff,
            weights,
            leftChildren);
    } else {
        // dummy child
        weights->push_back(0.0);
        leftChildren->push_back(0);
    }
    (*leftChildren)[outputIdx] = weights->size();

    if (stepNode.LeftSubtreeDiff != 0) {
        CalculateNodeWeightsAndLeftChildrenNonSymmetric(
            modelTrees,
            treeIdx,
            nodeIdx + stepNode.LeftSubtreeDiff,
            weights,
            leftChildren);
    } else {
        // dummy child
        weights->push_back(0.0);
        leftChildren->push_back(0);
    }
    (*weights)[outputIdx] = (*weights)[outputIdx + 1] + (*weights)[(*leftChildren)[outputIdx]];
}

static void OutputTree(
    const TFullModel& model,
    size_t treeIdx,
    TStringBuf targetName,
    const TOneHotValuesToIdx& oneHotValuesToIdx,
    size_t* obliviousTreeFirstGlobalLeafIdx,
    TXmlOutputContext* xmlOut) {

    TXmlElementOutputContext treeModel(xmlOut, "TreeModel");

    xmlOut->AddAttr("modelName", TStringBuilder() << "tree_" << treeIdx)
        .AddAttr("functionName", "regression")
        .AddAttr("missingValueStrategy", "defaultChild")
        .AddAttr("splitCharacteristic", "binarySplit");

    OutputMiningSchemaWithModelFeatures(model, /*mappedCategoricalFeatures*/ true, targetName, xmlOut);

    {
        TXmlElementOutputContext output(xmlOut, "Output");
        TXmlElementOutputContext outputField(xmlOut, "OutputField");
        xmlOut->AddAttr("name", targetName).AddAttr("optype", "continuous").AddAttr("dataType", "double");
    }

    if (model.IsOblivious()) {
        auto weights = CalculateNodeWeights(*model.ModelTrees, treeIdx, *obliviousTreeFirstGlobalLeafIdx);

        OutputNodeSymmetric(
            *model.ModelTrees,
            treeIdx,
            *obliviousTreeFirstGlobalLeafIdx,
            /*layer*/ 0,
            /*leafIdx*/ 0,
            oneHotValuesToIdx,
            weights,
            xmlOut);
        *obliviousTreeFirstGlobalLeafIdx += (size_t(1) << model.ModelTrees->GetModelTreeData()->GetTreeSizes()[treeIdx]);
    } else {
        const auto treeStartOffset = model.ModelTrees->GetModelTreeData()->GetTreeStartOffsets()[treeIdx];
        TVector<double> weights;
        TVector<size_t> leftChildren;
        CalculateNodeWeightsAndLeftChildrenNonSymmetric(
            *model.ModelTrees,
            treeIdx,
            treeStartOffset,
            &weights,
            &leftChildren);
        size_t outputNodeIdx = 0;
        OutputNodeNonSymmetric(
            *model.ModelTrees,
            treeIdx,
            treeStartOffset,
            &outputNodeIdx,
            /*isRightChild*/ false,
            /*isDummyLeaf*/ false,
            /*parentNodeIdx*/ 0,
            oneHotValuesToIdx,
            weights,
            leftChildren,
            xmlOut);
    }
}


static void OutputTreeEnsemble(
    const TFullModel& model,
    TStringBuf targetName,
    bool isChainPart,
    const THashMap<ui32, TString>* catFeaturesHashToString,
    TOneHotValuesToIdx* oneHotValuesToIdx,  //  mapping is empty - and must be filled - if it's top-level model
    TXmlOutputContext* xmlOut) {

    TXmlElementOutputContext miningModel(xmlOut, "MiningModel");

    // Tree ensemble outputs approx, so it is always a regression
    xmlOut->AddAttr("functionName", "regression");

    OutputMiningSchemaWithModelFeatures(
        model,
        /*mappedCategoricalFeatures*/ false,
        !isChainPart ? TMaybe<TStringBuf>(targetName) : Nothing(),
        xmlOut);

    {
        TXmlElementOutputContext output(xmlOut, "Output");
        TXmlElementOutputContext outputField(xmlOut, "OutputField");
        xmlOut->AddAttr("name", targetName).AddAttr("optype", "continuous").AddAttr("dataType", "double");
    }

    const auto& modelTrees = *model.ModelTrees;

    if (!isChainPart && modelTrees.GetOneHotFeatures().size()) {
        OutputCategoricalMapping(model, *catFeaturesHashToString, oneHotValuesToIdx, xmlOut);
    }

    {
        TXmlElementOutputContext segmentation(xmlOut, "Segmentation");
        xmlOut->AddAttr("multipleModelMethod", "sum");

        size_t obliviousTreeFirstGlobalLeafIdx = 0;
        for (auto treeIdx : xrange(modelTrees.GetModelTreeData()->GetTreeSizes().size())) {
            TXmlElementOutputContext segment(xmlOut, "Segment");
            xmlOut->AddAttr("id", treeIdx);

            // predicate
            {
                TXmlElementOutputContext predicate(xmlOut, "True");
            }

            OutputTree(model, treeIdx, targetName, *oneHotValuesToIdx, &obliviousTreeFirstGlobalLeafIdx, xmlOut);
        }
    }
    if (!isChainPart) {
        OutputTargetsFields(model, xmlOut);
    }
}


static void OutputClassFromApprox(TXmlOutputContext* xmlOut) {
    TXmlElementOutputContext treeModel(xmlOut, "TreeModel");

    xmlOut->AddAttr("modelName", "selectClass")
        .AddAttr("functionName", "classification")
        .AddAttr("splitCharacteristic", "binarySplit");

    {
        TXmlElementOutputContext miningSchema(xmlOut, "MiningSchema");
        {
            TXmlElementOutputContext miningField(xmlOut, "MiningField");
            xmlOut->AddAttr("name", "approx").AddAttr("usageType", "active");
        }
    }

    {
        TXmlElementOutputContext output(xmlOut, "Output");
        TXmlElementOutputContext outputField(xmlOut, "OutputField");
        xmlOut->AddAttr("name", "prediction").AddAttr("optype", "categorical").AddAttr("dataType", "boolean");
    }

    TXmlElementOutputContext node(xmlOut, "Node");
    xmlOut->AddAttr("id", "root");

    // predicate
    {
        TXmlElementOutputContext predicate(xmlOut, "True");
    }
    {
        TXmlElementOutputContext node(xmlOut, "Node");
        xmlOut->AddAttr("id", "1").AddAttr("score", "true");

        TXmlElementOutputContext simplePredicate(xmlOut, "SimplePredicate");
        xmlOut->AddAttr("field", "approx")
            .AddAttr("operator", "greaterThan")
            .AddAttr("value", "0.0");
    }
    {
        TXmlElementOutputContext node(xmlOut, "Node");
        xmlOut->AddAttr("id", "0").AddAttr("score", "false");

        TXmlElementOutputContext predicate(xmlOut, "True");
    }
}


static void OutputMiningModel(
    const TFullModel& model,
    bool isClassification,
    const THashMap<ui32, TString>* catFeaturesHashToString,
    TXmlOutputContext* xmlOut) {

    TOneHotValuesToIdx oneHotValuesToIdx;
    if (isClassification) {
        TXmlElementOutputContext miningModel(xmlOut, "MiningModel");
        xmlOut->AddAttr("functionName", "classification");

        OutputMiningSchemaWithModelFeatures(model, /*mappedCategoricalFeatures*/ false, "prediction", xmlOut);

        if (model.ModelTrees->GetOneHotFeatures().size()) {
            OutputCategoricalMapping(model, *catFeaturesHashToString, &oneHotValuesToIdx, xmlOut);
        }

        {
            TXmlElementOutputContext segmentation(xmlOut, "Segmentation");
            xmlOut->AddAttr("multipleModelMethod", "modelChain");

            {
                TXmlElementOutputContext segment(xmlOut, "Segment");
                xmlOut->AddAttr("id", "treeEnsemble");

                // predicate
                {
                    TXmlElementOutputContext predicate(xmlOut, "True");
                }

                OutputTreeEnsemble(model, "approx", /*isChainPart*/ true, catFeaturesHashToString, &oneHotValuesToIdx, xmlOut);
            }
            {
                TXmlElementOutputContext segment(xmlOut, "Segment");
                xmlOut->AddAttr("id", "classifier");

                // predicate
                {
                    TXmlElementOutputContext predicate(xmlOut, "True");
                }

                OutputClassFromApprox(xmlOut);
            }
        }
        OutputTargetsFields(model, xmlOut);
    } else {
        OutputTreeEnsemble(model, "prediction", /*isChainPart*/ false, catFeaturesHashToString, &oneHotValuesToIdx, xmlOut);
    }
}

namespace NCB {
    namespace NPmml {

    void OutputModel(
        const TFullModel& model,
        const TString& modelFile,
        const NJson::TJsonValue& userParameters,
        const THashMap<ui32, TString>* catFeaturesHashToString) {

        CB_ENSURE(
            SafeIntegerCast<size_t>(
                CountIf(
                    model.ModelTrees->GetCatFeatures(),
                    [](const TCatFeature& catFeature) { return catFeature.UsedInModel(); }))
                == model.ModelTrees->GetOneHotFeatures().size(),
            "PMML export requires that all categorical features in the model are one hot encoded");

        CB_ENSURE(
            model.GetDimensionsCount() == 1,
            "PMML export currently supports only single-dimensional models");

        CB_ENSURE_INTERNAL(
            !model.ModelTrees->GetOneHotFeatures().size() || catFeaturesHashToString,
            "catFeaturesHashToString has to be specified if the model contains one hot features");

        // assumed regression by default

        bool isClassification = false;
        const TString lossFunctionName = model.GetLossFunctionName();
        if (lossFunctionName) {
            isClassification = IsClassificationObjective(lossFunctionName);
        }

        TOFStream out(modelFile);
        TXmlOutputContext xmlOut(&out, "PMML");
        xmlOut.AddAttr("version", "4.3")
            .AddAttr("xmlns", "http://www.dmg.org/PMML-4_3")
            .AddAttr("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance");

        OutputHeader(model, userParameters, &xmlOut);

        OutputDataDictionary(model, isClassification, &xmlOut);

        OutputMiningModel(model, isClassification, catFeaturesHashToString, &xmlOut);
    }

    }
}
