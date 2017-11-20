#include "coreml_helpers.h"
#include <catboost/libs/helpers/exception.h>

using namespace CoreML::Specification;

void NCatboost::NCoreML::ConfigureTrees(const TFullModel& model, TreeEnsembleParameters* ensemble) {
    const auto classesCount = static_cast<size_t>(model.ObliviousTrees.ApproxDimension);
    CB_ENSURE(model.ObliviousTrees.CatFeatures.empty(), "model with only float features supported");
    auto& binFeatures = model.ObliviousTrees.GetBinFeatures();
    size_t currentSplitIndex = 0;
    for (size_t treeIdx = 0; treeIdx < model.ObliviousTrees.TreeSizes.size(); ++treeIdx) {
        const auto leafsCount = model.ObliviousTrees.LeafValues[treeIdx].size() / model.ObliviousTrees.ApproxDimension;
        size_t lastNodeId = 0;

        TVector<TreeEnsembleParameters::TreeNode*> outputLeafs(leafsCount);

        for (size_t leafIdx = 0; leafIdx < leafsCount; ++leafIdx) {
            auto leafNode = ensemble->add_nodes();
            leafNode->set_treeid(treeIdx);
            leafNode->set_nodeid(lastNodeId);
            ++lastNodeId;

            leafNode->set_nodebehavior(TreeEnsembleParameters::TreeNode::LeafNode);

            auto evalInfoArray = leafNode->mutable_evaluationinfo();

            for (size_t classIdx = 0; classIdx < classesCount; ++classIdx) {
                auto evalInfo = evalInfoArray->Add();
                evalInfo->set_evaluationindex(classIdx);
                evalInfo->set_evaluationvalue(
                    model.ObliviousTrees.LeafValues[treeIdx][leafIdx * model.ObliviousTrees.ApproxDimension + classIdx]);
            }

            outputLeafs[leafIdx] = leafNode;
        }

        auto& previousLayer = outputLeafs;
        auto treeDepth = model.ObliviousTrees.TreeSizes[treeIdx];
        for (int layer = treeDepth - 1; layer >= 0; --layer) {
            const auto& binFeature = binFeatures[model.ObliviousTrees.TreeSplits.at(currentSplitIndex)];
            auto featureId = binFeature.FloatFeature.FloatFeature;
            auto branchValue = binFeature.FloatFeature.Split;

            auto nodesInLayerCount = std::pow(2, layer);
            TVector<TreeEnsembleParameters::TreeNode*> currentLayer(nodesInLayerCount);

            for (size_t nodeIdx = 0; nodeIdx < nodesInLayerCount; ++nodeIdx) {
                auto branchNode = ensemble->add_nodes();

                branchNode->set_treeid(treeIdx);
                branchNode->set_nodeid(lastNodeId);
                ++lastNodeId;

                branchNode->set_nodebehavior(TreeEnsembleParameters::TreeNode::BranchOnValueGreaterThan);
                branchNode->set_branchfeatureindex(featureId);
                branchNode->set_branchfeaturevalue(branchValue);

                branchNode->set_falsechildnodeid(
                    previousLayer[2 * nodeIdx]->nodeid());
                branchNode->set_truechildnodeid(
                    previousLayer[2 * nodeIdx + 1]->nodeid());

                currentLayer[nodeIdx] = branchNode;
            }

            previousLayer = currentLayer;
        }
    }
}

void NCatboost::NCoreML::ConfigureIO(const TFullModel& model, const NJson::TJsonValue& userParameters, TreeEnsembleRegressor* regressor, ModelDescription* description) {
    for (const auto& floatFeature : model.ObliviousTrees.FloatFeatures) {
        auto feature = description->add_input();
        if (!floatFeature.FeatureId.empty())
            feature->set_name(floatFeature.FeatureId);
        else
            feature->set_name(("feature_" + std::to_string(floatFeature.FeatureIndex)).c_str());

        auto featureType = new FeatureType();
        featureType->set_isoptional(false);
        featureType->set_allocated_doubletype(new DoubleFeatureType());
        feature->set_allocated_type(featureType);
    }

    const auto classesCount = static_cast<size_t>(model.ObliviousTrees.ApproxDimension);
    regressor->mutable_treeensemble()->set_numpredictiondimensions(classesCount);
    for (size_t outputIdx = 0; outputIdx < classesCount; ++outputIdx) {
        regressor->mutable_treeensemble()->add_basepredictionvalue(0.0);
    }

    auto outputPrediction = description->add_output();
    outputPrediction->set_name("prediction");
    description->set_predictedfeaturename("prediction");
    description->set_predictedprobabilitiesname("prediction");

    auto featureType = outputPrediction->mutable_type();
    featureType->set_isoptional(false);

    auto outputArray = new ArrayFeatureType();
    outputArray->set_datatype(ArrayFeatureType::DOUBLE);
    outputArray->add_shape(classesCount);

    featureType->set_allocated_multiarraytype(outputArray);

    const auto& prediction_type = userParameters["prediction_type"].GetString();
    if (prediction_type == "probability") {
        regressor->set_postevaluationtransform(TreeEnsemblePostEvaluationTransform::Classification_SoftMax);
    } else {
        regressor->set_postevaluationtransform(TreeEnsemblePostEvaluationTransform::NoTransform);
    }
}

void NCatboost::NCoreML::ConfigureMetadata(const NJson::TJsonValue& userParameters, ModelDescription* description) {
    auto meta = description->mutable_metadata();

    meta->set_shortdescription(
        userParameters["coreml_description"].GetStringSafe("Catboost model"));

    meta->set_versionstring(
        userParameters["coreml_model_version"].GetStringSafe("1.0.0"));

    meta->set_author(
        userParameters["coreml_model_author"].GetStringSafe("Mr. Catboost Dumper"));

    meta->set_license(
        userParameters["coreml_model_license"].GetStringSafe(""));
}
