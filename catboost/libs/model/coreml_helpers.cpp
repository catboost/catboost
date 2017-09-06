#include "coreml_helpers.h"

using namespace CoreML::Specification;

void NCatboost::NCoreML::ConfigureTrees(const TFullModel& model, TreeEnsembleParameters* ensemble) {
    const auto classesCount = static_cast<size_t>(model.ApproxDimension);

    for (size_t treeIdx = 0; treeIdx < model.TreeStruct.size(); ++treeIdx) {
        const auto& tree = model.TreeStruct[treeIdx];
        const auto leafsCount = model.LeafValues[treeIdx][0].size();
        size_t lastNodeId = 0;

        yvector<TreeEnsembleParameters::TreeNode*> outputLeafs(leafsCount);

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
                    model.LeafValues[treeIdx][classIdx][leafIdx]);
            }

            outputLeafs[leafIdx] = leafNode;
        }

        auto& previousLayer = outputLeafs;
        for (int layer = tree.GetDepth() - 1; layer >= 0; --layer) {
            const auto& split = tree.SelectedSplits[tree.GetDepth() - 1 - layer];
            auto featureId = split.BinFeature.FloatFeature;
            auto splitId = split.BinFeature.SplitIdx;
            auto branchValue = model.Borders[featureId][splitId];

            auto nodesInLayerCount = std::pow(2, layer);
            yvector<TreeEnsembleParameters::TreeNode*> currentLayer(nodesInLayerCount);

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
    for (size_t featureIdx = 0; featureIdx < model.Borders.size(); ++featureIdx) {
        auto feature = description->add_input();
        if (featureIdx < model.FeatureIds.size())
            feature->set_name(model.FeatureIds[featureIdx]);
        else
            feature->set_name(("feature_" + std::to_string(featureIdx)).c_str());

        auto featureType = new FeatureType();
        featureType->set_isoptional(false);
        featureType->set_allocated_doubletype(new DoubleFeatureType());
        feature->set_allocated_type(featureType);
    }

    const auto classesCount = static_cast<size_t>(model.ApproxDimension);
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
