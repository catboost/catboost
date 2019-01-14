#include "coreml_helpers.h"
#include "model_build_helper.h"

#include <catboost/libs/helpers/exception.h>

#include <util/generic/bitops.h>
#include <util/generic/set.h>
#include <util/generic/vector.h>


using namespace CoreML::Specification;

void NCatboost::NCoreML::ConfigureTrees(const TFullModel& model, TreeEnsembleParameters* ensemble) {
    const auto classesCount = static_cast<size_t>(model.ObliviousTrees.ApproxDimension);
    CB_ENSURE(!model.HasCategoricalFeatures(), "model with only float features supported");
    auto& binFeatures = model.ObliviousTrees.GetBinFeatures();
    size_t currentSplitIndex = 0;
    auto currentTreeFirstLeafPtr = model.ObliviousTrees.LeafValues.data();
    for (size_t treeIdx = 0; treeIdx < model.ObliviousTrees.TreeSizes.size(); ++treeIdx) {
        const size_t leafCount = (1uLL << model.ObliviousTrees.TreeSizes[treeIdx]);
        size_t lastNodeId = 0;

        TVector<TreeEnsembleParameters::TreeNode*> outputLeaves(leafCount);

        for (size_t leafIdx = 0; leafIdx < leafCount; ++leafIdx) {
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
                    currentTreeFirstLeafPtr[leafIdx * model.ObliviousTrees.ApproxDimension + classIdx]);
            }

            outputLeaves[leafIdx] = leafNode;
        }
        currentTreeFirstLeafPtr += leafCount * model.ObliviousTrees.ApproxDimension;

        auto& previousLayer = outputLeaves;
        auto treeDepth = model.ObliviousTrees.TreeSizes[treeIdx];
        for (int layer = treeDepth - 1; layer >= 0; --layer) {
            const auto& binFeature = binFeatures[model.ObliviousTrees.TreeSplits.at(currentSplitIndex)];
            ++currentSplitIndex;
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
        if (!floatFeature.FeatureId.empty()) {
            feature->set_name(floatFeature.FeatureId);
        } else {
            feature->set_name(("feature_" + std::to_string(floatFeature.FeatureIndex)).c_str());
        }

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

void NCatboost::NCoreML::ConfigureMetadata(const TFullModel& model, const NJson::TJsonValue& userParameters, ModelDescription* description) {
    auto meta = description->mutable_metadata();

    meta->set_shortdescription(
        userParameters["coreml_description"].GetStringSafe("Catboost model"));

    meta->set_versionstring(
        userParameters["coreml_model_version"].GetStringSafe("1.0.0"));

    meta->set_author(
        userParameters["coreml_model_author"].GetStringSafe("Mr. Catboost Dumper"));

    meta->set_license(
        userParameters["coreml_model_license"].GetStringSafe(""));
    if (model.ModelInfo.empty()) {
        auto& userDefinedRef = *meta->mutable_userdefined();
        for (const auto& key_value : model.ModelInfo) {
            userDefinedRef[key_value.first] = key_value.second;
        }
    }
}

namespace {

void ProcessOneTree(const TVector<const TreeEnsembleParameters::TreeNode*>& tree, size_t approxDimension, TVector<TFloatSplit>* splits, TVector<TVector<double>>* leafValues) {
    TVector<int> nodeLayerIds(tree.size(), -1);
    leafValues->resize(approxDimension);
    for (size_t nodeId = 0; nodeId < tree.size(); ++nodeId) {
        const auto node = tree[nodeId];
        CB_ENSURE(node->nodeid() == nodeId, "incorrect nodeid order in tree");
        if (node->nodebehavior() == TreeEnsembleParameters::TreeNode::LeafNode) {
            CB_ENSURE(node->evaluationinfo_size() == (int)approxDimension, "incorrect coreml model");

            for (size_t dim = 0; dim < approxDimension; ++dim) {
                auto& lvdim = leafValues->at(dim);
                auto& evalInfo = node->evaluationinfo(dim);
                CB_ENSURE(evalInfo.evaluationindex() == dim, "missing evaluation index or incrorrect order");
                if (lvdim.size() <= node->nodeid()) {
                    lvdim.resize(node->nodeid() + 1);
                }
                lvdim[node->nodeid()] = evalInfo.evaluationvalue();
            }
        } else {
            CB_ENSURE(node->falsechildnodeid() < nodeId);
            CB_ENSURE(node->truechildnodeid() < nodeId);
            CB_ENSURE(nodeLayerIds[node->falsechildnodeid()] == nodeLayerIds[node->truechildnodeid()]);
            nodeLayerIds[nodeId] = nodeLayerIds[node->falsechildnodeid()] + 1;
            if (splits->ysize() <= nodeLayerIds[nodeId]) {
                splits->resize(nodeLayerIds[nodeId] + 1);
                auto& floatSplit = splits->at(nodeLayerIds[nodeId]);
                floatSplit.FloatFeature = node->branchfeatureindex();
                floatSplit.Split = node->branchfeaturevalue();
            } else {
                auto& floatSplit = splits->at(nodeLayerIds[nodeId]);
                CB_ENSURE(floatSplit.FloatFeature == (int)node->branchfeatureindex());
                CB_ENSURE(floatSplit.Split == node->branchfeaturevalue());
            }
        }
    }
    CB_ENSURE(splits->size() <= 16);
    CB_ENSURE(IsPowerOf2(leafValues->at(0).size()), "There should be 2^depth leaves in model");
}

}

void NCatboost::NCoreML::ConvertCoreMLToCatboostModel(const Model& coreMLModel, TFullModel* fullModel) {
    CB_ENSURE(coreMLModel.specificationversion() == 1, "expected specificationVersion == 1");
    CB_ENSURE(coreMLModel.has_treeensembleregressor(), "expected treeensembleregressor model");
    auto& regressor = coreMLModel.treeensembleregressor();
    CB_ENSURE(regressor.has_treeensemble(), "no treeensemble in tree regressor");
    auto& ensemble = regressor.treeensemble();
    CB_ENSURE(coreMLModel.has_description(), "expected description in model");
    auto& description = coreMLModel.description();

    int approxDimension = ensemble.numpredictiondimensions();

    TVector<TVector<const TreeEnsembleParameters::TreeNode*>> treeNodes;
    TVector<TVector<TFloatSplit>> trees;
    TVector<TVector<TVector<double>>> leafValues;
    for (const auto& node : ensemble.nodes()) {
        if (node.treeid() >= treeNodes.size()) {
            treeNodes.resize(node.treeid() + 1);
        }
        treeNodes[node.treeid()].push_back(&node);
    }
    for (const auto& tree : treeNodes) {
        CB_ENSURE(!tree.empty(), "incorrect coreml model: empty tree");
        auto& treeSplits = trees.emplace_back();
        auto& leaves = leafValues.emplace_back();
        ProcessOneTree(tree, approxDimension, &treeSplits, &leaves);
    }
    TVector<TFloatFeature> floatFeatures;
    {
        TSet<TFloatSplit> floatSplitsSet;
        for (auto& tree : trees) {
            floatSplitsSet.insert(tree.begin(), tree.end());
        }
        int maxFeatureIndex = -1;
        for (auto& split : floatSplitsSet) {
            maxFeatureIndex = Max(maxFeatureIndex, split.FloatFeature);
        }
        floatFeatures.resize(maxFeatureIndex + 1);
        for (int i = 0; i < maxFeatureIndex + 1; ++i) {
            floatFeatures[i].FlatFeatureIndex = i;
            floatFeatures[i].FeatureIndex = i;
        }
        for (auto& split : floatSplitsSet) {
            auto& floatFeature = floatFeatures[split.FloatFeature];
            floatFeature.Borders.push_back(split.Split);
        }
    }
    TObliviousTreeBuilder treeBuilder(floatFeatures, TVector<TCatFeature>(), approxDimension);
    for (size_t i = 0; i < trees.size(); ++i) {
        TVector<TModelSplit> splits(trees[i].begin(), trees[i].end());
        treeBuilder.AddTree(splits, leafValues[i]);
    }
    fullModel->ObliviousTrees = treeBuilder.Build();
    fullModel->ModelInfo.clear();
    if (description.has_metadata()) {
        auto& metadata = description.metadata();
        for (const auto& key_value : metadata.userdefined()) {
            fullModel->ModelInfo[key_value.first] = key_value.second;
        }
    }
    fullModel->UpdateDynamicData();
}
