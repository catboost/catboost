#include "coreml_helpers.h"
#include "model_build_helper.h"

#include <catboost/libs/helpers/exception.h>

#include <util/generic/bitops.h>
#include <util/generic/set.h>
#include <util/generic/vector.h>

#include <utility>

using namespace CoreML::Specification;

void NCatboost::NCoreML::ConfigureTrees(const TFullModel& model, TreeEnsembleParameters* ensemble, bool* createMappingModel) {
    const auto classesCount = static_cast<size_t>(model.ObliviousTrees.ApproxDimension);
    auto& binFeatures = model.ObliviousTrees.GetBinFeatures();
    size_t currentSplitIndex = 0;
    auto currentTreeFirstLeafPtr = model.ObliviousTrees.LeafValues.data();

    size_t catFeaturesCount  = model.ObliviousTrees.CatFeatures.size();
    size_t floatFeaturesCount  = model.ObliviousTrees.FloatFeatures.size();
    TVector<THashMap<int, double>> splitCategoricalValues(catFeaturesCount);
    TVector<int> categoricalFlatIndexes(catFeaturesCount);
    TVector<int> floatFlatIndexes(floatFeaturesCount);

    for (const auto& catFeature: model.ObliviousTrees.CatFeatures) {
        categoricalFlatIndexes[catFeature.FeatureIndex] = catFeature.FlatFeatureIndex;
    }

    for (const auto& floatFeature: model.ObliviousTrees.FloatFeatures) {
        floatFlatIndexes[floatFeature.FeatureIndex] = floatFeature.FlatFeatureIndex;
    }

    for (const auto& oneHotFeature: model.ObliviousTrees.OneHotFeatures) {
        THashMap<int, double> valuesMapping;
        for (size_t i = 0; i < oneHotFeature.Values.size(); i++) {
            valuesMapping.insert(std::pair<int, double>(oneHotFeature.Values[i], double(i)));
        }
        splitCategoricalValues[oneHotFeature.CatFeatureIndex] = std::move(valuesMapping);
    }

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
            auto featureType = binFeature.Type;
            CB_ENSURE(featureType == ESplitType::FloatFeature || featureType == ESplitType::OneHotFeature,
                      "model with only float features or one hot encoded features supported");

            int featureId = -1;
            float branchValueCat;
            float branchValueFloat;
            auto branchParameter = TreeEnsembleParameters::TreeNode::BranchOnValueGreaterThan;
            if (featureType == ESplitType::FloatFeature) {
                int floatFeatureId = binFeature.FloatFeature.FloatFeature;
                branchValueFloat = binFeature.FloatFeature.Split;
                featureId = floatFlatIndexes[floatFeatureId];
            } else {
                int catFeatureId = binFeature.OneHotFeature.CatFeatureIdx;
                branchValueCat = splitCategoricalValues[catFeatureId][binFeature.OneHotFeature.Value];
                branchParameter = TreeEnsembleParameters::TreeNode::BranchOnValueEqual;
                featureId = categoricalFlatIndexes[catFeatureId];
            }

            auto nodesInLayerCount = std::pow(2, layer);
            TVector<TreeEnsembleParameters::TreeNode*> currentLayer(nodesInLayerCount);
            *createMappingModel = false;

            for (size_t nodeIdx = 0; nodeIdx < nodesInLayerCount; ++nodeIdx) {
                auto branchNode = ensemble->add_nodes();

                branchNode->set_treeid(treeIdx);
                branchNode->set_nodeid(lastNodeId);
                ++lastNodeId;

                branchNode->set_nodebehavior(branchParameter);
                branchNode->set_branchfeatureindex(featureId);
                if (featureType == ESplitType::FloatFeature) {
                    branchNode->set_branchfeaturevalue(branchValueFloat);
                } else {
                    branchNode->set_branchfeaturevalue(branchValueCat);
                    *createMappingModel = true;
                }

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

void NCatboost::NCoreML::ConfigureCategoricalMappings(const TFullModel& model,
                                                      const THashMap<ui32, TString>* catFeaturesHashToString,
                                                      google::protobuf::RepeatedPtrField<CoreML::Specification::Model>* container) {
    size_t catFeaturesCount  = model.ObliviousTrees.CatFeatures.size();
    TVector<int> categoricalFlatIndexes(catFeaturesCount);
    for (const auto& catFeature: model.ObliviousTrees.CatFeatures) {
        categoricalFlatIndexes[catFeature.FeatureIndex] = catFeature.FlatFeatureIndex;
    }

    for (const auto& oneHotFeature : model.ObliviousTrees.OneHotFeatures) {
        int flatFeatureIndex = categoricalFlatIndexes[oneHotFeature.CatFeatureIndex];
        std::unordered_map<TString, long> categoricalMapping;
        auto* contained = container->Add();

        CoreML::Specification::Model mappingModel;
        auto mapping = mappingModel.mutable_categoricalmapping();

        auto valuesCount = oneHotFeature.Values.size();

        for (size_t j = 0; j < valuesCount; j++) {
            ui32 oneHotValue = ui32(oneHotFeature.Values[j]);
            categoricalMapping.insert(std::make_pair(catFeaturesHashToString->find(oneHotValue)->second, j));
        }

        mapping->set_int64value(valuesCount);
        auto* stringtoint64map = mapping->mutable_stringtoint64map();
        auto* map = stringtoint64map->mutable_map();
        map->insert(categoricalMapping.begin(), categoricalMapping.end());

        auto description = mappingModel.mutable_description();
        auto catFeature = description->add_input();
        catFeature->set_name(("feature_" + std::to_string(flatFeatureIndex)).c_str());

        auto featureType = new FeatureType();
        featureType->set_isoptional(false);
        featureType->set_allocated_stringtype(new StringFeatureType());
        catFeature->set_allocated_type(featureType);

        auto mappedCategoricalFeature = description->add_output();
        mappedCategoricalFeature->set_name(("mapped_feature_" + std::to_string(flatFeatureIndex)).c_str());

        auto mappedCategoricalFeatureType = new FeatureType();
        mappedCategoricalFeatureType->set_isoptional(false);
        mappedCategoricalFeatureType->set_allocated_doubletype(new DoubleFeatureType());
        mappedCategoricalFeature->set_allocated_type(mappedCategoricalFeatureType);

        *contained = mappingModel;
    }
}

void NCatboost::NCoreML::ConfigureFloatInput(const TFullModel& model, CoreML::Specification::ModelDescription* description) {
    for (const auto& floatFeature : model.ObliviousTrees.FloatFeatures) {
        auto feature = description->add_input();
        feature->set_name(("feature_" + std::to_string(floatFeature.FlatFeatureIndex)).c_str());

        auto featureType = new FeatureType();
        featureType->set_isoptional(false);
        featureType->set_allocated_doubletype(new DoubleFeatureType());
        feature->set_allocated_type(featureType);
    }
}

void NCatboost::NCoreML::ConfigurePipelineModelIO(const TFullModel& model, CoreML::Specification::ModelDescription* description) {
    ConfigureFloatInput(model, description);

    size_t catFeaturesCount  = model.ObliviousTrees.CatFeatures.size();
    TVector<int> categoricalFlatIndexes(catFeaturesCount);
    for (const auto& catFeature: model.ObliviousTrees.CatFeatures) {
        categoricalFlatIndexes[catFeature.FeatureIndex] = catFeature.FlatFeatureIndex;
    }

    for (const auto& oneHotFeature : model.ObliviousTrees.OneHotFeatures) {
        auto feature = description->add_input();
        int flatFeatureIndex = categoricalFlatIndexes[oneHotFeature.CatFeatureIndex];

        feature->set_name(("feature_" + std::to_string(flatFeatureIndex)).c_str());

        auto featureType = new FeatureType();
        featureType->set_isoptional(false);
        featureType->set_allocated_stringtype(new StringFeatureType());
        feature->set_allocated_type(featureType);
    }

    const auto classesCount = static_cast<size_t>(model.ObliviousTrees.ApproxDimension);
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
}

void NCatboost::NCoreML::ConfigureTreeModelIO(const TFullModel& model, const NJson::TJsonValue& userParameters, TreeEnsembleRegressor* regressor, ModelDescription* description) {
    ConfigureFloatInput(model, description);

    size_t catFeaturesCount  = model.ObliviousTrees.CatFeatures.size();
    TVector<int> categoricalFlatIndexes(catFeaturesCount);
    for (const auto& catFeature: model.ObliviousTrees.CatFeatures) {
        categoricalFlatIndexes[catFeature.FeatureIndex] = catFeature.FlatFeatureIndex;
    }

    for (const auto& oneHotFeature : model.ObliviousTrees.OneHotFeatures) {
        auto feature = description->add_input();
        int flatFeatureIndex = categoricalFlatIndexes[oneHotFeature.CatFeatureIndex];

        feature->set_name(("mapped_feature_" + std::to_string(flatFeatureIndex)).c_str());

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

    if (!model.ModelInfo.empty()) {
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
    TreeEnsembleRegressor regressor;
    TVector<TCatFeature> catFeatures;
    if (coreMLModel.has_pipeline()) {
        for (const auto& pipelineModel: coreMLModel.pipeline().models()) {
            CB_ENSURE(pipelineModel.has_treeensembleregressor(), "expected treeensembleregressor model");
            regressor = pipelineModel.treeensembleregressor();
            CB_ENSURE(pipelineModel.has_categoricalmapping(), "expected categorical mapping in model");
            auto& mapping = pipelineModel.categoricalmapping();
            CB_ENSURE(mapping.has_stringtoint64map(), "expected stringtoint64map in categorical mapping");
            auto& stringtoint64map = mapping.stringtoint64map();
            auto& categoricalMapping = stringtoint64map.map();
            for (const auto& el: categoricalMapping) {
                TCatFeature catFeature;
                // TODO: update FlatFeatureIndex and FeatureIndex
                catFeature.FeatureId = el.first;
                catFeatures.push_back(catFeature);
            }
        }
    } else {
        CB_ENSURE(coreMLModel.has_treeensembleregressor(), "expected treeensembleregressor model");
        regressor = coreMLModel.treeensembleregressor();
    }

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
    TObliviousTreeBuilder treeBuilder(floatFeatures, catFeatures, approxDimension);
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
