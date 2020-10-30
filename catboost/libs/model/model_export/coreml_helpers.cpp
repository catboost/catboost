#include "coreml_helpers.h"
#include <catboost/libs/model/model_build_helper.h>

#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/cat_feature/cat_feature.h>

#include <util/generic/bitops.h>
#include <util/generic/cast.h>
#include <util/generic/set.h>
#include <util/generic/vector.h>
#include <util/generic/xrange.h>

#include <utility>

using namespace CoreML::Specification;

void NCB::NCoreML::ConfigureTrees(const TFullModel& model, const TPerTypeFeatureIdxToInputIndex& perTypeFeatureIdxToInputIndex, TreeEnsembleParameters* ensemble) {
    const auto classesCount = static_cast<size_t>(model.ModelTrees->GetDimensionsCount());
    const auto binFeatures = model.ModelTrees->GetBinFeatures();
    size_t currentSplitIndex = 0;
    auto currentTreeFirstLeafPtr = model.ModelTrees->GetModelTreeData()->GetLeafValues().data();

    size_t catFeaturesCount = model.ModelTrees->GetCatFeatures().size();
    TVector<THashMap<int, double>> splitCategoricalValues(catFeaturesCount);

    for (const auto& oneHotFeature: model.ModelTrees->GetOneHotFeatures()) {
        THashMap<int, double> valuesMapping;
        for (size_t i = 0; i < oneHotFeature.Values.size(); i++) {
            valuesMapping.insert(std::pair<int, double>(oneHotFeature.Values[i], double(i)));
        }
        splitCategoricalValues[oneHotFeature.CatFeatureIndex] = std::move(valuesMapping);
    }

    for (size_t treeIdx = 0; treeIdx < model.ModelTrees->GetModelTreeData()->GetTreeSizes().size(); ++treeIdx) {
        const size_t leafCount = (1uLL << model.ModelTrees->GetModelTreeData()->GetTreeSizes()[treeIdx]);
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
                    currentTreeFirstLeafPtr[leafIdx * model.ModelTrees->GetDimensionsCount() + classIdx]);
            }

            outputLeaves[leafIdx] = leafNode;
        }
        currentTreeFirstLeafPtr += leafCount * model.ModelTrees->GetDimensionsCount();

        auto& previousLayer = outputLeaves;
        auto treeDepth = model.ModelTrees->GetModelTreeData()->GetTreeSizes()[treeIdx];
        for (int layer = treeDepth - 1; layer >= 0; --layer) {
            const auto& binFeature = binFeatures[model.ModelTrees->GetModelTreeData()->GetTreeSplits().at(currentSplitIndex)];
            ++currentSplitIndex;
            auto featureType = binFeature.Type;
            CB_ENSURE(featureType == ESplitType::FloatFeature || featureType == ESplitType::OneHotFeature,
                      "model with only float features or one hot encoded features supported");

            int inputFeatureIndex = -1;
            float branchValueCat;
            float branchValueFloat;
            auto branchParameter = TreeEnsembleParameters::TreeNode::BranchOnValueGreaterThan;
            if (featureType == ESplitType::FloatFeature) {
                int floatFeatureId = binFeature.FloatFeature.FloatFeature;
                branchValueFloat = binFeature.FloatFeature.Split;
                inputFeatureIndex = perTypeFeatureIdxToInputIndex.ForFloatFeatures.at(floatFeatureId);
            } else {
                int catFeatureId = binFeature.OneHotFeature.CatFeatureIdx;
                branchValueCat = splitCategoricalValues[catFeatureId][binFeature.OneHotFeature.Value];
                branchParameter = TreeEnsembleParameters::TreeNode::BranchOnValueEqual;
                inputFeatureIndex = perTypeFeatureIdxToInputIndex.ForCatFeatures.at(catFeatureId);
            }

            auto nodesInLayerCount = std::pow(2, layer);
            TVector<TreeEnsembleParameters::TreeNode*> currentLayer(nodesInLayerCount);

            for (size_t nodeIdx = 0; nodeIdx < nodesInLayerCount; ++nodeIdx) {
                auto branchNode = ensemble->add_nodes();

                branchNode->set_treeid(treeIdx);
                branchNode->set_nodeid(lastNodeId);
                ++lastNodeId;

                branchNode->set_nodebehavior(branchParameter);
                branchNode->set_branchfeatureindex(inputFeatureIndex);
                if (featureType == ESplitType::FloatFeature) {
                    branchNode->set_branchfeaturevalue(branchValueFloat);
                } else {
                    branchNode->set_branchfeaturevalue(branchValueCat);
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

void NCB::NCoreML::ConfigureCategoricalMappings(const TFullModel& model,
                                                      const THashMap<ui32, TString>* catFeaturesHashToString,
                                                      google::protobuf::RepeatedPtrField<CoreML::Specification::Model>* container) {
    size_t catFeaturesCount  = model.ModelTrees->GetCatFeatures().size();
    TVector<int> categoricalFlatIndexes(catFeaturesCount);
    for (const auto& catFeature: model.ModelTrees->GetCatFeatures()) {
        categoricalFlatIndexes[catFeature.Position.Index] = catFeature.Position.FlatIndex;
    }

    for (auto oneHotFeatureIdx : xrange(model.ModelTrees->GetOneHotFeatures().size())) {
        const auto& oneHotFeature = model.ModelTrees->GetOneHotFeatures()[oneHotFeatureIdx];

        int flatFeatureIndex = categoricalFlatIndexes[oneHotFeature.CatFeatureIndex];
        THashMap<TString, long> categoricalMapping;
        auto* contained = container->Add();

        CoreML::Specification::Model mappingModel;
        mappingModel.set_specificationversion(1);
        auto mapping = mappingModel.mutable_categoricalmapping();

        auto valuesCount = oneHotFeature.Values.size();
        for (size_t j = 0; j < valuesCount; j++) {
            ui32 oneHotValue = ui32(oneHotFeature.Values[j]);
            categoricalMapping.insert(std::make_pair(catFeaturesHashToString->find(oneHotValue)->second, j));
        }

        mapping->set_int64value(i64(valuesCount));
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
        mappedCategoricalFeatureType->set_allocated_int64type(new Int64FeatureType());
        mappedCategoricalFeature->set_allocated_type(mappedCategoricalFeatureType);

        *contained = mappingModel;
    }
}

void NCB::NCoreML::ConfigureFloatInput(
    const TFullModel& model,
    CoreML::Specification::ModelDescription* description,
    THashMap<int, int>* perTypeFeatureIdxToInputIndex) {
    for (auto floatFeatureIdx : xrange(model.ModelTrees->GetFloatFeatures().size())) {
        const auto& floatFeature = model.ModelTrees->GetFloatFeatures()[floatFeatureIdx];
        if (perTypeFeatureIdxToInputIndex) {
            (*perTypeFeatureIdxToInputIndex)[floatFeature.Position.Index] = description->input().size();
        }

        auto feature = description->add_input();
        feature->set_name(("feature_" + std::to_string(floatFeature.Position.FlatIndex)).c_str());

        auto featureType = new FeatureType();
        featureType->set_isoptional(false);
        featureType->set_allocated_doubletype(new DoubleFeatureType());
        feature->set_allocated_type(featureType);
    }
}

void NCB::NCoreML::ConfigurePipelineModelIO(const TFullModel& model,
                                                  CoreML::Specification::ModelDescription* description) {
    ConfigureFloatInput(model, description);

    size_t catFeaturesCount  = model.ModelTrees->GetCatFeatures().size();
    TVector<int> categoricalFlatIndexes(catFeaturesCount);
    for (const auto& catFeature: model.ModelTrees->GetCatFeatures()) {
        categoricalFlatIndexes[catFeature.Position.Index] = catFeature.Position.FlatIndex;
    }

    for (const auto& oneHotFeature : model.ModelTrees->GetOneHotFeatures()) {

        auto feature = description->add_input();
        int flatFeatureIndex = categoricalFlatIndexes[oneHotFeature.CatFeatureIndex];

        feature->set_name(("feature_" + std::to_string(flatFeatureIndex)).c_str());

        auto featureType = new FeatureType();
        featureType->set_isoptional(false);
        featureType->set_allocated_stringtype(new StringFeatureType());
        feature->set_allocated_type(featureType);
    }

    const auto classesCount = static_cast<size_t>(model.ModelTrees->GetDimensionsCount());
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

void NCB::NCoreML::ConfigureTreeModelIO(
    const TFullModel& model,
    const NJson::TJsonValue& userParameters,
    TreeEnsembleRegressor* regressor,
    ModelDescription* description,
    TPerTypeFeatureIdxToInputIndex* perTypeFeatureIdxToInputIndex) {

    ConfigureFloatInput(model, description, &(perTypeFeatureIdxToInputIndex->ForFloatFeatures));

    size_t catFeaturesCount  = model.ModelTrees->GetCatFeatures().size();
    TVector<int> categoricalFlatIndexes(catFeaturesCount);
    for (const auto& catFeature: model.ModelTrees->GetCatFeatures()) {
        categoricalFlatIndexes[catFeature.Position.Index] = catFeature.Position.FlatIndex;
    }

    for (const auto& oneHotFeature : model.ModelTrees->GetOneHotFeatures()) {
        (*perTypeFeatureIdxToInputIndex).ForCatFeatures[oneHotFeature.CatFeatureIndex] = description->input().size();

        auto feature = description->add_input();
        int flatFeatureIndex = categoricalFlatIndexes[oneHotFeature.CatFeatureIndex];

        feature->set_name(("mapped_feature_" + std::to_string(flatFeatureIndex)).c_str());

        auto featureType = new FeatureType();
        featureType->set_isoptional(false);
        featureType->set_allocated_int64type(new Int64FeatureType());
        feature->set_allocated_type(featureType);
    }

    const auto classesCount = static_cast<size_t>(model.ModelTrees->GetDimensionsCount());
    regressor->mutable_treeensemble()->set_numpredictiondimensions(classesCount);
    if (classesCount == 1) {
        regressor->mutable_treeensemble()->add_basepredictionvalue(
            model.ModelTrees->GetScaleAndBias().GetOneDimensionalBias(
                "Non single-dimension approxes are not supported")
        );
    } else {
        for (size_t outputIdx = 0; outputIdx < classesCount; ++outputIdx) {
            regressor->mutable_treeensemble()->add_basepredictionvalue(0.0);
        }
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

void NCB::NCoreML::ConfigureMetadata(const TFullModel& model, const NJson::TJsonValue& userParameters, ModelDescription* description) {
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

struct TFeaturesMetaData {
    TVector<TFloatFeature> FloatFeatures;
    TVector<TCatFeature> CategoricalFeatures;
    TVector<int> InputIndexToPerTypeIndex;
    THashMap<int, int> FlatFeatureIndexToPerTypeIndex;

    // key: catFeatureIndex, value: inverted categorical mapping from index (0, 1, ...) of category's hash value
    TVector<THashMap<int, int>> CategoricalMappings;
};

void ProcessOneTree(const TVector<const TreeEnsembleParameters::TreeNode*>& tree, size_t approxDimension,
                    const TFeaturesMetaData& featuresMetaData,
                    TVector<TSet<double>>* floatFeatureBorders,
                    TVector<TModelSplit>* splits,
                    TVector<TVector<double>>* leafValues) {
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

            if (node->nodebehavior() == TreeEnsembleParameters::TreeNode::BranchOnValueGreaterThan) {
                if (splits->ysize() <= nodeLayerIds[nodeId]) {
                    splits->resize(nodeLayerIds[nodeId] + 1);
                    auto& split = splits->at(nodeLayerIds[nodeId]);
                    split.FloatFeature.FloatFeature = featuresMetaData.InputIndexToPerTypeIndex.at(node->branchfeatureindex());
                    split.FloatFeature.Split = node->branchfeaturevalue();
                    split.Type = ESplitType::FloatFeature;

                    (*floatFeatureBorders)[split.FloatFeature.FloatFeature].insert(split.FloatFeature.Split);
                } else {
                    auto& floatSplit = splits->at(nodeLayerIds[nodeId]).FloatFeature;
                    CB_ENSURE(floatSplit.FloatFeature == featuresMetaData.InputIndexToPerTypeIndex.at(node->branchfeatureindex()));
                    CB_ENSURE(floatSplit.Split == node->branchfeaturevalue());
                }
            } else {
                if (splits->ysize() <= nodeLayerIds[nodeId]) {
                    splits->resize(nodeLayerIds[nodeId] + 1);
                    auto& split = splits->at(nodeLayerIds[nodeId]);
                    split.OneHotFeature.CatFeatureIdx = featuresMetaData.InputIndexToPerTypeIndex.at(node->branchfeatureindex());
                    split.OneHotFeature.Value = featuresMetaData.CategoricalMappings[split.OneHotFeature.CatFeatureIdx].at((int)node->branchfeaturevalue());
                    split.Type = ESplitType::OneHotFeature;
                } else {
                    auto& oneHotSplit = splits->at(nodeLayerIds[nodeId]).OneHotFeature;
                    CB_ENSURE(oneHotSplit.CatFeatureIdx == featuresMetaData.InputIndexToPerTypeIndex.at(node->branchfeatureindex()));
                    CB_ENSURE(oneHotSplit.Value == featuresMetaData.CategoricalMappings[oneHotSplit.CatFeatureIdx].at((int)node->branchfeaturevalue()));
                }
            }
        }
    }
    CB_ENSURE(splits->size() <= 16);
    CB_ENSURE(IsPowerOf2(leafValues->at(0).size()), "There should be 2^depth leaves in model");
}

}

void NCB::NCoreML::ConvertCoreMLToCatboostModel(const Model& coreMLModel, TFullModel* fullModel) {
    CB_ENSURE(coreMLModel.specificationversion() == 1, "expected specificationVersion == 1");

    TFeaturesMetaData featuresMetaData;

    const auto& input = coreMLModel.description().input();
    for (auto inputIdx : xrange(input.size())) {
        const auto& featureDescription = input.Get(inputIdx);
        // names of catFeatures are stored in description as "feature_" + std::to_string(flatFeatureIndex)
        int flatFeatureIdx = std::stoi(featureDescription.name().substr(8));
        if (featureDescription.type().has_stringtype()) {
            TCatFeature catFeature;
            catFeature.Position.Index = SafeIntegerCast<int>(featuresMetaData.CategoricalFeatures.size());
            featuresMetaData.InputIndexToPerTypeIndex.push_back(catFeature.Position.Index);
            featuresMetaData.FlatFeatureIndexToPerTypeIndex[flatFeatureIdx] = catFeature.Position.Index;
            catFeature.Position.FlatIndex = flatFeatureIdx;
            catFeature.FeatureId = featureDescription.name();
            featuresMetaData.CategoricalFeatures.push_back(std::move(catFeature));
        } else {
            CB_ENSURE_INTERNAL(featureDescription.type().has_doubletype(), "bad CoreML feature type");

            TFloatFeature floatFeature;
            floatFeature.Position.Index = SafeIntegerCast<int>(featuresMetaData.FloatFeatures.size());
            featuresMetaData.InputIndexToPerTypeIndex.push_back(floatFeature.Position.Index);
            featuresMetaData.FlatFeatureIndexToPerTypeIndex[flatFeatureIdx] = floatFeature.Position.Index;
            floatFeature.Position.FlatIndex = flatFeatureIdx;
            floatFeature.FeatureId = featureDescription.name();
            featuresMetaData.FloatFeatures.push_back(std::move(floatFeature));
        }
    }

    TreeEnsembleRegressor regressor;

    if (coreMLModel.has_pipeline()) {
        auto& models = coreMLModel.pipeline().models();
        size_t modelsCount = models.size();
        CB_ENSURE(modelsCount > 0, "expected nonempty pipeline");
        CB_ENSURE(models.Get(modelsCount - 1).has_treeensembleregressor(), "expected treeensembleregressor model");
        regressor = models.Get(modelsCount - 1).treeensembleregressor();

        for (size_t i = 0; i < modelsCount - 1; i++) {
            auto& model = models.Get(i);
            CB_ENSURE(model.has_categoricalmapping(), "expected categorical mappings in pipeline");

            auto& mapping = model.categoricalmapping().stringtoint64map().map();
            // names of catFeatures are stored in description as "feature_" + std::to_string(flatFeatureIndex)
            int flatFeatureIdx = std::stoi(model.description().input(0).name().substr(8));
            THashMap<int, int> invertedMapping;

            for (auto& key_value: mapping) {
                invertedMapping.insert(std::pair<int, int>(key_value.second, CalcCatFeatureHashInt(key_value.first)));
            }
            int catFeatureIdx = featuresMetaData.FlatFeatureIndexToPerTypeIndex.at(flatFeatureIdx);
            if (catFeatureIdx >= SafeIntegerCast<int>(featuresMetaData.CategoricalMappings.size())) {
                featuresMetaData.CategoricalMappings.resize(catFeatureIdx + 1);
            }
            featuresMetaData.CategoricalMappings[catFeatureIdx] = std::move(invertedMapping);
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
    TVector<TVector<TModelSplit>> treesSplits;
    TVector<TVector<TVector<double>>> leafValues;
    for (const auto& node : ensemble.nodes()) {
        if (node.treeid() >= treeNodes.size()) {
            treeNodes.resize(node.treeid() + 1);
        }
        treeNodes[node.treeid()].push_back(&node);
    }

    TVector<TSet<double>> floatFeatureBorders(featuresMetaData.FloatFeatures.size()); // [floatFeatureIdx]

    for (const auto& tree : treeNodes) {
        CB_ENSURE(!tree.empty(), "incorrect coreml model: empty tree");
        auto& treeSplits = treesSplits.emplace_back();
        auto& leaves = leafValues.emplace_back();
        ProcessOneTree(tree, approxDimension, featuresMetaData, &floatFeatureBorders, &treeSplits, &leaves);
    }

    for (auto floatFeatureIdx : xrange(featuresMetaData.FloatFeatures.size())) {
        featuresMetaData.FloatFeatures[floatFeatureIdx].Borders.assign(
            floatFeatureBorders[floatFeatureIdx].begin(),
            floatFeatureBorders[floatFeatureIdx].end());
    }


    TObliviousTreeBuilder treeBuilder(
        featuresMetaData.FloatFeatures,
        featuresMetaData.CategoricalFeatures,
        {},
        {},
        approxDimension
    );
    for (size_t i = 0; i < treesSplits.size(); ++i) {
        treeBuilder.AddTree(treesSplits[i], leafValues[i]);
    }

    treeBuilder.Build(fullModel->ModelTrees.GetMutable());
    fullModel->ModelInfo.clear();
    if (description.has_metadata()) {
        auto& metadata = description.metadata();
        for (const auto& key_value : metadata.userdefined()) {
            fullModel->ModelInfo[key_value.first] = key_value.second;
        }
    }
    if (approxDimension == 1) {
        fullModel->SetScaleAndBias({1., {ensemble.basepredictionvalue()[0]}});
    }
    fullModel->UpdateDynamicData();
}
