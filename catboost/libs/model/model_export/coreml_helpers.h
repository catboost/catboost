#pragma once

#include <catboost/libs/model/model.h>

#include <contrib/libs/coreml/TreeEnsemble.pb.h>
#include <contrib/libs/coreml/Model.pb.h>

#include <library/cpp/json/json_value.h>

#include <util/generic/hash.h>


namespace NCB {
    namespace NCoreML {
        struct TPerTypeFeatureIdxToInputIndex {
            THashMap<int, int> ForFloatFeatures;
            THashMap<int, int> ForCatFeatures;
        };

        void ConfigureTrees(const TFullModel& model, const TPerTypeFeatureIdxToInputIndex& perTypeFeatureIdxToInputIndex, CoreML::Specification::TreeEnsembleParameters* ensemble);
        void ConfigureCategoricalMappings(const TFullModel& model, const THashMap<ui32, TString>* catFeaturesHashToString, google::protobuf::RepeatedPtrField<CoreML::Specification::Model>* container);
        void ConfigureTreeModelIO(const TFullModel& model, const NJson::TJsonValue& userParameters, CoreML::Specification::TreeEnsembleRegressor* regressor, CoreML::Specification::ModelDescription* description, TPerTypeFeatureIdxToInputIndex* perTypeFeatureIdxToInputIndex);
        void ConfigurePipelineModelIO(const TFullModel& model, CoreML::Specification::ModelDescription* description);
        void ConfigureFloatInput(const TFullModel& model, CoreML::Specification::ModelDescription* description, THashMap<int, int>* perTypeFeatureIdxToInputIndex = nullptr);
        void ConfigureMetadata(const TFullModel& model, const NJson::TJsonValue& userParameters, CoreML::Specification::ModelDescription* description);

        void ConvertCoreMLToCatboostModel(const CoreML::Specification::Model& coreMLModel, TFullModel* fullModel);
    }
}
