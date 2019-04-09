#pragma once

#include "model.h"

#include <contrib/libs/coreml/TreeEnsemble.pb.h>
#include <contrib/libs/coreml/Model.pb.h>

#include <library/json/json_value.h>

namespace NCatboost {
    namespace NCoreML {
        void ConfigureTrees(const TFullModel& model, CoreML::Specification::TreeEnsembleParameters* ensemble, bool* createPipeline);
        void ConfigureCategoricalMappings(const TFullModel& model, google::protobuf::RepeatedPtrField<CoreML::Specification::Model>* container);
        void ConfigureArrayFeatureExtractor(const TFullModel& model, CoreML::Specification::ArrayFeatureExtractor* array, CoreML::Specification::ModelDescription* description);
        void ConfigureIO(const TFullModel& model, const NJson::TJsonValue& userParameters, CoreML::Specification::TreeEnsembleRegressor* regressor, CoreML::Specification::ModelDescription* description, bool pipeline);
        void ConfigureMetadata(const TFullModel& model, const NJson::TJsonValue& userParameters, CoreML::Specification::ModelDescription* description);

        void ConvertCoreMLToCatboostModel(const CoreML::Specification::Model& coreMLModel, TFullModel* fullModel);
    }
}
