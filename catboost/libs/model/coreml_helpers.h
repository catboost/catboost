#pragma once

#include "model.h"

#include <contrib/libs/coreml/TreeEnsemble.pb.h>
#include <contrib/libs/coreml/Model.pb.h>

#include <library/json/json_value.h>

namespace NCatboost {
    namespace NCoreML {
        void ConfigureTrees(const TFullModel& model, CoreML::Specification::TreeEnsembleParameters* ensemble, bool* createPipeline);
        void ConfigureCategoricalMappings(const TFullModel& model, google::protobuf::RepeatedPtrField<CoreML::Specification::Model>* container);
        void ConfigureTreeModelIO(const TFullModel& model, const NJson::TJsonValue& userParameters, CoreML::Specification::TreeEnsembleRegressor* regressor, CoreML::Specification::ModelDescription* description);
        void ConfigurePipelineModelIO(const TFullModel& model, CoreML::Specification::ModelDescription* description);
        void ConfigureFloatInput(const TFullModel& model, CoreML::Specification::ModelDescription* description);
        void ConfigureMetadata(const TFullModel& model, const NJson::TJsonValue& userParameters, CoreML::Specification::ModelDescription* description);

        void ConvertCoreMLToCatboostModel(const CoreML::Specification::Model& coreMLModel, TFullModel* fullModel);
    }
}
