#pragma once

#include "model.h"

#include <contrib/libs/coreml/TreeEnsemble.pb.h>
#include <contrib/libs/coreml/Model.pb.h>

#include <library/json/json_value.h>

namespace NCatboost {
    namespace NCoreML {
        void ConfigureTrees(const TFullModel& model, CoreML::Specification::TreeEnsembleParameters* ensemble);
        void ConfigureIO(const TFullModel& model, const NJson::TJsonValue& userParameters, CoreML::Specification::TreeEnsembleRegressor* regressor, CoreML::Specification::ModelDescription* description);
        void ConfigureMetadata(const TFullModel& model, const NJson::TJsonValue& userParameters, CoreML::Specification::ModelDescription* description);

        void ConvertCoreMLToCatboostModel(const CoreML::Specification::Model& coreMLModel, TFullModel* fullModel);
    }
}
