#include "model_import_interface.h"

#include <catboost/libs/logging/logging.h>
#include <catboost/private/libs/algo_helpers/custom_objective_descriptor.h>
#include <catboost/private/libs/options/json_helper.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/private/libs/options/check_train_options.h>

namespace NCB {
    class TBinaryModelLoader : public NCB::IModelLoader {
    public:
        TFullModel ReadModel(IInputStream* modelStream) const override {
            TFullModel model;
            Load(modelStream, model);
            CheckModel(&model);
            return model;
        }
    };

    NCB::TModelLoaderFactory::TRegistrator<TBinaryModelLoader> BinaryModelLoaderRegistrator(EModelType::CatboostBinary);

    void* BinaryModelLoaderRegistratorPointer = &BinaryModelLoaderRegistrator;

#ifndef CATBOOST_NO_PARAMS_CHECK_ON_LOAD
    static NJson::TJsonValue RemoveInvalidParams(const NJson::TJsonValue& params) {
        // We are defining those stubs only for deserialized model params checks, so it should be safe
        TCustomObjectiveDescriptor objectiveDescriptorStub;
        TCustomMetricDescriptor metricDescriptorStub;
        try {
            CheckFitParams(params, &objectiveDescriptorStub, &metricDescriptorStub);
            return params;
        } catch (...) {
            CATBOOST_WARNING_LOG << "There are invalid params and some of them will be ignored." << Endl;
        }
        NJson::TJsonValue result(NJson::JSON_MAP);
        // TODO(sergmiller): make proper validation for each parameter separately
        for (const auto& param : params.GetMap()) {
            result[param.first] = param.second;

            try {
                CheckFitParams(result, &objectiveDescriptorStub, &metricDescriptorStub);
            } catch (...) {
                result.EraseValue(param.first);

                NJson::TJsonValue badParam;
                badParam[param.first] = param.second;
                CATBOOST_WARNING_LOG << "Parameter " << ToString<NJson::TJsonValue>(badParam)
                    << " is ignored, because it cannot be parsed." << Endl;
            }
        }
        return result;
    }
#endif

    void IModelLoader::CheckModel(TFullModel* model) const {
#ifdef CATBOOST_NO_PARAMS_CHECK_ON_LOAD
        Y_UNUSED(model);
#else
        if (model->ModelInfo.contains("params")) {
            NJson::TJsonValue paramsJson = ReadTJsonValue(model->ModelInfo.at("params"));
            paramsJson["flat_params"] = RemoveInvalidParams(paramsJson["flat_params"]);
            model->ModelInfo["params"] = ToString<NJson::TJsonValue>(paramsJson);
        }
#endif
    }
}
