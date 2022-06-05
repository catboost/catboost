#pragma once

#include "export_helpers.h"

#include "model_exporter.h"

#include <catboost/libs/helpers/exception.h>

#include <util/stream/file.h>


namespace NCB {
    class TCatboostModelToCppConverter: public ICatboostModelExporter {
    private:
        TOFStream Out;

    public:
        TCatboostModelToCppConverter(const TString& modelFile, bool addFileFormatExtension, const TString& userParametersJson)
            : Out(modelFile + (addFileFormatExtension ? ".cpp" : ""))
        {
            CB_ENSURE(userParametersJson.empty(), "JSON user params for exporting the model to C++ are not supported");
        };

        void Write(const TFullModel& model, const THashMap<ui32, TString>* catFeaturesHashToString = nullptr) override {
            bool nanModeMax = NCatboostModelExportHelpers::IsNanModeMax(model);
            if (model.HasCategoricalFeatures()) {
                CB_ENSURE(catFeaturesHashToString != nullptr,
                          "need train pool to save mapping {categorical feature value, hash value} "
                          "due to absence of hash function in model");
                WriteHeader(/*forCatFeatures*/true, nanModeMax);
                WriteBinarizer(nanModeMax);
                WriteModelCatFeatures(model, catFeaturesHashToString);
                WriteApplicatorCatFeatures();
            } else {
                WriteHeader(/*forCatFeatures*/false, nanModeMax);
                WriteModel(model);
                WriteApplicator(nanModeMax);
            }
        }

    private:
        void WriteApplicator(bool nanModeMax);
        void WriteModel(const TFullModel& model);
        void WriteHeader(bool forCatFeatures, bool nanModeMax);
        void WriteBinarizer(bool nanModeMax);
        void WriteCTRStructs();
        void WriteModelCatFeatures(const TFullModel& model, const THashMap<ui32, TString>* catFeaturesHashToString);
        void WriteApplicatorCatFeatures();
    };
}
