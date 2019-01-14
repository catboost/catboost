#pragma once

#include "model_exporter.h"

#include <catboost/libs/helpers/exception.h>

#include <util/stream/file.h>


namespace NCatboost {
    class TCatboostModelToPythonConverter: public ICatboostModelExporter {
    private:
        TOFStream Out;

    public:
        TCatboostModelToPythonConverter(const TString& modelFile, bool addFileFormatExtension, const TString& userParametersJson)
            : Out(modelFile + (addFileFormatExtension ? ".py" : ""))
        {
            CB_ENSURE(userParametersJson.empty(), "JSON user params for exporting the model to Python are not supported");
        };

        void Write(const TFullModel& model, const THashMap<ui32, TString>* catFeaturesHashToString) override {
            if (model.HasCategoricalFeatures()) {
                CB_ENSURE(catFeaturesHashToString != nullptr, "need pool to output model hashes");
            }
            WriteModelCatFeatures(model, catFeaturesHashToString);
            WriteApplicatorCatFeatures();
        }

    private:
        void WriteCTRStructs();
        void WriteModelCatFeatures(const TFullModel& model, const THashMap<ui32, TString>* catFeaturesHashToString);
        void WriteApplicatorCatFeatures();

    };
}
