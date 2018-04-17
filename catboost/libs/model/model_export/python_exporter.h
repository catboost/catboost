#pragma once

#include "model_exporter.h"

namespace NCatboost {
    class TCatboostModelToPythonConverter: public ICatboostModelExporter {
    private:
        TOFStream Out;

    public:
        TCatboostModelToPythonConverter(const TString& modelFile, bool addFileFormatExtension, const TString& userParametersJSON)
            : Out(modelFile + (addFileFormatExtension ? ".py" : ""))
        {
            CB_ENSURE(userParametersJSON.empty(), "JSON user params for exporting the model to Python are not supported");
        };

        void Write(const TFullModel& model) override {
            if (model.HasCategoricalFeatures()) {
                WriteHeaderCatFeatures();
                WriteModelCatFeatures(model);
                WriteApplicatorCatFeatures();
            } else {
                WriteModel(model);
                WriteApplicator();
            }
        }

    private:
        void WriteApplicator();
        void WriteModel(const TFullModel& model);

        void WriteHeaderCatFeatures();
        void WriteCTRStructs();
        void WriteModelCatFeatures(const TFullModel& model);
        void WriteApplicatorCatFeatures();

    };
}
