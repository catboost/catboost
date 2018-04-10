#pragma once

#include "model_exporter.h"

namespace NCatboost {
    class TCatboostModelToCppConverter: public ICatboostModelExporter {
    private:
        TOFStream Out;

    public:
        TCatboostModelToCppConverter(const TString& modelFile, bool addFileFormatExtension, const TString& userParametersJSON)
            : Out(modelFile + (addFileFormatExtension ? ".cpp" : ""))
        {
            CB_ENSURE(userParametersJSON.empty(), "JSON user params for exporting the model to C++ are not supported");
        };

        void Write(const TFullModel& model) override {
            WriteHeader();
            WriteModel(model);
            WriteApplicator();
        }

    private:
        void WriteApplicator();
        void WriteModel(const TFullModel& model);
        void WriteHeader();
    };
}
