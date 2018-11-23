#pragma once

#include "model_exporter.h"

namespace NCatboost {
    class TCatboostModelToCppConverter: public ICatboostModelExporter {
    private:
        TOFStream Out;

    public:
        TCatboostModelToCppConverter(const TString& modelFile, bool addFileFormatExtension, const TString& userParametersJson)
            : Out(modelFile + (addFileFormatExtension ? ".cpp" : ""))
        {
            CB_ENSURE(userParametersJson.empty(), "JSON user params for exporting the model to C++ are not supported");
        };

        void Write(const TFullModel& model, const THashMap<int, TString>* catFeaturesHashToString = nullptr) override {
            if (model.HasCategoricalFeatures()) {
                WriteHeader(/*forCatFeatures*/true);
                WriteModelCatFeatures(model, catFeaturesHashToString);
                WriteApplicatorCatFeatures();
            } else {
                WriteHeader(/*forCatFeatures*/false);
                WriteModel(model);
                WriteApplicator();
            }
        }

    private:
        void WriteApplicator();
        void WriteModel(const TFullModel& model);
        void WriteHeader(bool forCatFeatures);
        void WriteCTRStructs();
        void WriteModelCatFeatures(const TFullModel& model, const THashMap<int, TString>* catFeaturesHashToString);
        void WriteApplicatorCatFeatures();
    };
}
