#pragma once

#include "model_exporter.h"

#include <catboost/libs/helpers/exception.h>

#include <util/stream/file.h>

#include <library/json/json_reader.h> 

namespace NCB {
    class TCatboostModelToCppConverter: public ICatboostModelExporter {
    private:
        TOFStream Out;
        TString Namespace; 

    public:
        TCatboostModelToCppConverter(const TString& modelFile, bool addFileFormatExtension, const TString& userParametersJson)
            : Out(modelFile + (addFileFormatExtension ? ".cpp" : ""))
        {
            if (!userParametersJson.empty()) {
              NJson::TJsonValue json;
              NJson::ReadJsonTree(userParametersJson, &json)
              if (json.Has(namespace)) {
                  Namespace = json["namespace"].GetString();
              }  
            }
        };

        void Write(const TFullModel& model, const THashMap<ui32, TString>* catFeaturesHashToString = nullptr) override {
            if (model.HasCategoricalFeatures()) {
                CB_ENSURE(catFeaturesHashToString != nullptr,
                          "Need to use training dataset Pool to save mapping {categorical feature value -> hash value} "
                          "due to the absence of a hash function in the model");
                WriteHeader(/*forCatFeatures*/true);
                WriteCTRStructs();
                WriteModel(/*forCatFeatures*/true, model, catFeaturesHashToString);
                WriteApplicator(/*forCatFeatures*/true);
            } else {
                WriteHeader(/*forCatFeatures*/false);
                WriteModel(/*forCatFeatures*/false, model, nullptr);
                WriteApplicator(/*forCatFeatures*/false);
            }
        }

    private:
        void WriteApplicator(bool forCatFeatures);
        void WriteModel(bool forCatFeatures, const TFullModel& model, const THashMap<ui32, TString>* catFeaturesHashToString);
        void WriteHeader(bool forCatFeatures);
        void WriteCTRStructs();
    };
}
