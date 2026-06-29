#pragma once

#include "model_exporter.h"
#include "library/cpp/json/json_reader.h"

#include <catboost/libs/helpers/exception.h>

#include <util/stream/file.h>


namespace NCB {
    class TCatboostModelToCppConverter: public ICatboostModelExporter {
    private:
        TOFStream Out;
        TString Namespace;

    public:
        TCatboostModelToCppConverter(const TString& modelFile, bool addFileFormatExtension, const TString& userParametersJson)
            : Out(modelFile + (addFileFormatExtension ? ".cpp" : ""))
        {
            if (userParametersJson.empty()) 
                return;

            // parse namespace using json_reader
            NJson::TJsonValue params;
            CB_ENSURE(
              NJson::ReadJsonTree(userParametersJson, &params), "Can't parse JSON user params for exporting model to C++");
            if (params.Has("namespace")) {
                const TString ns = params["namespace"].GetStringSafe();
                CB_ENSURE(IsValidCPPIdentifier(ns), "Invalid CPP identifier used for namespace: " << ns);
                Namespace = ns;
            }
        };

        void Write(const TFullModel& model, const THashMap<ui32, TString>* catFeaturesHashToString = nullptr) override {
            if (model.HasCategoricalFeatures()) {
                CB_ENSURE(catFeaturesHashToString != nullptr,
                          "Need to use training dataset Pool to save mapping {categorical feature value -> hash value} "
                          "due to the absence of a hash function in the model");
                WriteHeader(/*forCatFeatures*/true);
                WriteNamespaceBegin();
                WriteCTRStructs();
                WriteModel(/*forCatFeatures*/true, model, catFeaturesHashToString);
                WriteApplicator(/*forCatFeatures*/true);
                WriteNamespaceEnd();
            } else {
                WriteHeader(/*forCatFeatures*/false);
                WriteNamespaceBegin();
                WriteModel(/*forCatFeatures*/false, model, nullptr);
                WriteApplicator(/*forCatFeatures*/false);
                WriteNamespaceEnd();
            }
        }

    private:
        void WriteApplicator(bool forCatFeatures);
        void WriteModel(bool forCatFeatures, const TFullModel& model, const THashMap<ui32, TString>* catFeaturesHashToString);
        void WriteHeader(bool forCatFeatures);
        void WriteCTRStructs();
        void WriteNamespaceBegin();
        void WriteNamespaceEnd();
        bool IsValidCPPIdentifier(const TString& identifier);
    };
}
