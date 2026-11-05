#pragma once

#include "model_exporter.h"

#include <catboost/libs/helpers/exception.h>

#include <util/generic/string.h>
#include <util/stream/file.h>

namespace NCB {
    class TCatboostModelToCppConverter: public ICatboostModelExporter {
    private:
        TOFStream Out;
        TString Namespace;

    public:
        TCatboostModelToCppConverter(const TString& modelFile, bool addFileFormatExtension, const TString& userParametersJson);
        void Write(const TFullModel& model, const THashMap<ui32, TString>* catFeaturesHashToString = nullptr) override;

    private:
        void WriteApplicator(bool forCatFeatures);
        void WriteModel(bool forCatFeatures, const TFullModel& model, const THashMap<ui32, TString>* catFeaturesHashToString);
        void WriteHeader(bool forCatFeatures);
        void WriteCTRStructs();
        void WriteNamespaceBegin();
        void WriteNamespaceEnd();
    };
}
