#pragma once

#include <catboost/libs/model/model.h>
#include <util/generic/ptr.h>

namespace NCB {
    class ICatboostModelExporter: public TThrRefBase {
    public:
        virtual ~ICatboostModelExporter() = default;

        virtual void Write(const TFullModel& model, const THashMap<ui32, TString>* catFeaturesHashToString) = 0;
    };
    /**
     * Export model in our binary or protobuf CoreML format
     * @param model
     * @param modelFile
     * @param format
     * @param userParametersJson
     * @param addFileFormatExtension
     * @param featureId
     * @param catFeaturesHashToString
     */
    void ExportModel(
        const TFullModel& model,
        const TString& modelFile,
        EModelType format,
        const TString& userParametersJson = "",
        bool addFileFormatExtension = false,
        const TVector<TString>* featureId=nullptr,
        const THashMap<ui32, TString>* catFeaturesHashToString=nullptr);

    TString ConvertTreeToOnnxProto(
        const TFullModel& model,
        const TString& userParametersJson = "");
}
