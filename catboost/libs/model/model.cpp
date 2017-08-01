#include "model.h"
#include "coreml_helpers.h"

#include <catboost/libs/helpers/exception.h>

#include <contrib/libs/coreml/TreeEnsemble.pb.h>
#include <contrib/libs/coreml/Model.pb.h>

#include <library/json/json_reader.h>

#include <util/stream/str.h>
#include <util/stream/file.h>

void OutputModel(const TFullModel& model, const TString& modelFile) {
    TOFStream f(modelFile);
    Save(&f, model);
}

TFullModel ReadModel(const TString& modelFile) {
    TIFStream f(modelFile);
    TFullModel model;
    Load(&f, model);
    return model;
}

void OutputModelCoreML(const TFullModel& model, const TString& modelFile, const NJson::TJsonValue& userParameters) {
    CoreML::Specification::Model outModel;
    outModel.set_specificationversion(1);

    auto regressor = outModel.mutable_treeensembleregressor();
    auto ensemble = regressor->mutable_treeensemble();
    auto description = outModel.mutable_description();

    NCatboost::NCoreML::ConfigureMetadata(userParameters, description);
    NCatboost::NCoreML::ConfigureTrees(model, ensemble);
    NCatboost::NCoreML::ConfigureIO(model, userParameters, regressor, description);

    TString data;
    outModel.SerializeToString(&data);

    TOFStream out(modelFile);
    out.Write(data);
}

void ExportModel(const TFullModel& model, const TString& modelFile, const EModelExportType format, const TString& userParametersJSON) {
    switch (format) {
        case EModelExportType::CatboostBinary:
            CB_ENSURE(userParametersJSON.empty(), "user params for mode not supported");
            OutputModel(model, modelFile);
            break;
        case EModelExportType::AppleCoreML:
            TStringInput is(userParametersJSON);
            NJson::TJsonValue params;
            NJson::ReadJsonTree(&is, &params);

            OutputModelCoreML(model, modelFile, params);
            break;
    }
}

TString SerializeModel(const TFullModel& model) {
    TStringStream ss;
    Save(&ss, model);
    return ss.Str();
}

TFullModel DeserializeModel(const TString& serializeModelString) {
    TStringStream ss(serializeModelString);
    TFullModel model;
    Load(&ss, model);
    return model;
}
