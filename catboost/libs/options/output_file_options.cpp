#include "output_file_options.h"

TString NCatboostOptions::GetModelExtensionFromType(const EModelType& modelType) {
    switch(modelType) {
        case EModelType::CatboostBinary:
            return "bin";
        case EModelType::AppleCoreML:
            return "coreml";
        case EModelType::json:
            return "json";
        case EModelType::CPP:
            return "cpp";
        case EModelType::Python:
            return "py";
    }
}

bool NCatboostOptions::TryGetModelTypeFromExtension(const TString& modelExtension, EModelType& modelType) {
    if (modelExtension == "bin") {
        modelType = EModelType::CatboostBinary;
    } else if (modelExtension == "coreml") {
        modelType = EModelType::AppleCoreML;
    } else if (modelExtension =="json") {
        modelType = EModelType::json;
    } else if(modelExtension == "cpp") {
        modelType = EModelType::CPP;
    } else if (modelExtension == "py") {
        modelType = EModelType::Python;
    } else {
        return false;
    }
    return true;
}

EModelType NCatboostOptions::DefineModelFormat(const TString& modelPath) {
    EModelType modelType;
    TVector<TString> tokens;
    if (Split(modelPath, ".", tokens) > 1) {
        if (NCatboostOptions::TryGetModelTypeFromExtension(tokens.back(), modelType)) {
            return modelType;
        }
    }
    return EModelType::CatboostBinary;
}

void NCatboostOptions::AddExtension(const TString& extension, TString* modelFileName) {
    if (!modelFileName->EndsWith("." + extension)) {
        *modelFileName += "." + extension;
    }

}
