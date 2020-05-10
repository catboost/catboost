#pragma once

#include <catboost/libs/model/model.h>

#include <library/cpp/json/json_value.h>

NJson::TJsonValue ConvertModelToJson(
    const TFullModel& model,
    const TVector<TString>* featureId=nullptr,
    const THashMap<ui32, TString>* catFeaturesHashToString=nullptr);

void OutputModelJson(
    const TFullModel& model,
    const TString& outputPath,
    const TVector<TString>* featureId=nullptr,
    const THashMap<ui32, TString>* catFeaturesHashToString=nullptr);

void ConvertJsonToCatboostModel(const NJson::TJsonValue& jsonModel, TFullModel* fullModel);

TString ModelCtrBaseToStr(const TModelCtrBase& modelCtrBase);
