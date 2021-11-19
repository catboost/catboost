#pragma once

#include <catboost/private/libs/labels/label_converter.h>
#include <catboost/private/libs/options/catboost_options.h>
#include <catboost/private/libs/target/data_providers.h>

#include <catboost/libs/data/target.h>

#include <util/generic/array_ref.h>
#include <util/generic/fwd.h>
#include <util/generic/yexception.h>
#include <util/generic/vector.h>
#include <util/system/types.h>


class TClassTargetPreprocessor {
public:
    TClassTargetPreprocessor(
        const TString& plainJsonParamsAsString,
        NCB::ERawTargetType rawTargetType,
        bool trainDataHasWeights,
        bool trainDataHasGroups
    );

    bool IsNeedToProcessDistinctTargetValues() const;

    void ProcessDistinctFloatTargetValues(TConstArrayRef<float> distinctTargetValues);
    void ProcessDistinctIntTargetValues(TConstArrayRef<i32> distinctTargetValues);
    void ProcessDistinctStringTargetValues(const TVector<TString>& distinctTargetValues);

    TString GetLossFunction() const;

    // including possibly updated loss_function
    TString GetUpdatedCatBoostOptionsJsonAsString() const;

    TVector<i8> GetSerializedLabelConverter();

    // call after initialization and one of ProcessDistinct functions
    TVector<float> PreprocessFloatTarget(TConstArrayRef<float> targetValues);
    TVector<float> PreprocessIntTarget(TConstArrayRef<i32> targetValues);
    TVector<float> PreprocessStringTarget(const TVector<TString>& targetValues);

private:
    void UpdateLossFunctionIfNotSetAndInitCatBoostOptions(bool isMultiClass);

    void ProcessDistinctTargetValuesImpl(
        const NCB::TRawTarget& distinctTargetValues,
        NCB::ERawTargetType rawTargetType,
        size_t distinctTargetValueSize
    );

    TVector<float> PreprocessTargetImpl(
       const NCB::TRawTarget& targetValues,
       NCB::ERawTargetType rawTargetType
    );

private:
    // need to have JSON as well because CatBoostOptions.Save will save the default values as well and it is
    // not what we want (some values will be set on a later stage if they are not specified)
    NJson::TJsonValue CatBoostOptionsPlainJson;

    // Initialize only after loss function is set, otherwise Validate might not pass
    TMaybe<NCatboostOptions::TCatBoostOptions> CatBoostOptions;

    bool TrainDataHasWeights;
    bool TrainDataHasGroups;
    bool NeedToProcessDistinctTargetValues;
    TLabelConverter LabelConverter;
    ui32 ClassCount;
    bool IsRealTarget;
    NCB::TTargetCreationOptions TargetCreationOptions;
};
