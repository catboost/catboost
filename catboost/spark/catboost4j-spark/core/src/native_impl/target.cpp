#include "target.h"

#include "options_helper.h"
#include "vector_output.h"

#include <catboost/private/libs/algo/data.h>
#include <catboost/private/libs/labels/helpers.h>
#include <catboost/private/libs/options/enum_helpers.h>
#include <catboost/private/libs/options/plain_options_helper.h>
#include <catboost/private/libs/target/data_providers.h>

#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/polymorphic_type_containers.h>

#include <library/cpp/binsaver/util_stream_io.h>
#include <library/cpp/json/json_writer.h>

#include <util/generic/cast.h>


using namespace NCB;


TClassTargetPreprocessor::TClassTargetPreprocessor(
    const TString& plainJsonParamsAsString,
    ERawTargetType rawTargetType,
    bool trainDataHasWeights,
    bool trainDataHasGroups
)
    : CatBoostOptionsPlainJson(ParseCatBoostPlainParamsToJson(plainJsonParamsAsString))
    , TrainDataHasWeights(trainDataHasWeights)
    , TrainDataHasGroups(trainDataHasGroups)
    , NeedToProcessDistinctTargetValues(false)
    , ClassCount(0)
    , IsRealTarget(false)
{
    CB_ENSURE(
        rawTargetType != ERawTargetType::None,
        "CatBoostClassifier requires a label column in the training dataset"
    );

    TMaybe<ELossFunction> specifiedLossFunction;
    if (CatBoostOptionsPlainJson.Has("loss_function")) {
        NCatboostOptions::TLossDescription lossDescription;
        lossDescription.Load(CatBoostOptionsPlainJson["loss_function"]);
        specifiedLossFunction = lossDescription.LossFunction.Get();
    }

    if ((rawTargetType == ERawTargetType::Boolean) ||
        CatBoostOptionsPlainJson.Has("target_border") ||
        (specifiedLossFunction &&
         IsBinaryClassOnlyMetric(*specifiedLossFunction) &&
         (CatBoostOptionsPlainJson.Has("class_names") || (rawTargetType == ERawTargetType::Integer))))
    {
        NeedToProcessDistinctTargetValues = false;
        UpdateLossFunctionIfNotSetAndInitCatBoostOptions(false);
        LabelConverter.InitializeBinClass();
        ClassCount = 2;
    } else {
        NeedToProcessDistinctTargetValues = true;
    }
}

bool TClassTargetPreprocessor::IsNeedToProcessDistinctTargetValues() const {
    return NeedToProcessDistinctTargetValues;
}

void TClassTargetPreprocessor::ProcessDistinctFloatTargetValues(
    TConstArrayRef<float> distinctTargetValues
) {
    ITypedSequencePtr<float> distinctTargetValuesAsTypedSequencePtr
        = MakeNonOwningTypeCastArrayHolder<float>(distinctTargetValues.begin(), distinctTargetValues.end());

    ProcessDistinctTargetValuesImpl(
        TRawTarget(distinctTargetValuesAsTypedSequencePtr),
        ERawTargetType::Float,
        distinctTargetValues.size()
    );
}

void TClassTargetPreprocessor::ProcessDistinctIntTargetValues(
    TConstArrayRef<i32> distinctTargetValues
) {
    ITypedSequencePtr<float> distinctTargetValuesAsTypedSequencePtr
        = MakeNonOwningTypeCastArrayHolder<float>(distinctTargetValues.begin(), distinctTargetValues.end());

    ProcessDistinctTargetValuesImpl(
        TRawTarget(distinctTargetValuesAsTypedSequencePtr),
        ERawTargetType::Integer,
        distinctTargetValues.size()
    );
}

void TClassTargetPreprocessor::ProcessDistinctStringTargetValues(
    const TVector<TString>& distinctTargetValues
) {
    ProcessDistinctTargetValuesImpl(
        TRawTarget(distinctTargetValues),
        ERawTargetType::String,
        distinctTargetValues.size()
    );
}

TString TClassTargetPreprocessor::GetLossFunction() const {
    TStringStream ss;
    ss << CatBoostOptions->LossFunctionDescription.Get();
    ss.Finish();
    return ss.Str();
}

// including possibly updated loss_function and class labels
TString TClassTargetPreprocessor::GetUpdatedCatBoostOptionsJsonAsString() const {
    return NJson::WriteJson(CatBoostOptionsPlainJson, false);
}

TVector<i8> TClassTargetPreprocessor::GetSerializedLabelConverter() {
    TVector<i8> result;
    {
        TVectorOutput out(&result);
        SerializeToArcadiaStream(out, LabelConverter);
    }
    return result;
}

TVector<float> TClassTargetPreprocessor::PreprocessFloatTarget(
    TConstArrayRef<float> targetValues
) {
    ITypedSequencePtr<float> targetValuesAsTypedSequencePtr
        = MakeNonOwningTypeCastArrayHolder<float>(targetValues.begin(), targetValues.end());

    return PreprocessTargetImpl(
        TRawTarget(targetValuesAsTypedSequencePtr),
        ERawTargetType::Float
    );
}

TVector<float> TClassTargetPreprocessor::PreprocessIntTarget(
    TConstArrayRef<i32> targetValues
) {
    ITypedSequencePtr<float> targetValuesAsTypedSequencePtr
        = MakeNonOwningTypeCastArrayHolder<float>(targetValues.begin(), targetValues.end());

    return PreprocessTargetImpl(
        TRawTarget(targetValuesAsTypedSequencePtr),
        ERawTargetType::Integer
   );
}

TVector<float> TClassTargetPreprocessor::PreprocessStringTarget(
    const TVector<TString>& targetValues
) {
    return PreprocessTargetImpl(TRawTarget(targetValues), ERawTargetType::String);
}


void TClassTargetPreprocessor::UpdateLossFunctionIfNotSetAndInitCatBoostOptions(bool isMultiClass) {
    if (!CatBoostOptionsPlainJson.Has("loss_function")) {
        if (isMultiClass) {
            CatBoostOptionsPlainJson["loss_function"] = "MultiClass";
        } else {
            CatBoostOptionsPlainJson["loss_function"] = "Logloss";
        }
    }
    CatBoostOptions = LoadCatBoostOptions(CatBoostOptionsPlainJson);
}

void TClassTargetPreprocessor::ProcessDistinctTargetValuesImpl(
    const NCB::TRawTarget& distinctTargetValues,
    ERawTargetType rawTargetType,
    size_t distinctTargetValuesSize
) {
    UpdateLossFunctionIfNotSetAndInitCatBoostOptions(
        (distinctTargetValuesSize > 2) && !CatBoostOptionsPlainJson.Has("target_border")
    );

    const auto& dataProcessingOptions = CatBoostOptions->DataProcessingOptions.Get();

    TInputClassificationInfo inputClassificationInfo {
        dataProcessingOptions.ClassesCount.Get() ?
            TMaybe<ui32>(dataProcessingOptions.ClassesCount.Get())
            : Nothing(),
        dataProcessingOptions.ClassWeights.Get(),
        dataProcessingOptions.AutoClassWeights.Get(),
        dataProcessingOptions.ClassLabels.Get(),
        dataProcessingOptions.TargetBorder.IsSet() ?
            TMaybe<float>(dataProcessingOptions.TargetBorder.Get())
            : Nothing()
    };

    TargetCreationOptions = MakeTargetCreationOptions(
        TrainDataHasWeights,
        /*dataTargetDimension*/ 1,
        TrainDataHasGroups,
        GetMetricDescriptions(*CatBoostOptions),
        /*knownModelApproxDimension*/ Nothing(),
        /*knownIsClassification*/ true,
        inputClassificationInfo,
        dataProcessingOptions.AllowConstLabel.Get()
    );

    TMaybe<ui32> knownClassCount;
    TInputClassificationInfo updatedInputClassificationInfo;

    UpdateTargetProcessingParams(
        inputClassificationInfo,
        TargetCreationOptions,

        // in fact we already have distinct values but the logic later is to assume labels must be consecutive
        // numbers if this parameter is specified
        /*knownApproxDimension*/ Nothing(),
        &CatBoostOptions->LossFunctionDescription.Get(),
        &IsRealTarget,
        &knownClassCount,
        &updatedInputClassificationInfo
    );

    TVector<NJson::TJsonValue> outputClassLabels;

    NPar::LocalExecutor().RunAdditionalThreads(CatBoostOptions->SystemOptions->NumThreads.Get() - 1);

    ClassCount = knownClassCount.GetOrElse(0);

    TSharedVector<float> convertedTargetValues = ConvertTarget(
        TConstArrayRef<TRawTarget>(&distinctTargetValues, 1),
        rawTargetType,
        IsRealTarget,
        TargetCreationOptions.IsClass,
        TargetCreationOptions.IsMultiClass,
        TargetCreationOptions.IsMultiLabel,
        /*targetBorder*/ Nothing(), // don't process floats here
        !knownClassCount,
        updatedInputClassificationInfo.ClassLabels,
        TargetCreationOptions.AllowConstLabel,
        &outputClassLabels,
        &NPar::LocalExecutor(),
        &ClassCount)[0];

    if (TargetCreationOptions.IsMultiClass) {
        if ((ClassCount == 1) && TargetCreationOptions.AllowConstLabel) {
            // Training won't work properly with single-dimensional approx for multiclass, so make a
            // 'phantom' second dimension.
            ClassCount = 2;
            MaybeAddPhantomSecondClass(&outputClassLabels);
        }

        LabelConverter.InitializeMultiClass(
            *convertedTargetValues,
            ClassCount,
            TargetCreationOptions.AllowConstLabel
        );
    } else {
        ClassCount = 2;
        LabelConverter.InitializeBinClass();
    }

    if (!outputClassLabels.empty()) {
        NJson::TJsonArray classLabelsJson;
        for (const auto& classLabel : outputClassLabels) {
            classLabelsJson.AppendValue(classLabel);
        }
        CatBoostOptionsPlainJson["class_names"] = std::move(classLabelsJson);

        CatBoostOptions->DataProcessingOptions->ClassLabels.Set(std::move(outputClassLabels));
    }
}

TVector<float> TClassTargetPreprocessor::PreprocessTargetImpl(
    const NCB::TRawTarget& targetValues,
    ERawTargetType rawTargetType
) {
    const auto& dataProcessingOptions = CatBoostOptions->DataProcessingOptions.Get();

    // not really used
    TVector<NJson::TJsonValue> outputClassLabels;

    ui32 classCount = ClassCount;

    TSharedVector<float> convertedTargetValues = ConvertTarget(
        TConstArrayRef<TRawTarget>(&targetValues, 1),
        rawTargetType,
        IsRealTarget,
        TargetCreationOptions.IsClass,
        TargetCreationOptions.IsMultiClass,
        TargetCreationOptions.IsMultiLabel,
        dataProcessingOptions.TargetBorder.IsSet() ?
            TMaybe<float>(dataProcessingOptions.TargetBorder.Get())
            : Nothing(),
        /*classCountUnknown*/ false,
        dataProcessingOptions.ClassLabels.Get(),
        dataProcessingOptions.AllowConstLabel.Get(),
        &outputClassLabels,
        &NPar::LocalExecutor(),
        &classCount)[0];

    PrepareTargetCompressed(LabelConverter, &*convertedTargetValues);
    return std::move(*convertedTargetValues);
}
