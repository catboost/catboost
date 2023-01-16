%{
#include <catboost/spark/catboost4j-spark/core/src/native_impl/target.h>
%}

%include <bindings/swiglib/stroka.swg>

%include "catboost_enums.i"
%include "primitive_arrays.i"
%include "tvector.i"


%catches(yexception) TClassTargetPreprocessor::TClassTargetPreprocessor(
    const TString& plainJsonParamsAsString,
    NCB::ERawTargetType rawTargetType,
    bool trainDataHasWeights,
    bool trainDataHasGroups
);

%catches(yexception) TClassTargetPreprocessor::IsNeedToProcessDistinctTargetValues() const;

%catches(yexception) TClassTargetPreprocessor::ProcessDistinctFloatTargetValues(
    TConstArrayRef<float> distinctTargetValues
);

%catches(yexception) TClassTargetPreprocessor::ProcessDistinctIntTargetValues(
    TConstArrayRef<i32> distinctTargetValues
);

%catches(yexception) TClassTargetPreprocessor::ProcessDistinctStringTargetValues(
    const TVector<TString>& distinctTargetValues
);

%catches(yexception) TClassTargetPreprocessor::GetLossFunction() const;

%catches(yexception) TClassTargetPreprocessor::GetUpdatedCatBoostOptionsJsonAsString() const;

%catches(yexception) TClassTargetPreprocessor::GetSerializedLabelConverter();

%catches(yexception) TClassTargetPreprocessor::PreprocessFloatTarget(TConstArrayRef<float> targetValues);

%catches(yexception) TClassTargetPreprocessor::PreprocessIntTarget(TConstArrayRef<i32> targetValues);

%catches(yexception) TClassTargetPreprocessor::PreprocessStringTarget(const TVector<TString>& targetValues);

%include "target.h"
