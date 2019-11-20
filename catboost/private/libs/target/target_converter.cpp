#include "target_converter.h"

#include <catboost/libs/data/loader.h> // for IsNanValue
#include <catboost/libs/helpers/exception.h>
#include <catboost/private/libs/options/enum_helpers.h>
#include <catboost/private/libs/options/metric_options.h>

#include <library/threading/local_executor/local_executor.h>

#include <util/generic/algorithm.h>
#include <util/generic/cast.h>
#include <util/string/escape.h>


namespace NCB {

    TTargetConverter::TTargetConverter(const bool isClassTarget,
                                       const bool isMultiClassTarget,
                                       const EConvertTargetPolicy readingPoolTargetPolicy,
                                       const TVector<TString>& inputClassNames,
                                       TVector<TString>* const outputClassNames)
        : IsClassTarget(isClassTarget)
        , IsMultiClassTarget(isMultiClassTarget)
        , TargetPolicy(readingPoolTargetPolicy)
        , InputClassNames(inputClassNames)
        , OutputClassNames(outputClassNames)
    {
        if (TargetPolicy == EConvertTargetPolicy::MakeClassNames) {
            CB_ENSURE_INTERNAL(IsClassTarget, "Make class names is valid only for classification objectives.");
            CB_ENSURE(outputClassNames != nullptr,
                      "Cannot initialize target converter with null class names pointer and MakeClassNames target policy.");
        }

        if (TargetPolicy == EConvertTargetPolicy::UseClassNames) {
            CB_ENSURE_INTERNAL(IsClassTarget, "Use class names is valid only for classification objectives.");
            CB_ENSURE(!InputClassNames.empty(), "Cannot use empty class names for pool reading.");
            int id = 0;
            for (const auto& name : InputClassNames) {
                StringLabelToClass.emplace(name, id++);
            }
        }
    }

    TVector<float> TTargetConverter::Process(const TRawTarget& labels,
                                             NPar::TLocalExecutor* localExecutor) {
        switch (TargetPolicy) {
            case EConvertTargetPolicy::CastFloat:
                return ProcessCastFloat(labels, localExecutor);
            case EConvertTargetPolicy::UseClassNames:
                return ProcessUseClassNames(labels, localExecutor);
            case EConvertTargetPolicy::MakeClassNames:
                return ProcessMakeClassNames(labels, localExecutor);
        }
    }

    ui32 TTargetConverter::GetClassCount() const {
        CB_ENSURE_INTERNAL(IsClassTarget, "GetClassCount is valid only for class targets");
        switch (TargetPolicy) {
            case EConvertTargetPolicy::CastFloat:
                return IsMultiClassTarget ? SafeIntegerCast<ui32>(UniqueLabels.size()) : ui32(2);
            case EConvertTargetPolicy::UseClassNames:
            case EConvertTargetPolicy::MakeClassNames:
                if (!StringLabelToClass.empty()) {
                    return SafeIntegerCast<ui32>(StringLabelToClass.size());
                } else {
                    return SafeIntegerCast<ui32>(FloatLabelToClass.size());
                }
        }
        Y_FAIL("should be unreachable");
    }

    float TTargetConverter::CastFloatLabel(float label) {
        CB_ENSURE(!IsNan(label), "NaN values are not supported for target");
        if (IsMultiClassTarget) {
            UniqueLabels.insert(label);
        }
        return label;
    }

    float TTargetConverter::CastFloatLabel(TStringBuf label) {
         CB_ENSURE(
            !IsMissingValue(label),
            "Missing values like \"" << EscapeC(label) << "\" are not supported for target");
        float floatLabel;
        CB_ENSURE(
            TryFromString(label, floatLabel),
            "Target value \"" << EscapeC(label) << "\" cannot be parsed as float"
        );
        if (IsMultiClassTarget) {
            UniqueLabels.insert(floatLabel);
        }
        return floatLabel;
    }

    TVector<float> TTargetConverter::ProcessCastFloat(const TRawTarget& labels,
                                                      NPar::TLocalExecutor* localExecutor) {
        TVector<float> result;

        if (const ITypedSequencePtr<float>* typedSequence = GetIf<ITypedSequencePtr<float>>(&labels)) {
            result.yresize((*typedSequence)->GetSize());
            TArrayRef<float> resultRef = result;
            size_t i = 0;
            (*typedSequence)->ForEach(
                [this, resultRef, &i] (float value) { resultRef[i++] = CastFloatLabel(value); }
            );
        } else {
            TConstArrayRef<TString> stringLabels = Get<TVector<TString>>(labels);
            result.yresize(stringLabels.size());
            if (IsMultiClassTarget) {
                // can't use parallel processing because of UniqueLabels update
                for (auto i : xrange(stringLabels.size())) {
                    result[i] = CastFloatLabel(stringLabels[i]);
                }
            } else {
                TArrayRef<float> resultRef = result;
                localExecutor->ExecRangeBlockedWithThrow(
                    [&, resultRef] (int i) { resultRef[i] = CastFloatLabel(stringLabels[i]); },
                    0,
                    SafeIntegerCast<int>(stringLabels.size()),
                    /*batchSizeOrZeroForAutoBatchSize*/ 0,
                    NPar::TLocalExecutor::WAIT_COMPLETE
                );
            }
        }

        return result;
    }

    TVector<float> TTargetConverter::ProcessUseClassNames(const TRawTarget& labels,
                                                          NPar::TLocalExecutor* localExecutor) {
        TVector<float> result;

        if (const ITypedSequencePtr<float>* typedSequence = GetIf<ITypedSequencePtr<float>>(&labels)) {
            UpdateFloatLabelToClass();

            result.yresize((*typedSequence)->GetSize());
            TArrayRef<float> resultRef = result;
            size_t i = 0;
            (*typedSequence)->ForEach(
                [this, resultRef, &i] (float srcLabel) {
                    const auto it = FloatLabelToClass.find(srcLabel);
                    CB_ENSURE(it != FloatLabelToClass.end(), "Unknown class name: \"" << srcLabel << '"');
                    resultRef[i++] = (float)it->second;
                }
            );
        } else {
            UpdateStringLabelToClass();

            TConstArrayRef<TString> stringLabels = Get<TVector<TString>>(labels);
            result.yresize(stringLabels.size());
            TArrayRef<float> resultRef = result;
            localExecutor->ExecRangeBlockedWithThrow(
                [this, resultRef, stringLabels] (int i) {
                    const auto it = StringLabelToClass.find(stringLabels[i]);
                    CB_ENSURE(
                        it != StringLabelToClass.end(),
                        "Unknown class name: \"" << EscapeC(stringLabels[i]) << '"'
                    );
                    resultRef[i] = (float)it->second;
                },
                0,
                SafeIntegerCast<int>(stringLabels.size()),
                /*batchSizeOrZeroForAutoBatchSize*/ 0,
                NPar::TLocalExecutor::WAIT_COMPLETE
            );
        }

        return result;
    }


    TVector<float> TTargetConverter::ProcessMakeClassNames(const TRawTarget& labels,
                                                           NPar::TLocalExecutor* localExecutor) {
        TVector<float> result;
        Visit([&] (const auto& value) { result = ProcessMakeClassNamesImpl(value, localExecutor); }, labels);
        if (OutputClassNames) {
            SetOutputClassNames();
        }
        return result;
    }


    TVector<float> TTargetConverter::ProcessMakeClassNamesImpl(const ITypedSequencePtr<float>& labels,
                                                               NPar::TLocalExecutor* localExecutor) {
        CB_ENSURE(TargetPolicy == EConvertTargetPolicy::MakeClassNames,
                  "Cannot postprocess labels without MakeClassNames target policy.");

        TVector<float> targets = ToVector(*labels);
        THashSet<float> uniqueLabelsSet(targets.begin(), targets.end());
        TVector<float> uniqueLabels(uniqueLabelsSet.begin(), uniqueLabelsSet.end());
        Sort(uniqueLabels);

        CB_ENSURE(FloatLabelToClass.empty(), "ProcessMakeClassNames: label-to-class map must be empty before label converting.");
        int i = 0;
        for (auto label: uniqueLabels) {
            FloatLabelToClass.emplace(label, i++);
        }
        TArrayRef<float> targetsRef = targets;
        NPar::ParallelFor(
            *localExecutor,
            0,
            SafeIntegerCast<ui32>(targets.size()),
            [targetsRef, this] (int i) {
                targetsRef[i] = (float)FloatLabelToClass[targetsRef[i]];
            }
        );
        return targets;
    }

    TVector<float> TTargetConverter::ProcessMakeClassNamesImpl(TConstArrayRef<TString> labels,
                                                               NPar::TLocalExecutor* localExecutor) {
        CB_ENSURE(TargetPolicy == EConvertTargetPolicy::MakeClassNames,
                  "Cannot postprocess labels without MakeClassNames target policy.");
        THashSet<TString> uniqueLabelsSet(labels.begin(), labels.end());
        TVector<TString> uniqueLabels(uniqueLabelsSet.begin(), uniqueLabelsSet.end());
        // Kind of heuristic for proper ordering class names if they all are numeric
        if (AllOf(uniqueLabels, [](const TString& label) -> bool {
            float tmp;
            return TryFromString<float>(label, tmp);
        })) {
            Sort(uniqueLabels, [](const TString& label1, const TString& label2) {
                return FromString<float>(label1) < FromString<float>(label2);
            });
        } else {
            Sort(uniqueLabels);
        }
        CB_ENSURE(StringLabelToClass.empty(), "ProcessMakeClassNames: label-to-class map must be empty before label converting.");
        int i = 0;
        for (const auto& label: uniqueLabels) {
            StringLabelToClass.emplace(label, i++);
        }
        TVector<float> targets;
        targets.yresize(labels.size());
        TArrayRef<float> targetsRef = targets;
        NPar::ParallelFor(
            *localExecutor,
            0,
            SafeIntegerCast<ui32>(targets.size()),
            [targetsRef, labels, this] (int i) {
                targetsRef[i] = (float)StringLabelToClass[labels[i]];
            }
        );
        return targets;
    }


    void TTargetConverter::UpdateStringLabelToClass() {
        if (StringLabelToClass.empty()) {
            CB_ENSURE(!FloatLabelToClass.empty(), "Label-to-class mapping must be calced before setting class names.");
            for (const auto& [floatLabel, classIdx] : FloatLabelToClass) {
                StringLabelToClass.emplace(ToString(floatLabel), classIdx);
            }
        }
    }

    void TTargetConverter::UpdateFloatLabelToClass() {
        if (FloatLabelToClass.empty()) {
            CB_ENSURE(!StringLabelToClass.empty(), "Label-to-class mapping must be calced before using class names.");
            for (const auto& [stringLabel, classIdx] : StringLabelToClass) {
                float floatLabel;
                CB_ENSURE(
                    TryFromString<float>(stringLabel, floatLabel),
                    "Not all class names are numeric, but specified target data is"
                );
                FloatLabelToClass.emplace(floatLabel, classIdx);
            }
        }
    }

    void TTargetConverter::SetOutputClassNames() {
        CB_ENSURE(OutputClassNames != nullptr && OutputClassNames->empty(), "Cannot reset user-defined class names.");
        CB_ENSURE(TargetPolicy == EConvertTargetPolicy::MakeClassNames,
                  "Cannot set class names without MakeClassNames target policy.");
        UpdateStringLabelToClass();
        OutputClassNames->resize(StringLabelToClass.ysize());
        for (const auto& keyValue : StringLabelToClass) {
            (*OutputClassNames)[keyValue.second] = keyValue.first;
        }
    }


    TTargetConverter MakeTargetConverter(bool isClass,
                                         bool isMultiClass,
                                         bool classesCountUnknown,
                                         const TVector<TString>& inputClassNames,
                                         TVector<TString>* outputClassNames) {
        EConvertTargetPolicy targetPolicy = EConvertTargetPolicy::CastFloat;

        if (!inputClassNames.empty()) {
            targetPolicy = EConvertTargetPolicy::UseClassNames;
        } else {
            if (isMultiClass && classesCountUnknown) {
                targetPolicy = EConvertTargetPolicy::MakeClassNames;
            }
        }
        return NCB::TTargetConverter(isClass, isMultiClass, targetPolicy, inputClassNames, outputClassNames);
    }

} // NCB
