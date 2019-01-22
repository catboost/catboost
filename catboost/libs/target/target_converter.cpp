#include "target_converter.h"

#include <catboost/libs/data_new/loader.h> // for IsNanValue
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/options/enum_helpers.h>
#include <catboost/libs/options/metric_options.h>

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
            CB_ENSURE_INTERNAL(IsClassTarget, "Make class names is valid only for class targets");
            CB_ENSURE(outputClassNames != nullptr,
                      "Cannot initialize target converter with null class names pointer and MakeClassNames target policy.");
        }

        if (TargetPolicy == EConvertTargetPolicy::UseClassNames) {
            CB_ENSURE_INTERNAL(IsClassTarget, "Use class names is valid only for class targets");
            CB_ENSURE(!InputClassNames.empty(), "Cannot use empty class names for pool reading.");
            int id = 0;
            for (const auto& name : InputClassNames) {
                LabelToClass.emplace(name, id++);
            }
        }
    }


    float TTargetConverter::ConvertLabel(const TStringBuf& label) {
        switch (TargetPolicy) {
            case EConvertTargetPolicy::CastFloat: {
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
            case EConvertTargetPolicy::UseClassNames: {
                const auto it = LabelToClass.find(label);
                if (it != LabelToClass.end()) {
                    return static_cast<float>(it->second);
                }
                ythrow TCatBoostException() << "Unknown class name: \"" << EscapeC(label) << '"';
            }
            default: {
                ythrow TCatBoostException() <<
                    "Cannot convert label online if convert target policy is not CastFloat or UseClassNames.";
            }
        }
    }


    float TTargetConverter::ProcessLabel(const TString& label) {
        THashMap<TString, int>::insert_ctx ctx = nullptr;
        const auto& it = LabelToClass.find(label, ctx);

        if (it == LabelToClass.end()) {
            const int classIdx = LabelToClass.ysize();
            LabelToClass.emplace_direct(ctx, label, classIdx);
            return static_cast<float>(classIdx);
        } else {
            return static_cast<float>(it->second);
        }
    }

    TVector<float> TTargetConverter::PostprocessLabels(TConstArrayRef<TString> labels) {
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
        CB_ENSURE(LabelToClass.empty(), "PostprocessLabels: label-to-class map must be empty before label converting.");
        for (const auto& label: uniqueLabels) {
            ProcessLabel(label);
        }
        TVector<float> targets;
        targets.reserve(labels.size());
        for (const auto& label : labels) {
            targets.push_back(ProcessLabel(label));
        }
        return targets;
    }

    void TTargetConverter::SetOutputClassNames() const {
        CB_ENSURE(OutputClassNames != nullptr && OutputClassNames->empty(), "Cannot reset user-defined class names.");
        CB_ENSURE(TargetPolicy == EConvertTargetPolicy::MakeClassNames,
                  "Cannot set class names without MakeClassNames target policy.");
        CB_ENSURE(!LabelToClass.empty(), "Label-to-class mapping must be calced before setting class names.");
        OutputClassNames->resize(LabelToClass.ysize());
        for (const auto& keyValue : LabelToClass) {
            (*OutputClassNames)[keyValue.second] = keyValue.first;
        }
    }

    EConvertTargetPolicy TTargetConverter::GetTargetPolicy() const {
        return TargetPolicy;
    }

    const TVector<TString>& TTargetConverter::GetInputClassNames() const {
        return InputClassNames;
    }

    ui32 TTargetConverter::GetClassCount() const {
        CB_ENSURE_INTERNAL(IsClassTarget, "GetClassCount is valid only for class targets");
        switch (TargetPolicy) {
            case EConvertTargetPolicy::CastFloat:
                return IsMultiClassTarget ? SafeIntegerCast<ui32>(UniqueLabels.size()) : ui32(2);
            case EConvertTargetPolicy::UseClassNames:
            case EConvertTargetPolicy::MakeClassNames:
                return SafeIntegerCast<ui32>(LabelToClass.size());
        }
        Y_FAIL("should be unreachable");
    }

    TTargetConverter MakeTargetConverter(bool isClass,
                                         bool isMultiClass,
                                         bool classesCountUnknown,
                                         TVector<TString>* classNames) {
        EConvertTargetPolicy targetPolicy = EConvertTargetPolicy::CastFloat;

        if (!classNames->empty()) {
            targetPolicy = EConvertTargetPolicy::UseClassNames;
        } else {
            if (isMultiClass && classesCountUnknown) {
                targetPolicy = EConvertTargetPolicy::MakeClassNames;
            }
        }
        return NCB::TTargetConverter(isClass, isMultiClass, targetPolicy, *classNames, classNames);
    }

} // NCB
