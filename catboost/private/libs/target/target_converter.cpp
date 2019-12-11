#include "target_converter.h"

#include "binarize_target.h"

#include <catboost/libs/data/loader.h> // for IsMissingValue
#include <catboost/libs/helpers/exception.h>

#include <library/threading/local_executor/local_executor.h>

#include <util/generic/algorithm.h>
#include <util/generic/array_ref.h>
#include <util/generic/cast.h>
#include <util/generic/hash.h>
#include <util/generic/hash_set.h>
#include <util/generic/string.h>
#include <util/generic/variant.h>
#include <util/generic/ymath.h>
#include <util/string/cast.h>
#include <util/string/escape.h>
#include <util/system/yassert.h>

#include <cmath>


namespace NCB {
    static float ConvertToFloatTarget(const TString& stringLabel) {
        CB_ENSURE(
            !IsMissingValue(stringLabel),
            "Missing values like \"" << EscapeC(stringLabel)
                << "\" are not supported for target"
        );
        float floatLabel;
        CB_ENSURE(
            TryFromString(stringLabel, floatLabel),
            "Target value \"" << EscapeC(stringLabel) << "\" cannot be parsed as float"
        );
        return floatLabel;
    }

    static TVector<float> ConvertRawToFloatTarget(
        const TRawTarget& rawTarget,
        NPar::TLocalExecutor* localExecutor
    ) {
        TVector<float> result;

        if (const ITypedSequencePtr<float>* floatSequence = GetIf<ITypedSequencePtr<float>>(&rawTarget)) {
            result.yresize((*floatSequence)->GetSize());
            TArrayRef<float> resultRef = result;
            size_t i = 0;
            (*floatSequence)->ForEach(
                [resultRef, &i] (float value) {
                    CB_ENSURE(!IsNan(value), "NaN values are not supported for target");
                    resultRef[i++] = value;
                }
            );
        } else {
            TConstArrayRef<TString> stringLabels = Get<TVector<TString>>(rawTarget);
            result.yresize(stringLabels.size());
            TArrayRef<float> resultRef = result;
            localExecutor->ExecRangeBlockedWithThrow(
                [stringLabels, resultRef] (int i) {
                    resultRef[i] = ConvertToFloatTarget(stringLabels[i]);
                },
                0,
                SafeIntegerCast<int>(stringLabels.size()),
                /*batchSizeOrZeroForAutoBatchSize*/ 0,
                NPar::TLocalExecutor::WAIT_COMPLETE
            );
        }

        return result;
    }

    class TCastFloatTargetConverter : public ITargetConverter {
    public:
        TCastFloatTargetConverter() = default;

        TVector<float> Process(const TRawTarget& rawTarget, NPar::TLocalExecutor* localExecutor) override {
            return ConvertRawToFloatTarget(rawTarget, localExecutor);
        }

        ui32 GetClassCount() const override {
            /*
             * this target converter is unapplicable for multiclassification but can be used for
             * binary classification (e.g. loss_function == "CrossEntropy")
             */
            return 2;
        }
    };

    class TTargetBinarizer : public ITargetConverter {
    public:
        TTargetBinarizer(float targetBorder)
            : TargetBorder(targetBorder)
        {}

        TVector<float> Process(const TRawTarget& rawTarget, NPar::TLocalExecutor* localExecutor) override {
            TVector<float> floatTarget = ConvertRawToFloatTarget(rawTarget, localExecutor);
            PrepareTargetBinary(floatTarget, TargetBorder, &floatTarget);
            return floatTarget;
        }

        ui32 GetClassCount() const override {
            return 2;
        }

    private:
        float TargetBorder;
    };


    class TNumericClassTargetConverter : public ITargetConverter {
    public:
        TNumericClassTargetConverter(ui32 classCount)
            : ClassCount(static_cast<float>(classCount))
        {}

        TVector<float> Process(const TRawTarget& rawTarget, NPar::TLocalExecutor* localExecutor) override {
            TVector<float> result = ConvertRawToFloatTarget(rawTarget, localExecutor);

            TArrayRef<float> resultRef = result;
            localExecutor->ExecRangeBlockedWithThrow(
                [resultRef, this] (int i) {
                    CheckIsValidClassIdx(resultRef[i]);
                },
                0,
                SafeIntegerCast<int>(result.size()),
                /*batchSizeOrZeroForAutoBatchSize*/ 0,
                NPar::TLocalExecutor::WAIT_COMPLETE
            );

            return result;
        }

        ui32 GetClassCount() const override {
            return (ui32)ClassCount;
        }

    private:
        inline void CheckIsValidClassIdx(float classIdx) const {
            float intPart;
            CB_ENSURE(
                std::modf(classIdx, &intPart) == 0.0f,
                "Value in target (" << classIdx << ") is not expected class index"
            );
            CB_ENSURE(
                classIdx >= 0.0f,
                "Value in target (" << classIdx << ") is not expected class index"
            );
            CB_ENSURE(
                classIdx < ClassCount,
                "Value in target (" << classIdx << ") is greater than specified class count"
            );
        }

    private:
        // type is float to avoid casting because target is a vector of floats
        float ClassCount;
    };


    class TUseClassNamesTargetConverter : public ITargetConverter {
    public:
        TUseClassNamesTargetConverter(const TVector<TString>& inputClassNames) {
            float classIdx = 0;
            for (const auto& name : inputClassNames) {
                StringLabelToClass.emplace(name, classIdx++);
            }
        }

        TVector<float> Process(const TRawTarget& rawTarget, NPar::TLocalExecutor* localExecutor) override {
            TVector<float> result;

            if (const ITypedSequencePtr<float>* typedSequence = GetIf<ITypedSequencePtr<float>>(&rawTarget)) {
                UpdateFloatLabelToClass();

                result.yresize((*typedSequence)->GetSize());
                TArrayRef<float> resultRef = result;
                size_t i = 0;
                (*typedSequence)->ForEach(
                    [this, resultRef, &i] (float srcLabel) {
                        const auto it = FloatLabelToClass.find(srcLabel);
                        CB_ENSURE(it != FloatLabelToClass.end(), "Unknown class name: \"" << srcLabel << '"');
                        resultRef[i++] = it->second;
                    }
                );
            } else {
                TConstArrayRef<TString> stringLabels = Get<TVector<TString>>(rawTarget);
                result.yresize(stringLabels.size());
                TArrayRef<float> resultRef = result;
                localExecutor->ExecRangeBlockedWithThrow(
                    [this, resultRef, stringLabels] (int i) {
                        const auto it = StringLabelToClass.find(stringLabels[i]);
                        CB_ENSURE(
                            it != StringLabelToClass.end(),
                            "Unknown class name: \"" << EscapeC(stringLabels[i]) << '"'
                        );
                        resultRef[i] = it->second;
                    },
                    0,
                    SafeIntegerCast<int>(stringLabels.size()),
                    /*batchSizeOrZeroForAutoBatchSize*/ 0,
                    NPar::TLocalExecutor::WAIT_COMPLETE
                );
            }

            return result;
        }

        ui32 GetClassCount() const override {
            return SafeIntegerCast<ui32>(StringLabelToClass.size());
        }

    private:
        void UpdateFloatLabelToClass() {
            if (FloatLabelToClass.empty()) {
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

    private:
        // which map is used depends on source target data type
        // dst type is float to avoid casting because target is a vector of floats
        THashMap<TString, float> StringLabelToClass;
        THashMap<float, float> FloatLabelToClass;
    };

    class TMakeClassNamesTargetConverter : public ITargetConverter {
    public:
        TMakeClassNamesTargetConverter(bool isMultiClass)
            : IsMultiClass(isMultiClass)
        {}

        TVector<float> Process(const TRawTarget& rawTarget, NPar::TLocalExecutor* localExecutor) override {
            TVector<float> result;
            Visit(
                [&] (const auto& value) { result = ProcessMakeClassNamesImpl(value, localExecutor); },
                rawTarget
            );
            return result;
        }

        ui32 GetClassCount() const override {
            const ui32 classCount = SafeIntegerCast<ui32>(
                !StringLabelToClass.empty() ? StringLabelToClass.size() : FloatLabelToClass.size()
            );
            Y_ASSERT(classCount > 1);
            return classCount;
        }

        TMaybe<TVector<TString>> GetClassNames() override {
            UpdateStringLabelToClass();

            TVector<TString> result;
            result.yresize(StringLabelToClass.ysize());
            for (const auto& [stringLabel, classIdx] : StringLabelToClass) {
                result[static_cast<size_t>(classIdx)] = stringLabel;
            }

            return MakeMaybe<TVector<TString>>(std::move(result));
        }

    private:
        void CheckUniqueLabelsSize(size_t size) const {
            CB_ENSURE(size > 1, "Target contains only one unique value");
            CB_ENSURE(
                IsMultiClass || (size == 2),
                "Target with classes must contain only 2 unique values for binary classification"
            );
        }

        TVector<float> ProcessMakeClassNamesImpl(const ITypedSequencePtr<float>& labels,
                                                 NPar::TLocalExecutor* localExecutor) {
            TVector<float> targets = ToVector(*labels);

            THashSet<float> uniqueLabelsSet;
            for (float value : targets) {
                CB_ENSURE(!IsNan(value), "NaN values are not supported for target");
                uniqueLabelsSet.insert(value);
            }

            CheckUniqueLabelsSize(uniqueLabelsSet.size());

            TVector<float> uniqueLabels(uniqueLabelsSet.begin(), uniqueLabelsSet.end());
            Sort(uniqueLabels);

            CB_ENSURE(FloatLabelToClass.empty(), "ProcessMakeClassNames: label-to-class map must be empty before label converting.");
            float classIdx = 0;
            for (auto label: uniqueLabels) {
                FloatLabelToClass.emplace(label, classIdx++);
            }

            TArrayRef<float> targetsRef = targets;
            NPar::ParallelFor(
                *localExecutor,
                0,
                SafeIntegerCast<ui32>(targets.size()),
                [targetsRef, this] (int i) {
                    targetsRef[i] = FloatLabelToClass[targetsRef[i]];
                }
            );
            return targets;
        }

        TVector<float> ProcessMakeClassNamesImpl(TConstArrayRef<TString> labels,
                                                 NPar::TLocalExecutor* localExecutor) {
            THashSet<TString> uniqueLabelsSet(labels.begin(), labels.end());
            CheckUniqueLabelsSize(uniqueLabelsSet.size());

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
            float classIdx = 0;
            for (const auto& label: uniqueLabels) {
                StringLabelToClass.emplace(label, classIdx++);
            }
            TVector<float> targets;
            targets.yresize(labels.size());
            TArrayRef<float> targetsRef = targets;
            NPar::ParallelFor(
                *localExecutor,
                0,
                SafeIntegerCast<ui32>(targets.size()),
                [targetsRef, labels, this] (int i) {
                    targetsRef[i] = StringLabelToClass[labels[i]];
                }
            );
            return targets;
        }

        void UpdateStringLabelToClass() {
            if (StringLabelToClass.empty()) {
                CB_ENSURE(!FloatLabelToClass.empty(), "Label-to-class mapping must be calced before setting class names.");
                for (const auto& [floatLabel, classIdx] : FloatLabelToClass) {
                    StringLabelToClass.emplace(ToString(floatLabel), classIdx);
                }
            }
        }

    private:
        bool IsMultiClass;

        // which map is used depends on source target data type
        // dst type is float to avoid casting because target is a vector of floats
        THashMap<TString, float> StringLabelToClass;
        THashMap<float, float> FloatLabelToClass;
    };


    THolder<ITargetConverter> MakeTargetConverter(bool isRealTarget,
                                                  bool isClass,
                                                  bool isMultiClass,
                                                  TMaybe<float> targetBorder,
                                                  TMaybe<ui32> classCount,
                                                  const TVector<TString>& inputClassNames) {

        CB_ENSURE_INTERNAL(!isMultiClass || isClass, "isMultiClass is true, but isClass is false");

        if (isRealTarget) {
            CB_ENSURE(!isMultiClass, "Converted real target is incompatible with Multiclass");
            CB_ENSURE(!targetBorder.Defined(), "Converted real target is incompatible with targetBorder");
            CB_ENSURE(
                !classCount.Defined() || (*classCount == 2),
                "Converted real target is incompatible with class count not equal to 2"
            );
            CB_ENSURE(
                inputClassNames.empty(),
                "Converted real target is incompatible with specifying class names"
            );

            return MakeHolder<TCastFloatTargetConverter>();
        } else {
            CB_ENSURE_INTERNAL(isClass, "isRealTarget is false, but isClass is false");
        }

        if (targetBorder.Defined()) {
            CB_ENSURE(
                isClass && !isMultiClass,
                "targetBorder should be specified only for binary classification problems"
            );
            CB_ENSURE(
                !classCount.Defined() || (*classCount == 2),
                "Specifying target border is incompatible with class count not equal to 2"
            );
            CB_ENSURE(
                inputClassNames.empty(),
                "Specifying target border is incompatible with specifying class names"
            );

            return MakeHolder<TTargetBinarizer>(*targetBorder);
        }
        if (!inputClassNames.empty()) {
            CB_ENSURE(
                isClass,
                "classNames should be specified only for classification problems"
            );
            CB_ENSURE(
                isMultiClass || (inputClassNames.size() == 2),
                "binary classification problem, but class names count is not equal to 2"
            );
            CB_ENSURE(
                !classCount.Defined() || (size_t(*classCount) == inputClassNames.size()),
                "both classCount and classNames specified and length of classNames is not equal to classCount"
            );

            return MakeHolder<TUseClassNamesTargetConverter>(inputClassNames);
        }
        if (classCount.Defined()) {
            CB_ENSURE(
                isMultiClass,
                "classCount should be specified only for multi classification problems"
            );

            return MakeHolder<TNumericClassTargetConverter>(*classCount);
        }


        return MakeHolder<TMakeClassNamesTargetConverter>(isMultiClass);
    }

} // NCB
