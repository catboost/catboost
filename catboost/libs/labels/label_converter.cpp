#include "label_converter.h"

#include <catboost/libs/logging/logging.h>
#include <catboost/libs/options/json_helper.h>
#include <catboost/libs/options/multiclass_label_options.h>
#include <catboost/libs/options/option.h>

#include <util/generic/algorithm.h>
#include <util/string/builder.h>

static THashMap<float, int> CalcLabelToClassMap(TVector<float> targets, int classesCount) {
    SortUnique(targets);
    THashMap<float, int> labels;
    if (classesCount != 0) {  // classes-count or class-names are set
        CB_ENSURE(AllOf(targets, [&classesCount](float x) { return int(x) == x && x >= 0 && x < classesCount; }),
            "If classes count is specified each target label should be nonnegative integer in [0,..,classes_count - 1].");

        if (classesCount > targets.ysize()) {
            CATBOOST_WARNING_LOG << "Found only " << targets.ysize() <<
                " unique classes but defined " << classesCount
                << " classes probably something is wrong with data." << Endl;
        }
    }

    labels.reserve(targets.size());
    int id = 0;
    for (auto target : targets) {
        labels.emplace(target, id++);
    }

    return labels;
}

bool TLabelConverter::operator==(const TLabelConverter& rhs) const {
    if (Initialized != rhs.Initialized) {
        return false;
    }
    if (!Initialized) {
        return true;
    }

    return (LabelToClass == rhs.LabelToClass) &&
        (ClassToLabel == rhs.ClassToLabel) &&
        (ClassesCount == rhs.ClassesCount);
}

void TLabelConverter::Initialize(int approxDimension) {
    CB_ENSURE(!Initialized, "Can't initialize initialized object of TLabelConverter");

    ClassesCount = approxDimension;

    ClassToLabel.resize(approxDimension);

    for (int id = 0; id < approxDimension; ++id) {
        ClassToLabel[id] = id;
    }

    LabelToClass = CalcLabelToClassMap(ClassToLabel, 0);

    Initialized = true;
}


void TLabelConverter::Initialize(const TString& multiclassLabelParams) {
    CB_ENSURE(!Initialized, "Can't initialize initialized object of TLabelConverter");
    TMulticlassLabelOptions multiclassOptions;
    multiclassOptions.Load(ReadTJsonValue(multiclassLabelParams));

    int classesCount = multiclassOptions.ClassesCount.Get();
    const auto& classNames = multiclassOptions.ClassNames.Get();

    ClassesCount = GetClassesCount(classesCount, classNames);

    ClassToLabel = multiclassOptions.ClassToLabel.Get();
    LabelToClass = CalcLabelToClassMap(ClassToLabel, ClassesCount);

    ClassesCount = Max(ClassesCount, ClassToLabel.ysize());

    Initialized = true;
}

void TLabelConverter::Initialize(TConstArrayRef<float> targets, int classesCount) {
    CB_ENSURE(!Initialized, "Can't initialize initialized object of TLabelConverter");

    TVector<float> targetsCopy(targets.begin(), targets.end());
    LabelToClass = CalcLabelToClassMap(std::move(targetsCopy), classesCount);
    ClassesCount = Max(classesCount, LabelToClass.ysize());

    ClassToLabel.resize(LabelToClass.ysize());
    for (const auto& keyValue : LabelToClass) {
        ClassToLabel[keyValue.second] = keyValue.first;
    }
    Initialized = true;
}

int TLabelConverter::GetApproxDimension() const {
    CB_ENSURE(Initialized, "Can't use uninitialized object of TLabelConverter");
    return LabelToClass.ysize();
}

int TLabelConverter::GetClassIdx(float label) const {
    CB_ENSURE(Initialized, "Can't use uninitialized object of TLabelConverter");
    const auto it = LabelToClass.find(label);
    return it == LabelToClass.cend() ? 0 : it->second;
}

void TLabelConverter::ValidateLabels(TConstArrayRef<float> labels) const {
    CB_ENSURE(Initialized, "Can't use uninitialized object of TLabelConverter");

    THashSet<float> missingLabels;

    for (const auto& label : labels) {
        if (!LabelToClass.contains(label)) {
            if (ClassesCount > 0 && int(label) == label && label >= 0 && label < ClassesCount) {
                missingLabels.emplace(label);
            } else {
                CB_ENSURE(false, "Label " << label
                    << " is bad label and not contained in train set.");
            }
        }
    }

    if (!missingLabels.empty()) {
        TStringBuilder warningStrBuilder;
        bool isFirstLabel = true;

        warningStrBuilder << "Labels ";
        for (auto label : missingLabels) {
            warningStrBuilder << (isFirstLabel ? "": ", ") << label;
            isFirstLabel = false;
        }

        CB_ENSURE(!ClassToLabel.empty(), "ClassToLabel mapping must be not empty.");
        warningStrBuilder << " aren't contained in train set but still valid "
                      << "and will be processed the same as label " << ClassToLabel[0] << ".";
        CATBOOST_WARNING_LOG << warningStrBuilder << Endl;
    }
}

bool TLabelConverter::IsInitialized() const {
    return Initialized;
}

TString TLabelConverter::SerializeMulticlassParams(int classesCount, const TVector<TString>& classNames) const {
    CB_ENSURE(Initialized, "Can't use uninitialized object of TLabelConverter");
    TMulticlassLabelOptions multiclassLabelOptions;
    multiclassLabelOptions.ClassToLabel = ClassToLabel;
    multiclassLabelOptions.ClassesCount = classesCount;
    multiclassLabelOptions.ClassNames = classNames;
    NJson::TJsonValue json;

    multiclassLabelOptions.Save(&json);
    return ToString(json);
}

TVector<float> TLabelConverter::GetClassLabels() const {
    return ClassToLabel;
}

void PrepareTargetCompressed(const TLabelConverter& labelConverter, TVector<float>* labels) {
    CB_ENSURE(labelConverter.IsInitialized(), "Label converter isn't built.");
    labelConverter.ValidateLabels(*labels);
    for (auto& label : *labels) {
        label = labelConverter.GetClassIdx(label);
    }
}

int GetClassesCount(int classesCount, const TVector<TString>& classNames) {
    if (classNames.empty() || classesCount == 0) {
        return Max(classNames.ysize(), classesCount);
    }

    CB_ENSURE(classesCount == classNames.ysize(),
              "classes-count " << classesCount << " must be equal to size of class-names " << classNames.ysize() << "if both are specified.");
    return classesCount;
}
