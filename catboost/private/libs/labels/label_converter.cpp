#include "label_converter.h"

#include "helpers.h"

#include <catboost/libs/logging/logging.h>
#include <catboost/private/libs/options/json_helper.h>
#include <catboost/private/libs/options/class_label_options.h>
#include <catboost/private/libs/options/option.h>

#include <library/cpp/json/json_value.h>

#include <util/generic/algorithm.h>
#include <util/generic/utility.h>
#include <util/string/join.h>

static THashMap<float, int> CalcLabelToClassMap(TVector<float> targets, int classesCount, bool allowConstLabel = false) {
    SortUnique(targets);
    THashMap<float, int> labels;
    if (classesCount != 0) {  // classes-count or class-names are set
        CB_ENSURE(AllOf(targets, [&classesCount](float x) { return int(x) == x && x >= 0 && x < classesCount; }),
            "If classes count is specified each target label should be nonnegative integer in [0,..,classes_count - 1].");

        if ((classesCount > targets.ysize()) && !(allowConstLabel && (targets.ysize() == 1))) {
            CATBOOST_WARNING_LOG << "Found only " << targets.ysize() << " unique classes in the data"
                << ", but have defined " << classesCount << " classes."
                << " Probably something is wrong with data." << Endl;
        }
    }

    if (allowConstLabel && (targets.ysize() == 1)) {
        // make at least 2 labels/classes otherwise model won't apply
        if (targets.front() != 0.0f) {
            labels.emplace(0.0f, 0);
            labels.emplace(targets.front(), 1);
        } else {
            labels.emplace(targets.front(), 0);
            labels.emplace(1.0f, 1);
        }
    } else {
        labels.reserve(targets.size());
        int id = 0;
        for (auto target : targets) {
            labels.emplace(target, id++);
        }
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

    return (MultiClass == rhs.MultiClass) &&
        (LabelToClass == rhs.LabelToClass) &&
        (ClassToLabel == rhs.ClassToLabel) &&
        (ClassesCount == rhs.ClassesCount);
}

void TLabelConverter::Initialize(bool isMultiClass, const TString& classLabelParams) {
    CB_ENSURE(!Initialized, "Can't initialize initialized object of TLabelConverter");

    MultiClass = isMultiClass;

    TClassLabelOptions classOptions;
    classOptions.Load(ReadTJsonValue(classLabelParams));

    int classesCount = classOptions.ClassesCount.Get();

    const auto& classLabels = classOptions.ClassLabels.Get();

    ClassesCount = GetClassesCount(classesCount, classLabels);

    ClassToLabel = classOptions.ClassToLabel.Get();
    LabelToClass = CalcLabelToClassMap(ClassToLabel, ClassesCount);

    ClassesCount = Max(ClassesCount, ClassToLabel.ysize());

    CB_ENSURE(MultiClass || (ClassesCount == 2), "Class count is not 2 for binary classification");

    Initialized = true;
}

void TLabelConverter::InitializeBinClass() {
    CB_ENSURE(!Initialized, "Can't initialize initialized object of TLabelConverter");

    MultiClass = false;

    ClassesCount = 2;
    ClassToLabel = {0.0f, 1.0f};

    LabelToClass[0.0f] = 0;
    LabelToClass[1.0f] = 1;

    Initialized = true;
}

void TLabelConverter::InitializeMultiClass(int approxDimension) {
    CB_ENSURE(!Initialized, "Can't initialize initialized object of TLabelConverter");

    MultiClass = true;

    ClassesCount = approxDimension;

    ClassToLabel.resize(approxDimension);

    for (int id = 0; id < approxDimension; ++id) {
        ClassToLabel[id] = id;
    }

    LabelToClass = CalcLabelToClassMap(ClassToLabel, 0);

    Initialized = true;
}

void TLabelConverter::InitializeMultiClass(TConstArrayRef<float> targets, int classesCount, bool allowConstLabel) {
    CB_ENSURE(!Initialized, "Can't initialize initialized object of TLabelConverter");

    MultiClass = true;

    TVector<float> targetsCopy(targets.begin(), targets.end());
    LabelToClass = CalcLabelToClassMap(std::move(targetsCopy), classesCount, allowConstLabel);
    ClassesCount = Max(classesCount, LabelToClass.ysize());

    ClassToLabel.resize(LabelToClass.ysize());
    for (const auto& keyValue : LabelToClass) {
        ClassToLabel[keyValue.second] = keyValue.first;
    }
    Initialized = true;
}

int TLabelConverter::GetApproxDimension() const {
    CB_ENSURE(Initialized, "Can't use uninitialized object of TLabelConverter");
    // return at least 2 for MultiClass case, otherwise allowConstLabel won't work
    return MultiClass ? Max(LabelToClass.ysize(), 2) : 1;
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
                CATBOOST_WARNING_LOG << "Label " << label << " not present in train set.";
            }
        }
    }

    if (!missingLabels.empty()) {
        CATBOOST_WARNING_LOG << "Label(s) " << JoinSeq(", ", missingLabels) << " are not present in the train set."
            << " Perhaps, something is wrong with the data." << Endl;
    }
}

bool TLabelConverter::IsInitialized() const {
    return Initialized;
}

bool TLabelConverter::IsMultiClass() const {
    return MultiClass;
}

TString TLabelConverter::SerializeClassParams(
    int classesCount,
    const TVector<NJson::TJsonValue>& classLabels
) const {
    CB_ENSURE(Initialized, "Can't use uninitialized object of TLabelConverter");
    TClassLabelOptions classLabelOptions;
    if (!classLabels.empty()) {
        classLabelOptions.ClassLabelType = NCB::GetRawTargetType(classLabels[0]);
    } else {
        classLabelOptions.ClassLabelType = NCB::ERawTargetType::Integer;
    }
    classLabelOptions.ClassToLabel = ClassToLabel;
    classLabelOptions.ClassesCount = classesCount;
    classLabelOptions.ClassLabels = classLabels;
    NJson::TJsonValue json;

    classLabelOptions.Save(&json);

    return WriteTJsonValue(json);
}

void PrepareTargetCompressed(const TLabelConverter& labelConverter, TVector<float>* labels) {
    CB_ENSURE(labelConverter.IsInitialized(), "Label converter isn't built.");
    labelConverter.ValidateLabels(*labels);
    for (auto& label : *labels) {
        label = labelConverter.GetClassIdx(label);
    }
}

int GetClassesCount(int classesCount, const TVector<NJson::TJsonValue>& classLabels) {
    if (classLabels.empty() || classesCount == 0) {
        return Max(classLabels.ysize(), classesCount);
    }

    CB_ENSURE(classesCount == classLabels.ysize(),
              "classes-count " << classesCount << " must be equal to size of class-names " << classLabels.ysize() << "if both are specified.");
    return classesCount;
}
