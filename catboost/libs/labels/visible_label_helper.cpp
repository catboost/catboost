#include "visible_label_helper.h"
#include "label_converter.h"

#include <catboost/libs/options/multiclass_label_options.h>


void TVisibleLabelsHelper::Initialize(const TString& multiclassLabelParams) {
    CB_ENSURE(!Initialized, "Can't initialize initialized object of TVisibleLabelsHelper");
    TMulticlassLabelOptions multiclassOptions;
    multiclassOptions.Load(ReadTJsonValue(multiclassLabelParams));

    int classesCount = multiclassOptions.ClassesCount.Get();
    const auto& classNames = multiclassOptions.ClassNames.Get();
    const auto& classToLabel = multiclassOptions.ClassToLabel.Get();

    VisibleApproxDimension = Max(classesCount, classNames.ysize());
    if (VisibleApproxDimension == 0) {  // labels extracted from data
        VisibleApproxDimension = classToLabel.ysize();

        int id = 0;
        for (const auto& label: classToLabel) {
            VisibleClassNames.push_back(ToString<float>(label));
            LabelToName.emplace(label, ToString<float>(label));
            SignificantLabelsIds.push_back(id++);
        }
    } else {  // user-defined labels
        SignificantLabelsIds = TVector<int>(classToLabel.begin(), classToLabel.end());

        if (classNames.empty()) {
            for (int id = 0; id < classesCount; ++id) {
                VisibleClassNames.push_back(ToString<int>(id));
                LabelToName.emplace(id, ToString<int>(id));
            }
        } else {  // classNames are not empty
            VisibleClassNames = classNames;
            int id = 0;
            for (const auto& name : classNames) {
                LabelToName.emplace(id++, name);
            }
        }
    }


    Initialized = true;
}

void TVisibleLabelsHelper::Initialize(int approxDimension) {
    CB_ENSURE(!Initialized, "Can't initialize initialized object of TVisibleLabelsHelper");

    VisibleApproxDimension = approxDimension;
    VisibleClassNames.resize(VisibleApproxDimension);
    SignificantLabelsIds.resize(VisibleApproxDimension);
    LabelToName.reserve(VisibleApproxDimension);

    for(int id = 0; id < approxDimension; ++id) {
        VisibleClassNames[id] = ToString<int>(id);
        LabelToName[id] = ToString<int>(id);
        SignificantLabelsIds[id]= id;
    }

    Initialized = true;
}

int TVisibleLabelsHelper::GetVisibleApproxDimension() const {
    CB_ENSURE(Initialized, "Can't use uninitialized object of TVisibleLabelsHelper");
    return VisibleApproxDimension;
}

TString TVisibleLabelsHelper::GetVisibleClassNameFromClass(int classIdx) const {
    CB_ENSURE(Initialized, "Can't use uninitialized object of TVisibleLabelsHelper");
    CB_ENSURE(classIdx >= 0 && classIdx < VisibleClassNames.ysize(), "Can't convert invalid class index to its visible(external) name");
    return VisibleClassNames[classIdx];
}

TString TVisibleLabelsHelper::GetVisibleClassNameFromLabel(float label) const {
    CB_ENSURE(Initialized, "Can't use uninitialized object of TVisibleLabelsHelper");
    const auto it = LabelToName.find(label);
    CB_ENSURE(it != LabelToName.end(), "Can't convert bad label back to class name.");
    return it->second;
}

int TVisibleLabelsHelper::GetVisibleIndex(int approxIdx) const {
    CB_ENSURE(Initialized, "Can't use uninitialized object of TVisibleLabelsHelper");
    CB_ENSURE(approxIdx >= 0 && approxIdx < SignificantLabelsIds.ysize(), "Can't convert invalid approx index to its visible(external) index");
    return SignificantLabelsIds[approxIdx];
}

bool TVisibleLabelsHelper::IsInitialized() const {
    return Initialized;
}
