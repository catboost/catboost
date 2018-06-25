#include "visible_label_helper.h"
#include "label_converter.h"
#include "multiclass_label_options.h"


void TVisibleLabelsHelper::Initialize(const TString& multiclassLabelParams) {
    CB_ENSURE(!Initialized, "Can't initialize initialized object of TVisibleLabelsHelper");
    TMulticlassLabelOptions multiclassOptions;
    multiclassOptions.Load(ReadTJsonValue(multiclassLabelParams));

    int classesCount = multiclassOptions.ClassesCount.Get();
    const auto& classNames = multiclassOptions.ClassNames.Get();

    ClassToLabel = multiclassOptions.ClassToLabel.Get();

    VisibleApproxDimension = Max(classesCount, classNames.ysize());
    if (VisibleApproxDimension == 0) {  // labels extracted from data
        VisibleApproxDimension = ClassToLabel.ysize();

        int id = 0;
        for (const auto& label: ClassToLabel) {
            VisibleClassNames.push_back(ToString<float>(label));
            SignificantLabelsIds.push_back(id++);
        }
    } else {  // user-defined labels
        SignificantLabelsIds = TVector<int>(ClassToLabel.begin(), ClassToLabel.end());

        if (classNames.empty()) {
            for (int id = 0; id < classesCount; ++id) {
                VisibleClassNames.push_back(ToString<int>(id));
            }
        } else {  // classNames are not empty
            VisibleClassNames = classNames;
        }
    }

    Initialized = true;
}

void TVisibleLabelsHelper::Initialize(int approxDimension) {
    CB_ENSURE(!Initialized, "Can't initialize initialized object of TVisibleLabelsHelper");

    VisibleApproxDimension = approxDimension;
    VisibleClassNames.resize(VisibleApproxDimension);
    SignificantLabelsIds.resize(VisibleApproxDimension);
    ClassToLabel.resize(VisibleApproxDimension);

    for(int id = 0; id < approxDimension; ++id) {
        VisibleClassNames[id] = ToString<int>(id);
        SignificantLabelsIds[id]= id;
        ClassToLabel[id] = id;
    }

    Initialized = true;
}

int TVisibleLabelsHelper::GetVisibleApproxDimension() const {
    CB_ENSURE(Initialized, "Can't use uninitialized object of TVisibleLabelsHelper");
    return VisibleApproxDimension;
}

TString TVisibleLabelsHelper::GetVisibleClassName(int classIdx) const {
    CB_ENSURE(Initialized, "Can't use uninitialized object of TVisibleLabelsHelper");
    CB_ENSURE(classIdx >= 0 && classIdx < VisibleClassNames.ysize(), "Can't convert invalid class index to its visible(external) name");
    return VisibleClassNames[classIdx];
}

int TVisibleLabelsHelper::GetVisibleIndex(int approxIdx) const {
    CB_ENSURE(Initialized, "Can't use uninitialized object of TVisibleLabelsHelper");
    CB_ENSURE(approxIdx >= 0 && approxIdx < SignificantLabelsIds.ysize(), "Can't convert invalid approx index to its visible(external) index");
    return SignificantLabelsIds[approxIdx];
}

bool TVisibleLabelsHelper::IsInitialized() const {
    return Initialized;
}
