#include "external_label_helper.h"
#include "label_converter.h"

#include <util/generic/xrange.h>

#include <catboost/private/libs/options/json_helper.h>
#include <catboost/private/libs/options/multiclass_label_options.h>

#include <catboost/libs/model/model.h>

#include <util/string/cast.h>


TExternalLabelsHelper::TExternalLabelsHelper(const TFullModel& model)
    : Initialized(false)
    , ExternalApproxDimension(0)
{
    if (model.GetDimensionsCount() > 1) {  // is multiclass?
        if (model.ModelInfo.contains("multiclass_params")) {
            InitializeImpl(model.ModelInfo.at("multiclass_params"));
        }
        else {
            InitializeImpl(model.GetDimensionsCount());
        }
    } else {
        const TVector<TString> binclassNames = model.GetModelClassNames();
        if (!binclassNames.empty()) {
            InitializeImpl(binclassNames);
        }
    }
}


void TExternalLabelsHelper::InitializeImpl(const TString& multiclassLabelParams) {
    TMulticlassLabelOptions multiclassOptions;
    multiclassOptions.Load(ReadTJsonValue(multiclassLabelParams));

    int classesCount = multiclassOptions.ClassesCount.Get();
    const auto& classNames = multiclassOptions.ClassNames.Get();
    const auto& classToLabel = multiclassOptions.ClassToLabel.Get();

    ExternalApproxDimension = Max(classesCount, classNames.ysize());
    if (ExternalApproxDimension == 0) {  // labels extracted from data
        ExternalApproxDimension = classToLabel.ysize();

        for (auto classId : xrange(ExternalApproxDimension)) {
            TString className = ToString(classToLabel[classId]);
            VisibleClassNames.push_back(className);
            LabelToName.emplace(float(classId), className);
            SignificantLabelsIds.push_back(classId);
        }
    } else {  // user-defined labels
        SignificantLabelsIds.assign(classToLabel.begin(), classToLabel.end());

        if (classNames.empty()) {
            for (int id = 0; id < classesCount; ++id) {
                VisibleClassNames.push_back(ToString(id));
                LabelToName.emplace(id, ToString(id));
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

void TExternalLabelsHelper::InitializeImpl(int approxDimension) {
    ExternalApproxDimension = approxDimension;
    VisibleClassNames.resize(ExternalApproxDimension);
    SignificantLabelsIds.resize(ExternalApproxDimension);
    LabelToName.reserve(ExternalApproxDimension);

    for(int id = 0; id < approxDimension; ++id) {
        VisibleClassNames[id] = ToString(id);
        LabelToName[id] = ToString(id);
        SignificantLabelsIds[id]= id;
    }

    Initialized = true;
}

void TExternalLabelsHelper::InitializeImpl(const TVector<TString>& binclassNames) {
    CB_ENSURE(binclassNames.size() == 2, "binclassNames size is not equal to 2");

    ExternalApproxDimension = 1;
    VisibleClassNames = binclassNames;
    SignificantLabelsIds.assign({0, 1});
    LabelToName[0] = binclassNames[0];
    LabelToName[1] = binclassNames[1];

    Initialized = true;
}

int TExternalLabelsHelper::GetExternalApproxDimension() const {
    CB_ENSURE(Initialized, "Can't use uninitialized object of TExternalLabelsHelper");
    return ExternalApproxDimension;
}

TString TExternalLabelsHelper::GetVisibleClassNameFromClass(int classIdx) const {
    CB_ENSURE(Initialized, "Can't use uninitialized object of TExternalLabelsHelper");
    CB_ENSURE(classIdx >= 0 && classIdx < VisibleClassNames.ysize(), "Can't convert invalid class index to its visible(external) name");
    return VisibleClassNames[classIdx];
}

TString TExternalLabelsHelper::GetVisibleClassNameFromLabel(float label) const {
    CB_ENSURE(Initialized, "Can't use uninitialized object of TExternalLabelsHelper");
    const auto it = LabelToName.find(label);
    CB_ENSURE(it != LabelToName.end(), "Can't convert bad label back to class name.");
    return it->second;
}

int TExternalLabelsHelper::GetExternalIndex(int approxIdx) const {
    CB_ENSURE(Initialized, "Can't use uninitialized object of TExternalLabelsHelper");
    CB_ENSURE(approxIdx >= 0 && approxIdx < SignificantLabelsIds.ysize(), "Can't convert invalid approx index to its visible(external) index");
    return SignificantLabelsIds[approxIdx];
}

bool TExternalLabelsHelper::IsInitialized() const {
    return Initialized;
}
