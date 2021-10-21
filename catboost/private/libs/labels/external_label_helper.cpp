#include "external_label_helper.h"

#include "helpers.h"
#include "label_converter.h"

#include <util/generic/xrange.h>

#include <catboost/private/libs/options/json_helper.h>
#include <catboost/private/libs/options/class_label_options.h>

#include <catboost/libs/model/model.h>

#include <library/cpp/json/json_value.h>

#include <util/generic/cast.h>
#include <util/string/cast.h>


TExternalLabelsHelper::TExternalLabelsHelper(const TFullModel& model)
    : Initialized(false)
    , ExternalApproxDimension(0)
{
    if (model.GetDimensionsCount() > 1 || model.GetModelClassLabels().size() == 1) {  // is multiclass or multilabel?
        // "class_params" is new, more generic option, used for binclass as well
        for (const auto& paramName : {"class_params", "multiclass_params"}) {
            if (model.ModelInfo.contains(paramName)) {
                InitializeImpl(SafeIntegerCast<int>(model.GetDimensionsCount()), model.ModelInfo.at(paramName));
                return;
            }
        }

        InitializeImpl(model.GetDimensionsCount());
    } else {
        const TVector<NJson::TJsonValue> binclassLabels = model.GetModelClassLabels();
        if (!binclassLabels.empty()) {
            InitializeImpl(binclassLabels);
        }
    }
}


void TExternalLabelsHelper::InitializeImpl(int approxDimension, const TString& classLabelParams) {
    ExternalApproxDimension = approxDimension;

    TClassLabelOptions classLabelOptions;
    classLabelOptions.Load(ReadTJsonValue(classLabelParams));

    int classesCount = classLabelOptions.ClassesCount.Get();
    const auto& classLabels = classLabelOptions.ClassLabels.Get();
    const auto& classToLabel = classLabelOptions.ClassToLabel.Get();

    int specifiedClassCount = Max(classesCount, classLabels.ysize());
    ExternalApproxDimension = (approxDimension == 1) ? 1 : specifiedClassCount;

    if (specifiedClassCount == 0) {  // labels extracted from data
        ExternalApproxDimension = (approxDimension == 1) ? 1 : classToLabel.ysize();

        for (auto classId : xrange(classToLabel.ysize())) {
            TString className = ToString(classToLabel[classId]);
            VisibleClassNames.push_back(className);
            SignificantLabelsIds.push_back(classId);
        }
    } else {  // user-defined labels
        ExternalApproxDimension = (approxDimension == 1) ? 1 : specifiedClassCount;

        SignificantLabelsIds.assign(classToLabel.begin(), classToLabel.end());

        if (classLabels.empty()) {
            for (int id = 0; id < classesCount; ++id) {
                VisibleClassNames.push_back(ToString(id));
            }
        } else {  // classNames are not empty
            VisibleClassNames = NCB::ClassLabelsToStrings(classLabels);
        }
    }


    Initialized = true;
}

void TExternalLabelsHelper::InitializeImpl(int approxDimension) {
    ExternalApproxDimension = approxDimension;
    VisibleClassNames.resize(ExternalApproxDimension);
    SignificantLabelsIds.resize(ExternalApproxDimension);

    for(int id = 0; id < approxDimension; ++id) {
        VisibleClassNames[id] = ToString(id);
        SignificantLabelsIds[id]= id;
    }

    Initialized = true;
}

void TExternalLabelsHelper::InitializeImpl(const TVector<NJson::TJsonValue>& binclassLabels) {
    CB_ENSURE(binclassLabels.size() == 2, "binclassLabels size is not equal to 2");

    ExternalApproxDimension = 1;
    VisibleClassNames = NCB::ClassLabelsToStrings(binclassLabels);
    SignificantLabelsIds.assign({0, 1});

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

int TExternalLabelsHelper::GetExternalIndex(int approxIdx) const {
    CB_ENSURE(Initialized, "Can't use uninitialized object of TExternalLabelsHelper");
    CB_ENSURE(approxIdx >= 0 && approxIdx < SignificantLabelsIds.ysize(), "Can't convert invalid approx index to its visible(external) index");
    return SignificantLabelsIds[approxIdx];
}

bool TExternalLabelsHelper::IsInitialized() const {
    return Initialized;
}
