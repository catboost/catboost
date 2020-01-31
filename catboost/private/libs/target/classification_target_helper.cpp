#include "classification_target_helper.h"


TString NCB::TClassificationTargetHelper::Serialize() const {
    if (LabelConverter.IsInitialized()) {
        return LabelConverter.SerializeClassParams((int)Options.ClassesCount.Get(), Options.ClassLabels);
    }
    return "";
}
