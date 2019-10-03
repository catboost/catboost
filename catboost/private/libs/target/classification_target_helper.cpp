#include "classification_target_helper.h"


TString NCB::TClassificationTargetHelper::Serialize() const {
    if (LabelConverter.IsInitialized()) {
        return LabelConverter.SerializeMulticlassParams((int)Options.ClassesCount.Get(), Options.ClassNames);
    }
    return "";
}
