#include "model.h"


TFullModel ReadModelWrapper(const TString& modelFile, EModelType format) throw(yexception) {
    return ReadModel(modelFile, format);
}
