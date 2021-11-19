#include "model.h"


TFullModel ReadModelWrapper(const TString& modelFile, EModelType format) {
    return ReadModel(modelFile, format);
}
