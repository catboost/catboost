#include "train_templ.h"

template void TrainOneIter<TPoissonError>(const TTrainData&, const TTrainData*, TLearnContext*);
