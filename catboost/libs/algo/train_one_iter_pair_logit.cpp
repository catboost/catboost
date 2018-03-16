#include "train_templ.h"

template void TrainOneIter<TPairLogitError>(const TTrainData&, const TTrainData*, TLearnContext*);
