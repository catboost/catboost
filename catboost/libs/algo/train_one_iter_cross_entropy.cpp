#include "train_templ.h"

template void TrainOneIter<TCrossEntropyError>(const TTrainData&, const TTrainData*, TLearnContext*);
