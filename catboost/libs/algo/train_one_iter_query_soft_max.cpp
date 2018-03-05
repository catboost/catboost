#include "train_templ.h"

template void TrainOneIter<TQuerySoftMaxError>(const TTrainData&, const TTrainData*, TLearnContext*);
