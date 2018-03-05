#include "train_templ.h"

template void TrainOneIter<TLoglossError>(const TTrainData&, const TTrainData*, TLearnContext*);
