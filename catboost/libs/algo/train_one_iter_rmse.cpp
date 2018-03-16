#include "train_templ.h"

template void TrainOneIter<TRMSEError>(const TTrainData&, const TTrainData*, TLearnContext*);
