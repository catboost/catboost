#include "train_templ.h"

template void TrainOneIter<TMultiClassOneVsAllError>(const TTrainData&, const TTrainData*, TLearnContext*);
