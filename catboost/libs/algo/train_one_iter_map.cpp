#include "train_templ.h"

template void TrainOneIter<TMAPError>(const TTrainData&, TLearnContext*);
template void TrainOneIter<TSMAPError>(const TTrainData&, TLearnContext*);
