#include "train_templ.h"

template void TrainOneIter<TCrossEntropyError>(const NCB::TTrainingForCPUDataProviders&, TLearnContext*);
