#include "train_templ.h"

template void TrainOneIter<TPairLogitError>(const NCB::TTrainingForCPUDataProviders&, TLearnContext*);
