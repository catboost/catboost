#include "train_templ.h"

template void TrainOneIter<TQueryRmseError>(const NCB::TTrainingForCPUDataProviders&, TLearnContext*);
