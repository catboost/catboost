#include "train_templ.h"

template void TrainOneIter<TPoissonError>(const NCB::TTrainingForCPUDataProviders&, TLearnContext*);
