#include "train_templ.h"

template void TrainOneIter<TRMSEError>(const NCB::TTrainingForCPUDataProviders&, TLearnContext*);
