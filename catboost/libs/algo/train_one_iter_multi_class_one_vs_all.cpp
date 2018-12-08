#include "train_templ.h"

template void TrainOneIter<TMultiClassOneVsAllError>(const NCB::TTrainingForCPUDataProviders&, TLearnContext*);
