#include "train_templ.h"

template void TrainOneIter<TPairLogitError>(const TDataset&, const TDatasetPtrs&, TLearnContext*);
