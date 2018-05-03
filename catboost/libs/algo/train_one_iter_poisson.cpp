#include "train_templ.h"

template void TrainOneIter<TPoissonError>(const TDataset&, const TDatasetPtrs&, TLearnContext*);
