#include "train_templ.h"

template void TrainOneIter<TRMSEError>(const TDataset&, const TDataset*, TLearnContext*);
