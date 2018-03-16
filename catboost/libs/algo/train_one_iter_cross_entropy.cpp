#include "train_templ.h"

template void TrainOneIter<TCrossEntropyError>(const TDataset&, const TDataset*, TLearnContext*);
