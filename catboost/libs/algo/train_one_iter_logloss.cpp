#include "train_templ.h"

template void TrainOneIter<TLoglossError>(const TDataset&, const TDataset*, TLearnContext*);
