#include "train_templ.h"

template void TrainOneIter<TQueryRmseError>(const TDataset&, const TDataset*, TLearnContext*);
