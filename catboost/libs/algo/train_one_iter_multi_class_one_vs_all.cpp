#include "train_templ.h"

template void TrainOneIter<TMultiClassOneVsAllError>(const TDataset&, const TDataset*, TLearnContext*);
