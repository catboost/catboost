#include "ctr_calcers.h"

namespace NCatboostCuda {
    template class THistoryBasedCtrCalcer<NCudaLib::TSingleMapping>;
    template class THistoryBasedCtrCalcer<NCudaLib::TMirrorMapping>;

    template class TWeightedBinFreqCalcer<NCudaLib::TSingleMapping>;
    template class TWeightedBinFreqCalcer<NCudaLib::TMirrorMapping>;
}
