#include "ctr_helper.h"

namespace NCatboostCuda {
    template class TCalcCtrHelper<NCudaLib::TSingleMapping>;
    template class TCalcCtrHelper<NCudaLib::TMirrorMapping>;
}
