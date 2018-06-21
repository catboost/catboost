#include "bootstrap.h"

namespace NCatboostCuda {
    template class TBootstrap<NCudaLib::TStripeMapping>;

    template class TBootstrap<NCudaLib::TSingleMapping>;

    template class TBootstrap<NCudaLib::TMirrorMapping>;
}
