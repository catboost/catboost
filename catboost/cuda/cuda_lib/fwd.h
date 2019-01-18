#pragma once

#include <util/generic/fwd.h>

namespace NCudaLib {
    enum class EPtrType : int {
        CudaDevice,
        CudaHost, // pinned cuda memory
        Host      // CPU, non-pinned
    };
}

namespace NCudaLib {
    template <class T, class TMapping, EPtrType Type = EPtrType::CudaDevice>
    class TCudaBuffer;

    template <class T>
    class TDistributedObject;

    class TMirrorMapping;
    class TSingleMapping;
    class TStripeMapping;
}
