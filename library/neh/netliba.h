#pragma once

#include <util/datetime/base.h>

namespace NNeh {
    //global options
    struct TNetLibaOptions {
        static size_t ClientThreads;

        //period for quick send complete confirmation for next request to some address after receiving request ack
        static TDuration AckTailEffect;

        //set option, - return false, if option name not recognized
        static bool Set(TStringBuf name, TStringBuf value);
    };

    class IProtocol;

    IProtocol* NetLibaProtocol();
}
