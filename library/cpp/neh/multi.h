#pragma once

#include "rpc.h"

#include <util/generic/vector.h>
#include <util/generic/string.h>

namespace NNeh {
    typedef TVector<TString> TListenAddrs;

    IRequesterRef MultiRequester(const TListenAddrs& addrs, IOnRequest* rq);
}
