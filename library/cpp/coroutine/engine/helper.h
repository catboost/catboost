#pragma once

#include <util/generic/string.h>
#include <util/datetime/base.h>

namespace NCoro {

    // @brief check that address  `host`:`port` is connectable
    bool TryConnect(const TString& host, ui16 port, TDuration timeout = TDuration::Seconds(1));

    // @brief waits until address `host`:`port` became connectable, but not more than timeout
    // @return true on success, false if timeout exceeded
    bool WaitUntilConnectable(const TString& host, ui16 port, TDuration timeout);

}
