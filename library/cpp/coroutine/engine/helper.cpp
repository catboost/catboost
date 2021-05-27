#include "helper.h"

#include "impl.h"
#include "network.h"

namespace NCoro {

    bool TryConnect(const TString& host, ui16 port, TDuration timeout) {
        bool connected = false;

        auto f = [&connected, &host, port, timeout](TCont* c) {
            TSocketHolder socket;
            TNetworkAddress address(host, port);
            connected = (0 == NCoro::ConnectT(c, socket, address, timeout));
        };

        TContExecutor e(128 * 1024);
        e.Create(f, "try_connect");
        e.Execute();
        return connected;
    }

    bool WaitUntilConnectable(const TString& host, ui16 port, TDuration timeout) {
        const TInstant deadline = timeout.ToDeadLine();

        for (size_t i = 1; Now() < deadline; ++i) {
            const TDuration waitTime = TDuration::MilliSeconds(100) * i * i;
            SleepUntil(Min(Now() + waitTime, deadline));

            if (TryConnect(host, port, waitTime)) {
                return true;
            }
        }

        return false;
    }
}
