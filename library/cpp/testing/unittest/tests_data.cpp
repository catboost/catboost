#include "tests_data.h"
#include "registar.h"

#include <library/cpp/testing/common/network.h>

#include <util/system/env.h>
#include <util/system/mutex.h>

class TPortManager::TPortManagerImpl {
public:
    TPortManagerImpl(bool reservePortsForCurrentTest)
        : EnableReservePortsForCurrentTest(reservePortsForCurrentTest)
        , DisableRandomPorts(!GetEnv("NO_RANDOM_PORTS").empty())
    {
    }

    ui16 GetPort(ui16 port) {
        if (port && DisableRandomPorts) {
            return port;
        }

        TAtomicSharedPtr<NTesting::IPort> holder(NTesting::GetFreePort().Release());
        ReservePortForCurrentTest(holder);

        TGuard<TMutex> g(Lock);
        ReservedPorts.push_back(holder);
        return holder->Get();
    }

    ui16 GetUdpPort(ui16 port) {
        return GetPort(port);
    }

    ui16 GetTcpPort(ui16 port) {
        return GetPort(port);
    }

    ui16 GetTcpAndUdpPort(ui16 port) {
        return GetPort(port);
    }

    ui16 GetPortsRange(const ui16 startPort, const ui16 range) {
        Y_UNUSED(startPort);
        auto ports = NTesting::NLegacy::GetFreePortsRange(range);
        ui16 first = ports[0];
        TGuard<TMutex> g(Lock);
        for (auto& port : ports) {
            ReservedPorts.emplace_back(port.Release());
            ReservePortForCurrentTest(ReservedPorts.back());
        }
        return first;
    }

private:
    void ReservePortForCurrentTest(const TAtomicSharedPtr<NTesting::IPort>& portGuard) {
        if (EnableReservePortsForCurrentTest) {
            TTestBase* currentTest = NUnitTest::NPrivate::GetCurrentTest();
            if (currentTest != nullptr) {
                currentTest->RunAfterTest([guard = portGuard]() mutable {
                    guard = nullptr; // remove reference for allocated port
                });
            }
        }
    }

private:
    TMutex Lock;
    TVector<TAtomicSharedPtr<NTesting::IPort>> ReservedPorts;
    const bool EnableReservePortsForCurrentTest;
    const bool DisableRandomPorts;
};

TPortManager::TPortManager(bool reservePortsForCurrentTest)
    : Impl_(new TPortManagerImpl(reservePortsForCurrentTest))
{
}

TPortManager::~TPortManager() {
}

ui16 TPortManager::GetPort(ui16 port) {
    return Impl_->GetTcpPort(port);
}

ui16 TPortManager::GetTcpPort(ui16 port) {
    return Impl_->GetTcpPort(port);
}

ui16 TPortManager::GetUdpPort(ui16 port) {
    return Impl_->GetUdpPort(port);
}

ui16 TPortManager::GetTcpAndUdpPort(ui16 port) {
    return Impl_->GetTcpAndUdpPort(port);
}

ui16 TPortManager::GetPortsRange(const ui16 startPort, const ui16 range) {
    return Impl_->GetPortsRange(startPort, range);
}

ui16 GetRandomPort() {
    TPortManager* pm = Singleton<TPortManager>(false);
    return pm->GetPort();
}
