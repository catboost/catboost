#include <library/cpp/testing/common/network.h>
#include <library/cpp/testing/common/scope.h>

#include <util/generic/hash_set.h>

#include <util/folder/dirut.h>
#include <util/folder/path.h>
#include <util/folder/tempdir.h>
#include <util/network/sock.h>
#include <util/system/fs.h>

#include <library/cpp/testing/gtest/gtest.h>

static TTempDir TmpDir;

TEST(NetworkTest, FreePort) {
    NTesting::TScopedEnvironment envGuard("PORT_SYNC_PATH", TmpDir.Name());
    NTesting::InitPortManagerFromEnv();
    TVector<NTesting::TPortHolder> ports(Reserve(100));

    for (size_t i = 0; i < 100; ++i) {
        ports.push_back(NTesting::GetFreePort());
    }

    THashSet<ui16> uniqPorts;
    for (auto& port : ports) {
        const TString guardPath = TmpDir.Path() / ToString(static_cast<ui16>(port));
        EXPECT_TRUE(NFs::Exists(guardPath));
        EXPECT_TRUE(uniqPorts.emplace(port).second);

        TInetStreamSocket sock;
        TSockAddrInet addr(TIpHost{INADDR_ANY}, port);
        ASSERT_EQ(0, SetSockOpt(sock, SOL_SOCKET, SO_REUSEADDR, 1));
        EXPECT_EQ(0, sock.Bind(&addr));
    }
    ports.clear();
    for (ui16 port : uniqPorts) {
        const TString guardPath = TmpDir.Path() / ToString(port);
        EXPECT_FALSE(NFs::Exists(guardPath));
    }
}

TEST(NetworkTest, FreePortWithinRanges) {
    NTesting::TScopedEnvironment envGuard{{
            {"PORT_SYNC_PATH", TmpDir.Name()},
            {"VALID_PORT_RANGE", "3456:7654"},
        }};
    NTesting::InitPortManagerFromEnv();

    for (size_t i = 0; i < 100; ++i) {
        auto holder = NTesting::GetFreePort();
        ui16 port = holder;
        ASSERT_GE(port, 3456u);
        ASSERT_LE(port, 7654u);
    }
}

TEST(NetworkTest, GetPortRandom) {
    NTesting::TScopedEnvironment envGuard{{
            {"PORT_SYNC_PATH", TmpDir.Name()},
            {"NO_RANDOM_PORTS", ""},
        }};
    NTesting::InitPortManagerFromEnv();

    ui16 testPort = 80; // value just must be outside the assignable range
    for (size_t i = 0; i < 10; ++i) {
        NTesting::TPortHolder assigned = NTesting::NLegacy::GetPort(testPort);
        ui16 assignedInt = assigned;
        ASSERT_NE(testPort, assignedInt);
    }
}

TEST(NetworkTest, GetPortNonRandom) {
    NTesting::TScopedEnvironment envGuard{{
            {"PORT_SYNC_PATH", TmpDir.Name()},
            {"NO_RANDOM_PORTS", "1"},
        }};
    NTesting::InitPortManagerFromEnv();

    TVector<ui16> ports(Reserve(100)); // keep integers, we don't need the ports to remain allocated

    for (size_t i = 0; i < 10; ++i) {
        auto portHolder = NTesting::GetFreePort();
        ports.push_back(portHolder);
    }

    for (auto& testPort : ports) {
        NTesting::TPortHolder assigned = NTesting::NLegacy::GetPort(testPort);
        ui16 assignedInt = assigned;
        ASSERT_EQ(testPort, assignedInt);
    }
}


TEST(FreePortTest, FreePortsRange) {
    NTesting::TScopedEnvironment envGuard("PORT_SYNC_PATH", TmpDir.Name());
    NTesting::InitPortManagerFromEnv();

    for (ui16 i = 2; i < 10; ++i) {
        TVector<NTesting::TPortHolder> ports = NTesting::NLegacy::GetFreePortsRange(i);
        ASSERT_EQ(i, ports.size());
        for (ui16 j = 1; j < i; ++j) {
            EXPECT_EQ(static_cast<ui16>(ports[j]), static_cast<ui16>(ports[0]) + j);
        }
    }
}
