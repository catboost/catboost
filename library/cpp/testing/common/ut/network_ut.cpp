#include <library/cpp/testing/common/env.h>
#include <library/cpp/testing/common/network.h>
#include <library/cpp/testing/common/scope.h>
#include <library/cpp/testing/gtest/gtest.h>

#include <util/generic/hash_set.h>

#include <util/folder/dirut.h>
#include <util/folder/path.h>
#include <util/folder/tempdir.h>
#include <util/generic/scope.h>
#include <util/network/sock.h>
#include <util/system/file.h>
#include <util/system/fs.h>
#include <util/system/sysstat.h>

class NetworkTestBase: public ::testing::Test {
protected:
    void SetUp() override {
        TmpDir.ConstructInPlace();
    }

    void TearDown() override {
        TmpDir.Clear();
    }

public:
    TMaybe<TTempDir> TmpDir;
};

class NetworkTest: public NetworkTestBase {};
class FreePortTest: public NetworkTestBase {};

TEST_F(NetworkTest, FreePort) {
    NTesting::TScopedEnvironment envGuard("PORT_SYNC_PATH", TmpDir->Name());
    NTesting::InitPortManagerFromEnv();
    TVector<NTesting::TPortHolder> ports(Reserve(100));

    for (size_t i = 0; i < 100; ++i) {
        ports.push_back(NTesting::GetFreePort());
    }

    THashSet<ui16> uniqPorts;
    for (auto& port : ports) {
        const TString guardPath = TmpDir->Path() / ToString(static_cast<ui16>(port));
        EXPECT_TRUE(NFs::Exists(guardPath));
        EXPECT_TRUE(uniqPorts.emplace(port).second);

        TInetStreamSocket sock;
        TSockAddrInet addr(TIpHost{INADDR_ANY}, port);
        ASSERT_EQ(0, SetSockOpt(sock, SOL_SOCKET, SO_REUSEADDR, 1));
        EXPECT_EQ(0, sock.Bind(&addr));
    }
    ports.clear();
    for (ui16 port : uniqPorts) {
        const TString guardPath = TmpDir->Path() / ToString(port);
        EXPECT_FALSE(NFs::Exists(guardPath));
    }
}

TEST_F(NetworkTest, FreePortWithinRanges) {
    NTesting::TScopedEnvironment envGuard{{
        {"PORT_SYNC_PATH", TmpDir->Name()},
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

TEST_F(NetworkTest, GetPortRandom) {
    NTesting::TScopedEnvironment envGuard{{
        {"PORT_SYNC_PATH", TmpDir->Name()},
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

TEST_F(NetworkTest, GetPortNonRandom) {
    NTesting::TScopedEnvironment envGuard{{
        {"PORT_SYNC_PATH", TmpDir->Name()},
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

TEST_F(NetworkTest, Permissions) {
    constexpr ui16 loPort = 3456;
    constexpr ui16 hiPort = 7654;
    NTesting::TScopedEnvironment envGuard{{
        {"PORT_SYNC_PATH", TmpDir->Name()},
        {"VALID_PORT_RANGE", ToString(loPort) + ":" + ToString(hiPort)},
    }};
    NTesting::InitPortManagerFromEnv();
    TVector<NTesting::TPortHolder> ports(Reserve(100));
    for (ui64 port = loPort; port <= hiPort; ++port) {
        const TString guardPath = TmpDir->Path() / ToString(static_cast<ui16>(port));
        TFile f{guardPath, OpenAlways | RdOnly};
        ASSERT_TRUE(f.IsOpen());
        ASSERT_TRUE(Chmod(f.GetName().c_str(), 0444) != -1);
    }
    Y_DEFER {
        Chmod(TmpDir->Path().c_str(), NFs::FP_COMMON_FILE);
    };
    ASSERT_TRUE(Chmod(TmpDir->Path().c_str(), 0555) != -1) << errno << " " << strerror(errno); // Lock dir
    for (size_t i = 0; i < 100; ++i) {
        NTesting::TPortHolder p;
        EXPECT_NO_THROW(p = NTesting::GetFreePort());
        ui16 port = p;
        ASSERT_GE(port, 3456u);
        ASSERT_LE(port, 7654u);
    }
}

TEST_F(FreePortTest, FreePortsRange) {
    NTesting::TScopedEnvironment envGuard("PORT_SYNC_PATH", TmpDir->Name());
    NTesting::InitPortManagerFromEnv();

    for (ui16 i = 2; i < 10; ++i) {
        TVector<NTesting::TPortHolder> ports = NTesting::NLegacy::GetFreePortsRange(i);
        ASSERT_EQ(i, ports.size());
        for (ui16 j = 1; j < i; ++j) {
            EXPECT_EQ(static_cast<ui16>(ports[j]), static_cast<ui16>(ports[0]) + j);
        }
    }
}
