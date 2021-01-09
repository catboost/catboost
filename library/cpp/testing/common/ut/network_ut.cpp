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


TEST(FreePortTest, FreePortsRange) {
    NTesting::TScopedEnvironment envGuard("PORT_SYNC_PATH", TmpDir.Name());

    for (ui16 i = 2; i < 10; ++i) {
        TVector<NTesting::TPortHolder> ports = NTesting::NLegacy::GetFreePortsRange(i);
        ASSERT_EQ(i, ports.size());
        for (ui16 j = 1; j < i; ++j) {
            EXPECT_EQ(static_cast<ui16>(ports[j]), static_cast<ui16>(ports[0]) + j);
        }
    }
}
