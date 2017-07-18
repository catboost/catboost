#include <util/system/env.h>

#include <library/unittest/registar.h>
#include <library/unittest/tests_data.h>
#include <library/unittest/gtest.h>
#include <util/generic/set.h>
#include <util/network/sock.h>

TEST(GTest, Test1) {
    UNIT_ASSERT_EQUAL(1, 1);
}

TEST(GTest, Test2) {
    UNIT_ASSERT_EQUAL(2, 2);
}

namespace {
    struct TFixture : ::testing::Test {
        TFixture()
            : I(0)
        {
        }

        void SetUp() override {
            I = 5;
        }

        int I;
    };
}

TEST_F(TFixture, Test1) {
    ASSERT_EQ(I, 5);
}

TEST(ETest, Test1) {
    UNIT_CHECK_GENERATED_EXCEPTION(ythrow yexception(), yexception);
    UNIT_CHECK_GENERATED_NO_EXCEPTION(true, yexception);
}

SIMPLE_UNIT_TEST_SUITE(TPortManagerTest) {
    SIMPLE_UNIT_TEST(TestValidPortsIpv4) {
        TPortManager pm;
        ui16 port = pm.GetPort();
        TInetStreamSocket sock;
        TSockAddrInet addr((TIpHost)INADDR_ANY, port);
        SetSockOpt(sock, SOL_SOCKET, SO_REUSEADDR, 1);
        int ret = sock.Bind(&addr);
        UNIT_ASSERT_EQUAL(ret, 0);
    }

    SIMPLE_UNIT_TEST(TestValidPortsIpv6) {
        TPortManager pm;
        ui16 port = pm.GetPort();
        TInet6StreamSocket sock;
        TSockAddrInet6 addr("::", port);
        SetSockOpt(sock, SOL_SOCKET, SO_REUSEADDR, 1);
        int ret = sock.Bind(&addr);
        UNIT_ASSERT_EQUAL(ret, 0);
    }

    SIMPLE_UNIT_TEST(TestOccupancy) {
        TPortManager pm;
        yset<ui16> ports;
        for (int i = 0; i < 1000; i++) {
            ui16 port = pm.GetPort();
            UNIT_ASSERT_VALUES_EQUAL(ports.has(port), false);
            ports.insert(port);
        }
    }

    SIMPLE_UNIT_TEST(TestRandomPort) {
        TPortManager pm;
        UNIT_ASSERT_VALUES_UNEQUAL(pm.GetPort(0), pm.GetPort(0));
        UNIT_ASSERT_VALUES_UNEQUAL(pm.GetPort(8123), pm.GetPort(8123));
        UNIT_ASSERT_VALUES_UNEQUAL(pm.GetPort(8123), 8123);
    }

    SIMPLE_UNIT_TEST(TestRequiredPort) {
        TPortManager pm;
        SetEnv("NO_RANDOM_PORTS", "1");
        UNIT_ASSERT_VALUES_UNEQUAL(pm.GetPort(0), pm.GetPort(0));
        UNIT_ASSERT_VALUES_EQUAL(pm.GetPort(8123), pm.GetPort(8123));
        SetEnv("NO_RANDOM_PORTS", "");
    }
}
