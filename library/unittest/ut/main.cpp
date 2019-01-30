#include <library/unittest/gtest.h>
#include <library/unittest/registar.h>
#include <library/unittest/tests_data.h>

#include <util/generic/set.h>
#include <util/network/sock.h>
#include <util/system/env.h>
#include <util/system/fs.h>

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

    struct TSimpleFixture : public NUnitTest::TBaseFixture {
        size_t Value = 24;
    };

    struct TOtherFixture : public NUnitTest::TBaseFixture {
        size_t TheAnswer = 42;
    };
}

TEST_F(TFixture, Test1) {
    ASSERT_EQ(I, 5);
}

TEST(ETest, Test1) {
    UNIT_CHECK_GENERATED_EXCEPTION(ythrow yexception(), yexception);
    UNIT_CHECK_GENERATED_NO_EXCEPTION(true, yexception);
}

Y_UNIT_TEST_SUITE(TPortManagerTest) {
    TFsPath workDir(GetWorkPath() + "/tmp/ports_test");

    Y_UNIT_TEST(TestValidPortsIpv4) {
        TPortManager pm(workDir);
        ui16 port = pm.GetPort();
        TInetStreamSocket sock;
        TSockAddrInet addr((TIpHost)INADDR_ANY, port);
        SetReuseAddressAndPort(sock);
        int ret = sock.Bind(&addr);
        UNIT_ASSERT_EQUAL(ret, 0);
    }

    Y_UNIT_TEST(TestValidPortsIpv6) {
        TPortManager pm(workDir);
        ui16 port = pm.GetPort();
        TInet6StreamSocket sock;
        TSockAddrInet6 addr("::", port);
        SetReuseAddressAndPort(sock);
        int ret = sock.Bind(&addr);
        UNIT_ASSERT_EQUAL(ret, 0);
    }

    Y_UNIT_TEST(TestOccupancy) {
        TPortManager pm(workDir);
        TSet<ui16> ports;
        for (int i = 0; i < 1000; i++) {
            ui16 port = pm.GetPort();
            UNIT_ASSERT_VALUES_EQUAL(ports.contains(port), false);
            ports.insert(port);
        }
    }

    Y_UNIT_TEST(TestRandomPort) {
        TPortManager pm(workDir);
        UNIT_ASSERT_VALUES_UNEQUAL(pm.GetPort(0), pm.GetPort(0));
        UNIT_ASSERT_VALUES_UNEQUAL(pm.GetPort(8123), pm.GetPort(8123));
        UNIT_ASSERT_VALUES_UNEQUAL(pm.GetPort(8123), 8123);
    }

    Y_UNIT_TEST(TestRequiredPort) {
        TPortManager pm(workDir);
        SetEnv("NO_RANDOM_PORTS", "1");
        UNIT_ASSERT_VALUES_UNEQUAL(pm.GetPort(0), pm.GetPort(0));
        UNIT_ASSERT_VALUES_EQUAL(pm.GetPort(8123), pm.GetPort(8123));
        SetEnv("NO_RANDOM_PORTS", "");
    }

    int CheckPort(ui16 port) {
        TInetStreamSocket sock;
        TSockAddrInet addr((TIpHost)INADDR_ANY, port);
        SetReuseAddressAndPort(sock);
        return sock.Bind(&addr);
    }

    Y_UNIT_TEST(TestPortsRange) {
        TPortManager pm(workDir);
        ui16 port = pm.GetPortsRange(3000, 3);
        UNIT_ASSERT(port >= 3000);

        for (ui32 i = 0; i < 3; ++i) {
            UNIT_ASSERT_EQUAL(CheckPort(port + i), 0);
        }

        ui16 anotherPort = pm.GetPortsRange(port, 3);
        UNIT_ASSERT(anotherPort >= port + 3);

        port = pm.GetPortsRange(anotherPort, 1);
        UNIT_ASSERT(port > anotherPort);
    }

    Y_UNIT_TEST(TestGivenValidPortRange) {
        ui16 port = 0;
        {
            // We need to free provided port.
            TPortManager pm(workDir);
            port = pm.GetPort(0);
        }
        SetEnv("VALID_PORT_RANGE", ToString(port) + ":" + ToString(port + 1));
        TPortManager pm(workDir);
        UNIT_ASSERT_VALUES_EQUAL(pm.GetPort(0), port);
        SetEnv("VALID_PORT_RANGE", "");
    }
}

Y_UNIT_TEST_SUITE(TestParams) {
    Y_UNIT_TEST(TestDefault){
        UNIT_ASSERT_EQUAL(UNIT_GET_PARAM("key", "default"), "default")}

    Y_UNIT_TEST(TestSetParam) {
        ut_context.Processor->SetParam("key", "value");
        UNIT_ASSERT_EQUAL(UNIT_GET_PARAM("key", ""), "value")
    }
}

Y_UNIT_TEST_SUITE(TestSingleTestFixture)
{
    Y_UNIT_TEST_F(Test3, TSimpleFixture) {
        UNIT_ASSERT_EQUAL(Value, 24);
    }
}

Y_UNIT_TEST_SUITE_F(TestSuiteFixture, TSimpleFixture)
{
    Y_UNIT_TEST(Test1) {
        UNIT_ASSERT(Value == 24);
        Value = 25;
    }

    Y_UNIT_TEST(Test2) {
        UNIT_ASSERT_EQUAL(Value, 24);
    }

    Y_UNIT_TEST_F(Test3, TOtherFixture) {
        UNIT_ASSERT_EQUAL(TheAnswer, 42);
    }
}
