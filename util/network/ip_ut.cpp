#include "ip.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/generic/yexception.h>

class TSysIpTest: public TTestBase {
    UNIT_TEST_SUITE(TSysIpTest);
    UNIT_TEST(TestIpFromString);
    UNIT_TEST_EXCEPTION(TestIpFromString2, yexception);
    UNIT_TEST_EXCEPTION(TestIpFromString3, yexception);
    UNIT_TEST_EXCEPTION(TestIpFromString4, yexception);
    UNIT_TEST_EXCEPTION(TestIpFromString5, yexception);
    UNIT_TEST(TestIpToString);
    UNIT_TEST_SUITE_END();

private:
    void TestIpFromString();
    void TestIpFromString2();
    void TestIpFromString3();
    void TestIpFromString4();
    void TestIpFromString5();
    void TestIpToString();
};

UNIT_TEST_SUITE_REGISTRATION(TSysIpTest);

void TSysIpTest::TestIpFromString() {
    const char* ipStr[] = {"192.168.0.1", "87.255.18.167", "255.255.0.31", "188.225.124.255"};
    ui8 ipArr[][4] = {{192, 168, 0, 1}, {87, 255, 18, 167}, {255, 255, 0, 31}, {188, 225, 124, 255}};

    for (size_t i = 0; i < Y_ARRAY_SIZE(ipStr); ++i) {
        const ui32 ip = IpFromString(ipStr[i]);

        UNIT_ASSERT(memcmp(&ip, ipArr[i], sizeof(ui32)) == 0);
    }
}

void TSysIpTest::TestIpFromString2() {
    IpFromString("XXXXXXWXW");
}

void TSysIpTest::TestIpFromString3() {
    IpFromString("986.0.37.255");
}

void TSysIpTest::TestIpFromString4() {
    IpFromString("256.0.22.365");
}

void TSysIpTest::TestIpFromString5() {
    IpFromString("245.12..0");
}

void TSysIpTest::TestIpToString() {
    ui8 ipArr[][4] = {{192, 168, 0, 1}, {87, 255, 18, 167}, {255, 255, 0, 31}, {188, 225, 124, 255}};

    const char* ipStr[] = {"192.168.0.1", "87.255.18.167", "255.255.0.31", "188.225.124.255"};

    for (size_t i = 0; i < Y_ARRAY_SIZE(ipStr); ++i) {
        UNIT_ASSERT(IpToString(*reinterpret_cast<TIpHost*>(&(ipArr[i]))) == ipStr[i]);
    }
}
