#include <library/cpp/unittest/registar.h>
#include "udp_address.h"

namespace NNetliba_v12 {
    class TNetliba_v12UdpAddressTest: public TTestBase {
        UNIT_TEST_SUITE(TNetliba_v12UdpAddressTest)
        UNIT_TEST(IsValidIPv6Test)
        UNIT_TEST(IsValidIPv4forV6Test)
        UNIT_TEST_SUITE_END();

        void IsValidIPv6Test() {
            UNIT_ASSERT_EQUAL(true, IsValidIPv6("2001:0db8:0000:0042:0000:8a2e:0370:7334"));
            UNIT_ASSERT_EQUAL(true, IsValidIPv6("fe80::50a6:87ff:fe3a:1bea%awdl0"));
            UNIT_ASSERT_EQUAL(true, IsValidIPv6("fe80::feaa:14ff:fea7:67ba"));
            UNIT_ASSERT_EQUAL(true, IsValidIPv6("::1"));
            UNIT_ASSERT_EQUAL(true, IsValidIPv6("2001:db8::ff00:42:8329"));
            UNIT_ASSERT_EQUAL(true, IsValidIPv6("2001:db8:0:0:0:ff00:42:8329"));
            UNIT_ASSERT_EQUAL(true, IsValidIPv6("2001:db8::ff00:42:8329"));

            UNIT_ASSERT_EQUAL(false, IsValidIPv6(""));
            UNIT_ASSERT_EQUAL(false, IsValidIPv6("127.0.0.1"));
            UNIT_ASSERT_EQUAL(false, IsValidIPv6("fe80"));
            UNIT_ASSERT_EQUAL(false, IsValidIPv6("fe80f"));
            UNIT_ASSERT_EQUAL(false, IsValidIPv6(":fe80"));
        }

        void IsValidIPv4forV6Test() {
            UNIT_ASSERT_EQUAL(true, IsValidIPv6("::10.1.1.1"));
            UNIT_ASSERT_EQUAL(true, IsValidIPv6("::ffff:10.1.1.1"));

            UNIT_ASSERT_EQUAL(false, IsValidIPv6(":ffff::10.1.1."));
            UNIT_ASSERT_EQUAL(false, IsValidIPv6(":ffff::10.1.1"));
            UNIT_ASSERT_EQUAL(false, IsValidIPv6(":ffff::10.1."));
            UNIT_ASSERT_EQUAL(false, IsValidIPv6(":ffff::10.1"));
            UNIT_ASSERT_EQUAL(false, IsValidIPv6(":ffff::10"));
            UNIT_ASSERT_EQUAL(false, IsValidIPv6(":ffff::1"));
        }
    };

    UNIT_TEST_SUITE_REGISTRATION(TNetliba_v12UdpAddressTest);
}
