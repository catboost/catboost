#include "endpoint.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/generic/hash_set.h>
#include <util/generic/strbuf.h>

Y_UNIT_TEST_SUITE(TEndpointTest) {
    Y_UNIT_TEST(TestSimple) {
        TVector<TNetworkAddress> addrs;

        TEndpoint ep0;

        UNIT_ASSERT(ep0.IsIpV4());
        UNIT_ASSERT_VALUES_EQUAL(0, ep0.Port());
        UNIT_ASSERT_VALUES_EQUAL("0.0.0.0", ep0.IpToString());

        TEndpoint ep1;

        try {
            TNetworkAddress na1("25.26.27.28", 24242);

            addrs.push_back(na1);

            ep1 = TEndpoint(new NAddr::TAddrInfo(&*na1.Begin()));

            UNIT_ASSERT(ep1.IsIpV4());
            UNIT_ASSERT_VALUES_EQUAL("25.26.27.28", ep1.IpToString());
            UNIT_ASSERT_VALUES_EQUAL(24242, ep1.Port());
        } catch (const TNetworkResolutionError&) {
            TNetworkAddress n("2a02:6b8:0:1420:0::5f6c:f3c2", 11111);

            addrs.push_back(n);

            ep1 = TEndpoint(new NAddr::TAddrInfo(&*n.Begin()));
        }

        ep0.SetPort(12345);

        TEndpoint ep2(ep0);

        ep0.SetPort(0);

        UNIT_ASSERT_VALUES_EQUAL(12345, ep2.Port());

        TEndpoint ep2_;

        ep2_.SetPort(12345);

        UNIT_ASSERT(ep2 == ep2_);

        TNetworkAddress na3("2a02:6b8:0:1410::5f6c:f3c2", 54321);
        TEndpoint ep3(new NAddr::TAddrInfo(&*na3.Begin()));

        UNIT_ASSERT(ep3.IsIpV6());
        UNIT_ASSERT(ep3.IpToString().StartsWith(TStringBuf("2a02:6b8:0:1410:")));
        UNIT_ASSERT(ep3.IpToString().EndsWith(TStringBuf(":5f6c:f3c2")));
        UNIT_ASSERT_VALUES_EQUAL(54321, ep3.Port());

        TNetworkAddress na4("2a02:6b8:0:1410:0::5f6c:f3c2", 1);
        TEndpoint ep4(new NAddr::TAddrInfo(&*na4.Begin()));

        TEndpoint ep3_ = ep4;

        ep3_.SetPort(54321);

        THashSet<TEndpoint> he;

        he.insert(ep0);
        he.insert(ep1);
        he.insert(ep2);

        UNIT_ASSERT_VALUES_EQUAL(3u, he.size());

        he.insert(ep2_);

        UNIT_ASSERT_VALUES_EQUAL(3u, he.size());

        he.insert(ep3);
        he.insert(ep3_);

        UNIT_ASSERT_VALUES_EQUAL(4u, he.size());

        he.insert(ep4);

        UNIT_ASSERT_VALUES_EQUAL(5u, he.size());
    }

    Y_UNIT_TEST(TestEqual) {
        const TString ip1 = "2a02:6b8:0:1410::5f6c:f3c2";
        const TString ip2 = "2a02:6b8:0:1410::5f6c:f3c3";

        TNetworkAddress na1(ip1, 24242);
        TEndpoint ep1(new NAddr::TAddrInfo(&*na1.Begin()));

        TNetworkAddress na2(ip1, 24242);
        TEndpoint ep2(new NAddr::TAddrInfo(&*na2.Begin()));

        TNetworkAddress na3(ip2, 24242);
        TEndpoint ep3(new NAddr::TAddrInfo(&*na3.Begin()));

        TNetworkAddress na4(ip2, 24243);
        TEndpoint ep4(new NAddr::TAddrInfo(&*na4.Begin()));

        UNIT_ASSERT(ep1 == ep2);
        UNIT_ASSERT(!(ep1 == ep3));
        UNIT_ASSERT(!(ep1 == ep4));
    }

    Y_UNIT_TEST(TestIsUnixSocket) {
        TNetworkAddress na1(TUnixSocketPath("/tmp/unixsocket"));
        TEndpoint ep1(new NAddr::TAddrInfo(&*na1.Begin()));

        TNetworkAddress na2("2a02:6b8:0:1410::5f6c:f3c2", 24242);
        TEndpoint ep2(new NAddr::TAddrInfo(&*na2.Begin()));

        UNIT_ASSERT(ep1.IsUnix());
        UNIT_ASSERT(ep1.SockAddr()->sa_family == AF_UNIX);

        UNIT_ASSERT(!ep2.IsUnix());
        UNIT_ASSERT(ep2.SockAddr()->sa_family != AF_UNIX);
    }
} // Y_UNIT_TEST_SUITE(TEndpointTest)
