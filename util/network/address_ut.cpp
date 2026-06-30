#include <library/cpp/testing/unittest/registar.h>

#include "address.h"

using namespace NAddr;

Y_UNIT_TEST_SUITE(IRemoteAddr_ToString) {
    Y_UNIT_TEST(Raw) {
        THolder<TOpaqueAddr> opaque(new TOpaqueAddr);
        IRemoteAddr* addr = opaque.Get();

        TString s = ToString(*addr);
        UNIT_ASSERT_VALUES_EQUAL("(raw all zeros)", s);

        opaque->MutableAddr()->sa_data[10] = 17;

        TString t = ToString(*addr);

        UNIT_ASSERT_C(t.StartsWith("(raw 0 0"), t);
        UNIT_ASSERT_C(t.EndsWith(')'), t);
    }

    Y_UNIT_TEST(Ipv6) {
        TNetworkAddress address("::1", 22);
        TNetworkAddress::TIterator it = address.Begin();
        UNIT_ASSERT(it != address.End());
        UNIT_ASSERT(it->ai_family == AF_INET6);
        TString toString = ToString((const IRemoteAddr&)TAddrInfo(&*it));
        UNIT_ASSERT_VALUES_EQUAL(TString("[::1]:22"), toString);
    }

    Y_UNIT_TEST(Loopback) {
        TNetworkAddress localAddress("127.70.0.1", 22);
        UNIT_ASSERT_VALUES_EQUAL(NAddr::IsLoopback(TAddrInfo(&*localAddress.Begin())), true);

        TNetworkAddress localAddress2("127.0.0.1", 22);
        UNIT_ASSERT_VALUES_EQUAL(NAddr::IsLoopback(TAddrInfo(&*localAddress2.Begin())), true);
    }
} // Y_UNIT_TEST_SUITE(IRemoteAddr_ToString)
