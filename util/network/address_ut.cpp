#include <library/unittest/registar.h>

#include "address.h"

using namespace NAddr;

SIMPLE_UNIT_TEST_SUITE(IRemoteAddr_ToString) {
    SIMPLE_UNIT_TEST(Raw) {
        THolder<TOpaqueAddr> opaque(new TOpaqueAddr);
        IRemoteAddr* addr = opaque.Get();

        TString s = ToString(*addr);
        UNIT_ASSERT_VALUES_EQUAL("(raw all zeros)", s);

        opaque->MutableAddr()->sa_data[10] = 17;

        TString t = ToString(*addr);

        UNIT_ASSERT_C(t.StartsWith("(raw 0 0"), t);
        UNIT_ASSERT_C(t.EndsWith(')'), t);
    }

    SIMPLE_UNIT_TEST(Ipv6) {
        TNetworkAddress address("::1", 22);
        TNetworkAddress::TIterator it = address.Begin();
        UNIT_ASSERT(it != address.End());
        UNIT_ASSERT(it->ai_family == AF_INET6);
        TString toString = ToString((const IRemoteAddr&)TAddrInfo(&*it));
        UNIT_ASSERT_VALUES_EQUAL(TString("[::1]:22"), toString);
    }
}
