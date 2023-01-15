#include <library/cpp/testing/unittest/registar.h>
#include <library/cpp/dns/cache.h>
#include <util/network/address.h>

Y_UNIT_TEST_SUITE(TestDNS) {
    using namespace NDns;

    Y_UNIT_TEST(TestMagic) {
        UNIT_ASSERT_EXCEPTION(CachedThrResolve(TResolveInfo("?", 80)), yexception);
    }

    Y_UNIT_TEST(TestAsteriskAlias) {
        AddHostAlias("*", "localhost");
        const TResolvedHost* rh = CachedThrResolve(TResolveInfo("yandex.ru", 80));
        UNIT_ASSERT(rh != nullptr);

        const TNetworkAddress& addr = rh->Addr;
        for (TNetworkAddress::TIterator ai = addr.Begin(); ai != addr.End(); ai++) {
            if (ai->ai_family == AF_INET || ai->ai_family == AF_INET6) {
                NAddr::TAddrInfo info(&*ai);
                UNIT_ASSERT(IsLoopback(info));
            }
        }
    }
}
