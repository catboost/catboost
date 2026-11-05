#include <library/cpp/testing/gtest/gtest.h>

#include <library/cpp/yt/system/local_host.h>

#include <string>

namespace NYT {
namespace {

////////////////////////////////////////////////////////////////////////////////

TEST(TInternHostNameTest, Simple)
{
    auto hostName = std::string("m001-hahn.sas.yp-c.yandex.net");
    auto interned = InternHostName(hostName);
    EXPECT_EQ(interned, hostName);
    EXPECT_NE(interned.data(), hostName.data());
}

TEST(TInternHostNameTest, SameStorage)
{
    auto interned1 = InternHostName(std::string("m002-hahn.sas.yp-c.yandex.net"));
    auto interned2 = InternHostName(std::string("m002-hahn.sas.yp-c.yandex.net"));
    EXPECT_EQ(interned1.data(), interned2.data());

    auto interned3 = InternHostName(std::string("m003-hahn.sas.yp-c.yandex.net"));
    EXPECT_NE(interned1.data(), interned3.data());
}

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT
