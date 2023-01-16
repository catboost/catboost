#include <yt/yt/core/test_framework/framework.h>

#include <yt/yt/core/misc/farm_hash.h>

namespace NYT {
namespace {

/////////////////////////////////////////////////////////////////////////////
/*
 * NB: This provides a sanity check for stability
 * of FarmHash and FarmFingerprint functions.
 */

TEST(TFarmHashTest, Test)
{
    // NB: At the time of creation of this test, FarmHash function relied on
    // FarmFingerprint, however we do not require it, so this test is more or less a sanity check.
    static_assert(std::is_same<ui64, decltype(FarmHash(42ULL))>::value);
    EXPECT_EQ(17355217915646310598ULL, FarmHash(42ULL));
}

TEST(TFarmFingerprintTest, Test)
{
    static_assert(std::is_same<ui64, decltype(FarmFingerprint(42ULL))>::value);
    EXPECT_EQ(17355217915646310598ULL, FarmFingerprint(42ULL));

    TString buf = "MDwhat?";

    static_assert(std::is_same<ui64, decltype(FarmFingerprint(buf.Data(), buf.Size()))>::value);
    EXPECT_EQ(10997514911242581312ULL, FarmFingerprint(buf.Data(), buf.Size()));

    static_assert(std::is_same<ui64, decltype(FarmFingerprint(1234ULL, 5678ULL))>::value);
    EXPECT_EQ(16769064555670434975ULL, FarmFingerprint(1234ULL, 5678ULL));
}

/////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT
