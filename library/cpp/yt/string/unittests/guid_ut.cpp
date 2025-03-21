#include <library/cpp/testing/gtest/gtest.h>

#include <library/cpp/yt/string/guid.h>
#include <library/cpp/yt/string/format.h>

#include <util/string/hex.h>

namespace NYT {
namespace {

////////////////////////////////////////////////////////////////////////////////

static_assert(CFormattable<TGuid>);

std::string CanonicalToString(TGuid value)
{
   return Sprintf("%x-%x-%x-%x",
        value.Parts32[3],
        value.Parts32[2],
        value.Parts32[1],
        value.Parts32[0]);
}

const ui32 TrickyValues[] = {
    0, 0x1, 0x12, 0x123, 0x1234, 0x12345, 0x123456, 0x1234567, 0x12345678
};

TEST(TGuidTest, FormatAllTricky)
{
    for (ui32 a : TrickyValues) {
        for (ui32 b : TrickyValues) {
            for (ui32 c : TrickyValues) {
                for (ui32 d : TrickyValues) {
                    auto value = TGuid(a, b, c, d);
                    EXPECT_EQ(CanonicalToString(value), ToString(value));
                }
            }
        }
    }
}

TEST(TGuidTest, FormatAllSymbols)
{
    const auto Value = TGuid::FromString("12345678-abcdef01-12345678-abcdef01");
    EXPECT_EQ(CanonicalToString(Value), ToString(Value));
}

TEST(TGuidTest, ByteOrder)
{
    auto guid = TGuid::FromStringHex32("12345678ABCDEF0112345678ABCDEF01");
    std::string bytes{reinterpret_cast<const char*>(&(guid.Parts32[0])), 16};
    EXPECT_EQ(HexEncode(bytes), "01EFCDAB7856341201EFCDAB78563412");
}

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT
