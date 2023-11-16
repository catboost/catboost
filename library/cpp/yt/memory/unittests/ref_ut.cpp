#include <library/cpp/testing/gtest/gtest.h>

#include <library/cpp/testing/gtest_extensions/assertions.h>

#include <library/cpp/yt/memory/ref.h>

namespace NYT::NYson {
namespace {

////////////////////////////////////////////////////////////////////////////////

TEST(TSharedRefTest, Save)
{
    const TSharedRef expected = TSharedRef::FromString("My tests data");
    TStringStream s;
    ::Save(&s, expected);  // only Save supported for TSharedRef. You can ::Load serialized data to vector.
}

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT::NYson
