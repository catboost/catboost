#include <library/cpp/yson/parser.h>
#include <library/cpp/yson/writer.h>

#include <library/cpp/testing/gtest/gtest.h>

#include <util/stream/mem.h>

using namespace NYson;

TEST(TTestYson, YT_17658)
{
   const auto data = TStringBuf{"\x01\xcc\xcc\xcc\xcc\xcc\xcc\xcc\xcc\xcc"};

    auto input = TMemoryInput{data};

    TStringStream out;
    {
        TYsonWriter writer(&out, EYsonFormat::Text);
        EXPECT_THROW(ParseYsonStringBuffer(data, &writer), std::exception);
    }
}
