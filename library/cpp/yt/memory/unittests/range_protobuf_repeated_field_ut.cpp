#include <library/cpp/testing/gtest/gtest.h>

#include <library/cpp/testing/gtest_extensions/assertions.h>

#include <library/cpp/yt/memory/range.h>
#include <vector>

namespace {

////////////////////////////////////////////////////////////////////////////////

/*
    Test build
    https://a.yandex-team.ru/arcadia/yt/yt/library/backtrace_introspector/introspect.cpp?rev=r14560536#L165
*/

TEST(TRangeVector, ImmutableRange)
{
    std::vector<const void*> backtrace;
    [[maybe_unused]] auto range = NYT::TRange(backtrace);
}

TEST(TRangeVector, MutableRange)
{
    std::vector<const void*> backtrace;
    [[maybe_unused]] auto range = NYT::TMutableRange(backtrace);
}

////////////////////////////////////////////////////////////////////////////////

/*
    Test build
    https://a.yandex-team.ru/arcadia/yt/yt/library/oom/oom.cpp?rev=r14560536#L112
*/

static const char* TDummyStringArray[] = {
    "xxx",
    "yyy",
    "zzz",
};

TEST(TRangeArrayOfStrings, ImmutableRange)
{
    [[maybe_unused]] auto range = NYT::TRange(TDummyStringArray);
}

TEST(TRangeArrayOfStrings, MutableRange)
{
    [[maybe_unused]] auto range = NYT::TMutableRange(TDummyStringArray);
}

////////////////////////////////////////////////////////////////////////////////

/*
    Test build
    https://a.yandex-team.ru/arcadia/yt/yt/orm/server/access_control/object_cluster.cpp?rev=r14560536#L188
*/

TEST(TRangeRawPtr, ImmutableRange)
{
    struct TDummyData{};
    TDummyData** ptr = nullptr;
    constexpr int size = 0x10;
    [[maybe_unused]] auto range = NYT::TRange<TDummyData*>(ptr, size);
}

TEST(TRangeRawPtr, MutableRange)
{
    struct TDummyData{};
    TDummyData** ptr = nullptr;
    constexpr int size = 0x10;
    [[maybe_unused]] auto range = NYT::TMutableRange<TDummyData*>(ptr, size);
}

////////////////////////////////////////////////////////////////////////////////
} // namespace
