#include <yt/yt/core/test_framework/framework.h>

#include <library/cpp/yt/assert/assert.h>

namespace NYT {
namespace {

////////////////////////////////////////////////////////////////////////////////

TEST(TAssertTest, Abort)
{
    EXPECT_DEATH(
        {
            YT_ABORT();
        },
        "YT_ABORT");

    EXPECT_DEATH(
        {
            YT_ABORT("MY_PROBLEM");
        },
        "MY_PROBLEM");
}

TEST(TAssertTest, Verify)
{
    YT_VERIFY(true);
    YT_VERIFY(true, "12345");
    YT_VERIFY(true, [] {
        YT_ABORT();
        return "NEVER_PRINTED";
    } ());
    EXPECT_DEATH(
        {
            YT_VERIFY(false);
        },
        "YT_VERIFY");
    EXPECT_DEATH(
        {
            YT_VERIFY(false, "MY_MESSAGE");
        },
        "YT_VERIFY.*MY_MESSAGE");
    EXPECT_DEATH(
        {
            YT_VERIFY(false && "my_specific_expression_part", "MY_MESSAGE");
        },
        "false && \"my_specific_expression_part\"");
}

TEST(TAssertTest, Assert)
{
    YT_ASSERT(true);
    YT_ASSERT(true, [] {
        YT_ABORT();
        return "NEVER_PRINTED";
    } ());
    #ifndef NDEBUG
    EXPECT_DEATH(
        {
            YT_ASSERT(false);
        },
        "YT_ASSERT");
    EXPECT_DEATH(
        {
            YT_ASSERT(false, "MY_MESSAGE");
        },
        "MY_MESSAGE");
    EXPECT_DEATH(
        {
            YT_ASSERT(false && "my_specific_expression_part", "MY_MESSAGE");
        },
        "my_specific_expression_part");
    #endif
}

TEST(TAssertTest, Unimplemented)
{
    EXPECT_DEATH(
        {
            YT_UNIMPLEMENTED();
        },
        "YT_UNIMPLEMENTED");
}

TEST(TAssertTest, AcceptString)
{
    EXPECT_DEATH(
        {
            YT_ABORT(std::string("MY_PROBLEM"));
        },
        "MY_PROBLEM");
}


////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT
