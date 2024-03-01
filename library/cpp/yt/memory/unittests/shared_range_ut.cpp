#include <library/cpp/testing/gtest/gtest.h>

#include <library/cpp/yt/memory/new.h>
#include <library/cpp/yt/memory/shared_range.h>

namespace NYT {
namespace {

////////////////////////////////////////////////////////////////////////////////

DECLARE_REFCOUNTED_STRUCT(THolder);

struct THolder
    : public TRefCounted
{
    int Value;

    THolder(int value)
        : Value(value)
    { }
};

DEFINE_REFCOUNTED_TYPE(THolder);

////////////////////////////////////////////////////////////////////////////////

TEST(TSharedRange, Move)
{
    auto holder = New<THolder>(0);
    int* raw = &holder->Value;

    auto sharedRange = MakeSharedRange<int>({raw, 1}, std::move(holder));

    {
        auto sharedRangeMovedTo = std::move(sharedRange);
    }

    EXPECT_TRUE(sharedRange.ToVector().empty());
}

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT
