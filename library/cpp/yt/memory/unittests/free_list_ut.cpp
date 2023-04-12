#include <library/cpp/testing/gtest/gtest.h>

#include <library/cpp/yt/memory/free_list.h>

namespace NYT {
namespace {

////////////////////////////////////////////////////////////////////////////////

TEST(TFreeListTest, CompareAndSet)
{
    TAtomicUint128 v = 0;
    ui64 p1 = 0;
    ui64 p2 = 0;
    EXPECT_TRUE(NYT::NDetail::CompareAndSet(&v, p1, p2, ui64{13}, ui64{9}));
    EXPECT_FALSE(NYT::NDetail::CompareAndSet(&v, p1, p2, ui64{100}, ui64{500}));
    EXPECT_EQ(13u, p1);
    EXPECT_EQ(9u, p2);
    EXPECT_TRUE(NYT::NDetail::CompareAndSet(&v, p1, p2, ui64{100}, ui64{500}));
    EXPECT_EQ(TAtomicUint128{500} << 64 | 100, v);
}

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT
