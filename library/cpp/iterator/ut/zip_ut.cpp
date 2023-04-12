#include <library/cpp/iterator/zip.h>

#include <library/cpp/testing/gtest/gtest.h>

#include <util/generic/vector.h>

TEST(TIterator, ZipSimplePostIncrement) {
    TVector<int> left{1, 2, 3};
    TVector<int> right{4, 5, 6};

    auto zipped = Zip(left, right);
    auto cur = zipped.begin();
    auto last = zipped.end();

    {
        auto first = *(cur++);
        EXPECT_EQ(std::get<0>(first), 1);
        EXPECT_EQ(std::get<1>(first), 4);
    }
    {
        auto second = *(cur++);
        EXPECT_EQ(std::get<0>(second), 2);
        EXPECT_EQ(std::get<1>(second), 5);
    }

    EXPECT_EQ(std::next(cur), last);
}

